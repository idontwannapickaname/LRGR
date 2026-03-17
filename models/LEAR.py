# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn
import os
from datasets import ContinualDataset
from models.utils.continual_model import ContinualModel
from utils.args import ArgumentParser
import torch
import numpy as np
from collections import Counter
from tqdm import tqdm
from backbone.utils.hsic import hsic
import torch.nn.functional as F
import random
import copy
from torch.distributions import MultivariateNormal
from scipy.stats import multivariate_normal


class LEAR(ContinualModel):
    NAME = 'LEAR'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        parser.add_argument('--enable_local_hybrid_rematch', action='store_true',
                            help='Enable evaluation-time hybrid rematching for the local backbone.')
        parser.add_argument('--local_rematch_top_k', type=int, default=2,
                            help='Number of ranked local experts to consider for rematching.')
        parser.add_argument('--local_rematch_confidence_threshold', type=float, default=0.5,
                            help='Confidence threshold for second-stage local rematching.')
        parser.add_argument('--enable_global_ctird', action='store_true',
                            help='Enable CTIRD on the global backbone features.')
        parser.add_argument('--global_ctird_weight', type=float, default=0.1,
                            help='Weight for the global CTIRD loss.')
        parser.add_argument('--disable_global_feature_distill', action='store_true',
                            help='Disable the default global feature-space distillation loss.')
        parser.add_argument('--global_feature_distill_weight', type=float, default=1.0,
                            help='Weight for the global feature-space distillation loss.')
        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        super(LEAR, self).__init__(backbone, loss, args, transform, dataset=dataset)

        self.train_loader_size = None
        self.iter = 0
        self.last_rematch_stats = {
            'direct_candidates': 0,
            'direct_applied': 0,
            'confidence_candidates': 0,
            'confidence_applied': 0,
        }

    def _extract_local_reference_features(self, x):
        processX = self.net.vitProcess(x)

        features = self.net.local_vitmodel.patch_embed(processX)
        cls_token = self.net.local_vitmodel.cls_token.expand(features.shape[0], -1, -1)
        features = torch.cat((cls_token, features), dim=1)
        features = features + self.net.local_vitmodel.pos_embed

        for block in self.net.local_vitmodel.blocks[:-3]:
            features = block(features)

        features = self.net.Forever_freezed_blocks(features)
        features = self.net.local_vitmodel.norm(features)
        return features[:, 0, :]

    def _compute_expert_distances(self, class_token):
        if len(self.net.fcArr) == 0:
            return torch.empty(class_token.shape[0], 0, device=class_token.device)

        distances = []
        for fc, dist in zip(self.net.fcArr, self.net.distributions):
            fc_feature = fc(class_token)
            delta = fc_feature - dist.mean.unsqueeze(0)
            inv_cov = torch.linalg.pinv(dist.covariance_matrix)
            mahalanobis = torch.einsum('bi,ij,bj->b', delta, inv_cov, delta)
            distances.append(torch.sqrt(mahalanobis.clamp_min(1e-12)))

        return torch.stack(distances, dim=1)

    def end_task(self, dataset) -> None:
        #calculate distribution
        train_loader = dataset.train_loader
        num_choose = 100
        with torch.no_grad():
            train_iter = iter(train_loader)

            pbar = tqdm(train_iter, total=num_choose,
                        desc=f"Calculate distribution for task {self.current_task + 1}",
                        disable=False, mininterval=0.5)

            fc_features_list = []

            count = 0
            while count < num_choose:
                try:
                    data = next(train_iter)
                except StopIteration:
                    break

                x = data[0]
                x = x.to(self.device)

                class_token = self._extract_local_reference_features(x)

                fc_features_list.append(self.net.fcArr[self.current_task](class_token))

                count += 1
                pbar.update()

            pbar.close()
            fc_features = torch.cat(fc_features_list, dim=0)  # [num*b,fc_size]
            mu = torch.mean(fc_features, dim=0)
            sigma = torch.cov(fc_features.T) + torch.eye(fc_features.shape[1], device=fc_features.device) * 1e-4
            self.net.distributions.append(MultivariateNormal(mu, sigma))

        #deal with grad and blocks
        for fc in self.net.fcArr:
            for param in fc.parameters():
                param.requires_grad = False

        for cls in self.net.classifierArr:
            for param in cls.parameters():
                param.requires_grad = False

        self.net.Freezed_local_blocks = copy.deepcopy(torch.nn.Sequential(*list(self.net.local_vitmodel.blocks[-3:])))
        self.net.Freezed_global_blocks = copy.deepcopy(torch.nn.Sequential(*list(self.net.global_vitmodel.blocks[-3:])))

        for block in self.net.Freezed_local_blocks:
            for param in block.parameters():
                param.requires_grad = False

        for block in self.net.Freezed_global_blocks:
            for param in block.parameters():
                param.requires_grad = False

    def begin_task(self, dataset, threshold=0) -> None:
        train_loader = dataset.train_loader
        if self.current_task > 0:
            num_choose = 50
            with torch.no_grad():
                train_iter = iter(train_loader)
                distances = torch.zeros(len(self.net.fcArr), device=self.device)

                pbar = tqdm(train_iter, total=num_choose,
                            desc=f"Choose params for task {self.current_task + 1}",
                            disable=False, mininterval=0.5)

                count = 0
                while count < num_choose:
                    try:
                        data = next(train_iter)
                    except StopIteration:
                        break

                    x = data[0]
                    x = x.to(self.device)

                    class_token = self._extract_local_reference_features(x)
                    batch_distances = self._compute_expert_distances(class_token)
                    distances += batch_distances.mean(dim=0)

                    count += 1
                    averaged_distances = distances / count
                    bar_log = {'distances': [round(value.item(), 2) for value in averaged_distances]}
                    pbar.set_postfix(bar_log, refresh=False)
                    pbar.update()
                pbar.close()

                min_idx = torch.argmin(distances).item()
                self.net.CreateNewExper(min_idx, dataset.N_CLASSES)

        self.opt = self.get_optimizer()

    def myPrediction(self,x,k):
        with torch.no_grad():
            #Perform the prediction according to the seloeced expert
            out = self.net.myprediction(x,k)
        return out

    def hybrid_rematch_logits(self, inputs, base_expert_idx, n_classes=None):
        logits = self.myPrediction(inputs, base_expert_idx)
        stats = {
            'direct_candidates': 0,
            'direct_applied': 0,
            'confidence_candidates': 0,
            'confidence_applied': 0,
        }

        if not self.args.enable_local_hybrid_rematch or len(self.net.fcArr) < 2:
            self.last_rematch_stats = stats
            return logits, stats

        top_k = max(2, min(self.args.local_rematch_top_k, len(self.net.fcArr)))
        _, ranked_experts = self.cal_expert_dist(inputs, return_per_sample=True, top_k=top_k)
        top1_experts = ranked_experts[:, 0]

        direct_mask = top1_experts != int(base_expert_idx)
        stats['direct_candidates'] = int(direct_mask.sum().item())
        if direct_mask.any():
            logits = logits.clone()
            logits[direct_mask] = self.myPrediction(inputs[direct_mask], top1_experts[direct_mask])
            stats['direct_applied'] = stats['direct_candidates']

        if ranked_experts.shape[1] < 2:
            self.last_rematch_stats = stats
            return logits, stats

        logits_for_confidence = logits[:, :n_classes] if n_classes is not None else logits
        confidence = F.softmax(logits_for_confidence, dim=1).amax(dim=1)
        second_experts = ranked_experts[:, 1]
        active_experts = torch.full_like(top1_experts, int(base_expert_idx))
        active_experts[direct_mask] = top1_experts[direct_mask]

        confidence_mask = confidence < self.args.local_rematch_confidence_threshold
        confidence_mask &= second_experts != active_experts
        stats['confidence_candidates'] = int(confidence_mask.sum().item())

        if confidence_mask.any():
            candidate_logits = self.myPrediction(inputs[confidence_mask], second_experts[confidence_mask])
            current_logits = logits[confidence_mask]
            candidate_conf = F.softmax(
                candidate_logits[:, :n_classes] if n_classes is not None else candidate_logits,
                dim=1,
            ).amax(dim=1)
            current_conf = F.softmax(
                current_logits[:, :n_classes] if n_classes is not None else current_logits,
                dim=1,
            ).amax(dim=1)
            better_mask = candidate_conf > current_conf
            if better_mask.any():
                selected_indices = confidence_mask.nonzero(as_tuple=False).flatten()[better_mask]
                logits[selected_indices] = candidate_logits[better_mask]
                stats['confidence_applied'] = int(better_mask.sum().item())

        self.last_rematch_stats = stats
        return logits, stats

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        l2_distance = torch.nn.MSELoss()

        self.opt.zero_grad()
        if len(self.net.fcArr) > 1:
            outputs, Freezed_global_features, Freezed_local_features, global_features, local_features = self.net(inputs, return_features=True)
            loss_kd = kl_loss(local_features, Freezed_local_features)
            loss_hsic = -hsic(global_features, local_features)
            loss_ce = self.loss(outputs, labels)
            loss_feature = torch.zeros((), device=inputs.device)
            if not self.args.disable_global_feature_distill:
                loss_feature = self.args.global_feature_distill_weight * l2_distance(global_features, Freezed_global_features)

            loss_ctird = torch.zeros((), device=inputs.device)
            if self.args.enable_global_ctird:
                loss_ctird = self.args.global_ctird_weight * ctird_loss(global_features, Freezed_global_features)

            loss_tot = loss_ce + loss_kd + loss_hsic + loss_feature + loss_ctird
            loss_vis = [
                loss_ce.item(),
                loss_kd.item(),
                loss_hsic.item(),
                loss_feature.item(),
                loss_ctird.item(),
            ]
        else:
            outputs, global_features, local_features = self.net(inputs)
            loss_hsic = -hsic(global_features, local_features)
            loss_ce = self.loss(outputs, labels)
            loss_tot = loss_ce + loss_hsic
            loss_vis = [loss_ce.item(), loss_hsic.item()]

        loss_tot.backward()


        self.opt.step()

        return loss_vis

    def cal_expert_dist(self, x, return_per_sample=False, top_k=2):
        class_token = self._extract_local_reference_features(x)
        distances = self._compute_expert_distances(class_token)
        if return_per_sample:
            effective_top_k = max(1, min(top_k, distances.shape[1]))
            ranked_experts = torch.topk(distances, k=effective_top_k, largest=False, dim=1).indices
            return distances, ranked_experts
        return distances.mean(dim=0).tolist()



def kl_loss(student_feat, teacher_feat):
    student_feat = F.normalize(student_feat, p=2, dim=1)
    teacher_feat = F.normalize(teacher_feat, p=2, dim=1)

    student_prob = (student_feat + 1) / 2
    teacher_prob = (teacher_feat.detach() + 1) / 2

    loss_kld = F.kl_div(
        torch.log(student_prob + 1e-10),
        teacher_prob,
        reduction='batchmean'
    )
    return loss_kld


def ctird_loss(student_feat, teacher_feat, eps=1e-8):
    student_feat = F.normalize(student_feat, p=2, dim=1)
    teacher_feat = F.normalize(teacher_feat.detach(), p=2, dim=1)

    student_similarity = torch.mm(student_feat, student_feat.t())
    student_similarity = torch.exp(student_similarity)
    student_similarity = student_similarity / student_similarity.sum(dim=1, keepdim=True).clamp_min(eps)

    teacher_similarity = torch.mm(teacher_feat, teacher_feat.t())
    teacher_similarity = torch.exp(teacher_similarity)
    teacher_similarity = teacher_similarity / teacher_similarity.sum(dim=1, keepdim=True).clamp_min(eps)

    return F.kl_div(torch.log(student_similarity.clamp_min(eps)), teacher_similarity, reduction='batchmean')
