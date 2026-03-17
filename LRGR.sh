# run the application
#nohup python main_domain.py --dataset seq-cifar10 --model LEAR --lr 0.03 --batch_size 32 --n_epochs 10 --num_workers 0 --backbone lear > LEAR.out 2>&1 &
python main_domain.py --dataset seq-cifar10 --model LEAR --lr 0.03 --batch_size 32 --n_epochs 5 --num_workers 4 --backbone lear \
	--enable_local_hybrid_rematch \
	--local_rematch_top_k 2 \
	--local_rematch_confidence_threshold 0.5 \
	--enable_global_ctird \
	--global_ctird_weight 0.1 \
	--global_feature_distill_weight 1.0
