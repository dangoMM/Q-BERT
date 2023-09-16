.PHONY: train
train:
	cd qbert/extraction && \
	pdm run python -m torch.distributed.launch --nproc-per-node=2 run_squad.py\
	 --model_type albert --model_name_or_path albert-large-v2 --do_train  --train_file data/cleaned_qa_train.json\
	 --predict_file data/cleaned_qa_dev.json --per_gpu_eval_batch_size 8 --learning_rate 3e-5 --max_seq_length 512\
	 --doc_stride 128 --output_dir ./models/ --warmup_steps 814 --max_steps 8144 --version_2_with_negative\
	 --gradient_accumulation_steps 24 --overwrite_output_dir