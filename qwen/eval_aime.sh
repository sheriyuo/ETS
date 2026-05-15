accelerate launch eval_qwen.py --tasks aime24 --model qwen --model_args model_path='Qwen/Qwen3-8B',max_length=6144

accelerate launch eval_qwen.py --tasks aime24 --model qwen-beam --model_args model_path='Qwen/Qwen3-8B',max_length=6144,num_beams=50

accelerate launch eval_qwen.py --tasks aime24 --model qwen-ets \
    --model_args dataset='aime',model_path='Qwen/Qwen3-8B',m_candidates=20,k_monte_carlo=1,block_size=6144,max_length=6144,temperature=0.7

accelerate launch eval_qwen.py --tasks aime24 --model qwen-ets \
    --model_args dataset='aime',model_path='Qwen/Qwen3-8B',m_candidates=5,k_monte_carlo=3,block_size=2048,max_length=6144,temperature=0.7

accelerate launch eval_qwen.py --tasks aime24 --model qwen-ets \
    --model_args dataset='aime',model_path='Qwen/Qwen3-8B',m_candidates=5,k_monte_carlo=3,block_size=1536,max_length=6144,temperature=0.7,small_model_path='Qwen/Qwen3-1.7B',use_importance_sampling=True