# Base
accelerate launch eval_qwen.py --tasks minerva_math500 --model qwen --model_args model_path='Qwen/Qwen3-8B',max_length=512
accelerate launch eval_qwen.py --tasks gsm8k --model qwen --model_args model_path='Qwen/Qwen3-8B',max_length=512
accelerate launch eval_qwen.py --tasks gpqa_diamond_generative_n_shot --num_fewshot 5 --model qwen --model_args model_path='Qwen/Qwen3-8B',max_length=512
HF_ALLOW_CODE_EVAL=1 accelerate launch eval_qwen.py --tasks humaneval --model qwen --model_args model_path='Qwen/Qwen3-8B',max_length=512 --confirm_run_unsafe_code

# Beam Search
accelerate launch eval_qwen.py --tasks minerva_math500 --model qwen-beam \ 
    --model_args model_path='Qwen/Qwen3-8B',max_length=512,num_beams=20
accelerate launch eval_qwen.py --tasks gsm8k --model qwen-beam \ 
    --model_args model_path='Qwen/Qwen3-8B',max_length=512,num_beams=20
accelerate launch eval_qwen.py --tasks gpqa_diamond_generative_n_shot --num_fewshot 5 --model qwen-beam \ 
    --model_args model_path='Qwen/Qwen3-8B',max_length=512,num_beams=50
HF_ALLOW_CODE_EVAL=1 accelerate launch eval_qwen.py --tasks humaneval --model qwen-beam --confirm_run_unsafe_code \ 
    --model_args model_path='Qwen/Qwen3-8B',max_length=512,num_beams=20

# ETS
accelerate launch eval_qwen.py --tasks minerva_math500 --model qwen-ets \
    --model_args dataset='math500',model_path='Qwen/Qwen3-8B',m_candidates=15,k_monte_carlo=3,block_size=64,max_length=512,temperature=0.7
accelerate launch eval_qwen.py --tasks gsm8k --model qwen-ets \
    --model_args dataset='gsm8k',model_path='Qwen/Qwen3-8B',m_candidates=15,k_monte_carlo=3,block_size=128,max_length=512,temperature=1.5
accelerate launch eval_qwen.py --tasks gpqa_diamond_generative_n_shot --num_fewshot 5 --model qwen-ets \
    --model_args dataset='gpqa',model_path='Qwen/Qwen3-8B',m_candidates=15,k_monte_carlo=3,block_size=64,max_length=512,temperature=0.25
HF_ALLOW_CODE_EVAL=1 accelerate launch eval_qwen.py --tasks humaneval --model qwen-ets --confirm_run_unsafe_code \ 
    --model_args dataset='humaneval',model_path='Qwen/Qwen3-8B',m_candidates=20,k_monte_carlo=3,block_size=64,max_length=512,temperature=0.25

# ETS-IS
accelerate launch eval_qwen.py --tasks minerva_math500 --model qwen-ets \
    --model_args dataset='math500',model_path='Qwen/Qwen3-8B',m_candidates=15,k_monte_carlo=3,block_size=128,max_length=512,temperature=0.7,small_model_path='Qwen/Qwen3-1.7B',use_importance_sampling=True
accelerate launch eval_qwen.py --tasks gsm8k --model qwen-ets \
    --model_args dataset='gsm8k',model_path='Qwen/Qwen3-8B',m_candidates=15,k_monte_carlo=3,block_size=128,max_length=512,temperature=1.5,small_model_path='Qwen/Qwen3-1.7B',use_importance_sampling=True
accelerate launch eval_qwen.py --tasks gpqa_diamond_generative_n_shot --num_fewshot 5 --model qwen-ets \
    --model_args dataset='gpqa',model_path='Qwen/Qwen3-8B',m_candidates=15,k_monte_carlo=3,block_size=64,max_length=512,temperature=0.25,small_model_path='Qwen/Qwen3-1.7B',use_importance_sampling=True
HF_ALLOW_CODE_EVAL=1 accelerate launch eval_qwen.py --tasks humaneval --model qwen-ets --confirm_run_unsafe_code \ 
    --model_args dataset='humaneval',model_path='Qwen/Qwen3-8B',m_candidates=10,k_monte_carlo=3,block_size=64,max_length=512,temperature=0.25,small_model_path='Qwen/Qwen3-1.7B',use_importance_sampling=True