accelerate launch eval_llada.py --tasks gsm8k --model llada_dist --model_args \
model_path='GSAI-ML/LLaDA-8B-Instruct',gen_length=256,steps=256,block_length=8,temperature=1.0,\
guide_steps=4,use_vanilla=True,num_candidates=15,monte_carlo_num=3,energy_weight=0.1,task="gsm8k",accumulate_x0=True

accelerate launch eval_llada.py --tasks minerva_math500 --model llada_dist --model_args \
model_path='GSAI-ML/LLaDA-8B-Instruct',gen_length=256,steps=256,block_length=32,temperature=0.5,\
guide_steps=4,use_vanilla=True,num_candidates=15,monte_carlo_num=3,energy_weight=0.1,task="math500",accumulate_x0=True

accelerate launch eval_llada.py --tasks humaneval --confirm_run_unsafe_code --model llada_dist --model_args \
model_path='GSAI-ML/LLaDA-8B-Instruct',gen_length=512,steps=512,block_length=32,temperature=0.5,\
guide_steps=4,use_vanilla=True,num_candidates=15,monte_carlo_num=3,energy_weight=0.1,task='humaneval',accumulate_x0=True 

accelerate launch eval_llada.py --tasks gpqa_diamond_generative_n_shot --num_fewshot 5 --model llada_dist --model_args \
model_path='GSAI-ML/LLaDA-8B-Instruct',gen_length=64,steps=64,block_length=8,temperature=0.5,\
guide_steps=4,use_vanilla=True,num_candidates=15,monte_carlo_num=3,energy_weight=0.1,task='gpqa',accumulate_x0=True 

accelerate launch eval_llada.py --tasks gsm8k --model llada_dist --model_args \
model_path='GSAI-ML/LLaDA-8B-Instruct',gen_length=256,steps=256,block_length=8,temperature=1.0,\
guide_steps=4,is_accelerate_available=True,num_candidates=15,monte_carlo_num=3,energy_weight=0.1,task="gsm8k",accumulate_x0=True

accelerate launch eval_llada.py --tasks minerva_math500 --model llada_dist --model_args \
model_path='GSAI-ML/LLaDA-8B-Instruct',gen_length=256,steps=256,block_length=32,temperature=0.5,\
guide_steps=4,is_accelerate_available=True,num_candidates=15,monte_carlo_num=3,energy_weight=0.1,task="math500",accumulate_x0=True

accelerate launch eval_llada.py --tasks humaneval --confirm_run_unsafe_code --model llada_dist --model_args \
model_path='GSAI-ML/LLaDA-8B-Instruct',gen_length=512,steps=512,block_length=32,temperature=0.5,\
guide_steps=4,is_accelerate_available=True,num_candidates=15,monte_carlo_num=3,energy_weight=0.1,task='humaneval',accumulate_x0=True

accelerate launch eval_llada.py --tasks gpqa_diamond_generative_n_shot --num_fewshot 5 --model llada_dist --model_args \
model_path='GSAI-ML/LLaDA-8B-Instruct',gen_length=64,steps=64,block_length=8,temperature=0.5,\
guide_steps=4,is_accelerate_available=True,num_candidates=15,monte_carlo_num=3,energy_weight=0.1,task='gpqa',accumulate_x0=True