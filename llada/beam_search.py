import torch
import json
import numpy as np
import os
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
import math
import time
from datetime import datetime
from collections import Counter


from transformers import AutoTokenizer, AutoModel
from gsm8k import compute_score
from extract_fn import gsm8k_extract, humaneval_extract, math_extract, parse_answer_gpqa
import random

import matplotlib.pyplot as plt

def cal_log_probs(logits, x0, prompt_length, mask_index):
    log_probs = F.log_softmax(logits, dim=-1)   # [B, L, V]

    token_log_probs = log_probs.gather(
        dim=-1, index=x0.unsqueeze(-1)
    ).squeeze(-1)  # [B, L]
    token_log_probs = token_log_probs[:, prompt_length:]
    print("token_log_probs shape:", token_log_probs.shape)
    log_p = []
    for i in range(token_log_probs.size(0)):
        masked_token_log_probs = token_log_probs[i][mask_index[i][prompt_length:]]
        log_p.append(masked_token_log_probs.sum().item())
    return log_p

def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64) / temperature
    # prob = F.log_softmax(logits, dim=-1)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = -torch.log(-torch.log(noise))
    gumbel_max = logits + gumbel_noise
    return gumbel_max
    # if temperature == 0:
    #     return logits
    # logits = logits.to(torch.float64)
    # noise = torch.rand_like(logits, dtype=torch.float64)
    # gumbel_noise = (- torch.log(noise)) ** temperature
    # return logits.exp() / gumbel_noise

@torch.no_grad()
def gumbel_max_sample(logits, temperature, chunk_size):
    if temperature == 0:
        return torch.argmax(logits, dim=-1)
    B = logits.size(0)
    out = torch.empty((B, logits.size(1)), device=logits.device, dtype=torch.long)
    for i in range(0, B, chunk_size):
        end_pos = min(i + chunk_size, B)
        logits_chunk = logits[i:end_pos]
        logits_chunk = logits_chunk.to(torch.float64) / temperature
        noise = torch.rand_like(logits_chunk, dtype=torch.float64)
        # logits_chunk = logits_chunk.to(torch.float32) / temperature
        # noise = torch.rand_like(logits_chunk, dtype=torch.float32)
        gumbel_noise = -torch.log(-torch.log(noise))
        gumbel_max = logits_chunk + gumbel_noise
        out[i:end_pos] = torch.argmax(gumbel_max, dim=-1)
        del logits_chunk, noise, gumbel_noise, gumbel_max
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    return out

def get_num_transfer_tokens(mask_index, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens

@torch.no_grad()
def generate(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
             cfg_scale=0., remasking='low_confidence', mask_id=126336):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    '''
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    prompt_index = (x != mask_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    for num_block in range(num_blocks):
        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length:] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        for i in range(steps):
            mask_index = (x == mask_id)
            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                logits = model(x_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x).logits

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

            if remasking == 'low_confidence':
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]

    return x

@torch.no_grad()
def generate_beam_search(
    model,
    prompt,                     # (1, Lp)
    gen_length=128,
    block_length=128,
    num_beams=5,
    num_return_sequences=1,
    temperature=1.0,            # beam search 一般不用采样，这里用于 logit scaling（=1 表示不缩放）
    cfg_scale=0.0,
    remasking="low_confidence", # 'low_confidence' or 'random'
    mask_id=126336,
    length_penalty=1.0,         # 和 transformers 对齐（这里长度固定，基本是常数项）
):
    """
    DLM / LLaDA beam search generation under the constraint:
      - steps == gen_length (i.e., transfer exactly 1 token per step)
      - support semi-AR blocks when block_length < gen_length

    Return:
      - sequences: (num_return_sequences, Lp + gen_length)
        (默认返回最优的 num_return_sequences 条)
    """

    device = model.device
    assert prompt.dim() == 2 and prompt.size(0) == 1
    prompt = prompt.to(device)

    Lp = prompt.size(1)
    L = Lp + gen_length

    assert gen_length % block_length == 0, "gen_length must be divisible by block_length"
    num_blocks = gen_length // block_length

    steps_per_block = block_length

    x = torch.full((num_beams, L), mask_id, dtype=torch.long, device=device)
    x[:, :Lp] = prompt.expand(num_beams, -1).clone()

    prompt_index = (x != mask_id)

    beam_scores = torch.zeros((num_beams,), dtype=torch.float, device=device)
    beam_scores[1:] = -1e9

    def forward_logits_with_cfg(x_in):
        if cfg_scale > 0.0:
            un_x = x_in.clone()
            un_x[prompt_index[: x_in.size(0)]] = mask_id  # mask out prompt for unconditional branch
            x_cat = torch.cat([x_in, un_x], dim=0)        # (2B, L)
            logits_cat = model(x_cat).logits              # (2B, L, V)
            logits, un_logits = torch.chunk(logits_cat, 2, dim=0)
            logits = un_logits + (cfg_scale + 1.0) * (logits - un_logits)
            return logits
        else:
            return model(x_in).logits

    for b in range(num_blocks):
        block_end = Lp + (b + 1) * block_length

        for t in range(steps_per_block):
            mask_index = (x == mask_id)                  # (B, L)
            editable_mask = mask_index.clone()
            editable_mask[:, block_end:] = False         # forbid future blocks

            logits = forward_logits_with_cfg(x)

            if temperature is not None and temperature > 0 and temperature != 1.0:
                logits = logits / temperature

            if remasking == "low_confidence":
                x0 = torch.argmax(logits, dim=-1)

                p = F.softmax(logits, dim=-1)
                x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)

                confidence = torch.where(editable_mask, x0_p, torch.full_like(x0_p, -float("inf")))

            elif remasking == "random":
                rand_conf = torch.rand((num_beams, L), device=device)
                confidence = torch.where(editable_mask, rand_conf, torch.full_like(rand_conf, -float("inf")))
            else:
                raise NotImplementedError(remasking)

            # pos: (B,)
            pos = torch.argmax(confidence, dim=-1)

            # log_probs: (B, L, V)
            log_probs = F.log_softmax(logits, dim=-1)

            # lp_i: (B, V)
            beam_ids = torch.arange(num_beams, device=device)
            lp_i = log_probs[beam_ids, pos, :]  # gather per-beam position distribution

            # expand each beam by topk tokens
            # topk_tokens/topk_logp: (B, K) where K=num_beams
            K = num_beams
            topk_logp, topk_tokens = torch.topk(lp_i, k=K, dim=-1)

            # candidate_scores: (B, K)
            candidate_scores = beam_scores.unsqueeze(1) + topk_logp

            flat_scores = candidate_scores.reshape(-1)            # (B*K,)
            next_scores, flat_indices = torch.topk(flat_scores, k=num_beams, dim=-1)

            next_beam_indices = flat_indices // K                 # (B,)
            next_token_indices = flat_indices % K                 # (B,)

            parent_pos = pos[next_beam_indices]                   # (B,)
            next_tokens = topk_tokens[next_beam_indices, next_token_indices]  # (B,)

            x_next = x[next_beam_indices].clone()
            x_next[beam_ids, parent_pos] = next_tokens

            x = x_next
            beam_scores = next_scores

    if length_penalty is not None and length_penalty != 1.0:
        norm = float(gen_length) ** length_penalty
        final_scores = beam_scores / norm
    else:
        final_scores = beam_scores

    # return best num_return_sequences
    num_return_sequences = min(num_return_sequences, num_beams)
    best_scores, best_idx = torch.topk(final_scores, k=num_return_sequences, dim=-1)

    sequences = x[best_idx]  # (num_return_sequences, L)
    return sequences

def main():
    device = 'cuda'

    model = AutoModel.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True, dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)
    # question_raw = "A deep-sea monster rises from the waters once every hundred years to feast on a ship and sate its hunger. Over three hundred years, it has consumed 847 people. Ships have been built larger over time, so each new ship has twice as many people as the last ship. How many people were on the ship the monster ate in the first hundred years?"
    question_raw = '''Question: Kimberly went hiking and took a 4-liter bottle full of water with her. The first time she drank from it, she consumed a quarter of the water in the bottle. Later on, she drank 2/3rd of the remaining water. How much water is left in the bottle (in liters)?
Answer: Her first drink consumed 1/4 * 4 = <<1/4*4=1>>1 liter of water.
Thus there were 4 - 1 = <<4-1=3>>3 liters of water left in the bottle.
Next, she drank 2/3 * 3 = <<2/3*3=2>>2 liters.
Thus, there were 3 - 2 = <<3-2=1>>1 liters remaining.
#### 1
Question: Rachel earned $200 babysitting. She spent 1/4 of the money on lunch. She spent 1/2 of the money on a DVD. How much did Rachel have left?
Answer: Rachel spent 1/4*200 = $<<200*1/4=50>>50 on lunch.
Rachel spent 1/2*200 = $<<200*1/2=100>>100 on a DVD.
Rachel has 200-50-100 = $<<200-50-100=50>>50 left.
#### 50
Question: Jan enters a double dutch competition.  After training she doubles her speed which used to be 70 skips per minute.  How many skips does she do in 5 minutes?
Answer: She does 70*2=<<70*2=140>>140 skips a minute
So he does 140*5=<<140*5=700>>700 skips in the 5 minutes
#### 700
Question: Lauren's social media channel makes $0.50 for every commercial that's viewed and $1.00 for every person who subscribes.  On Tuesday, 100 people watched commercials before viewing her content and 27 people subscribed.  How much money did she make?
Answer: She makes $0.50 for viewed commercials and 100 people watched them on Tuesday for a total of .50*100 = $<<0.50*100=50.00>>50.00
She makes $1.00 for every person who subscribes and she had 27 people sign up so that's 1*27 = $<<1*27=27.00>>27.00
Between the viewed commercials that made $50.00 and the $27.00 she made from new subscribers, she made 50+27 = $<<50+27=77.00>>77.00
#### 77
Question: Jack bought an ice cream cone before jogging to the beach. If the ice cream cone will melt in 10 minutes, the beach is 16 blocks away, and each block is 1/8th of a mile, how fast does Jack need to jog (in miles per hour) to get to the beach before the ice cream melts?
Answer: First find the total distance to the beach: 16 blocks * 1/8 mile/block = <<16*1/8=2>>2 miles
Then find how long in hours Jack has to get to the beach: 10 minutes / 60 minutes/hour = 1/6 hour
Then divide the distance Jack needs to cover by the time in hours he has to find how fast he needs to jog: 2 miles / 1/6 hour = 12 miles/hour
#### 12
Question: While on vacation in Bali, Thea bought a hat from a craftsman worth $70. If she gave the craftsman four $20 bills, how much change did she get?
'''
    # instruction_following = " Let's think step by step and output the final answer after \'####\'."
    instruction_following = ""
    prompt = question_raw + instruction_following
#     answer = '''Let S be the number of people on the first hundred years' ship.
# The second hundred years' ship had twice as many as the first, so it had 2S people.
# The third hundred years' ship had twice as many as the second, so it had 2 * 2S = <<2*2=4>>4S people.
# All the ships had S + 2S + 4S = 7S = 847 people.
# Thus, the ship that the monster ate in the first hundred years had S = 847 / 7 = <<847/7=121>>121 people on it.
# #### 121'''
    answer = '''If she gave the craftsman four $20 bills, the total amount of money she gave to the craftsman is 4*$20 = $<<4*20=80>>80.
Since the hat was worth $70, the craftsman gave her change equal to $80-$70=$10
#### 10'''

    # Add special tokens for the Instruct model. The Base model does not require the following two lines.
    # m = [{"role": "user", "content": prompt}, ]
    # prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
    # print(f"Prompt:\n{prompt}\n{'=' * 100}")
    # import time; time.sleep(10)
    input_ids = tokenizer(prompt)['input_ids']
    input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)

    out = generate_beam_search(
        model,
        prompt=input_ids,
        gen_length=256,
        block_length=32,
        num_beams=5,
        num_return_sequences=1,
        temperature=1.0,
        cfg_scale=0.0,
        remasking="low_confidence",
        mask_id=126336,
        length_penalty=1.0,
    )
    print(tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=False)[0])


if __name__ == '__main__':
    main()