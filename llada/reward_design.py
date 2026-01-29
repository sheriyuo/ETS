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


def true_reward(generated_text, ground_truth):
    reward = compute_score(generated_text, ground_truth)
    return reward
    
def self_certainty_reward(logits, mask_index, temperature=0.):
    if temperature != 0.:
        logits = logits / temperature
    lsm = F.log_softmax(logits[0], dim=-1)        # (L, V) : log p
    V = lsm.size(-1)
    # log(V * p) = logV + log p
    logV = np.log(V)
    pos_mask = mask_index[0]                      # (L,)
    if pos_mask.sum().item() == 0:
        return 0.0

    # (L, V) -> (L,)：sum the log prob over vocab 
    token_term = (logV + lsm).sum(dim=-1)         # sum_j log(V * p_ij)
    # self-certainty = - (1/V) * average( token_term )， over masked positions
    sc_tokens = - token_term[pos_mask] / V        # (num_masked,)
    reward = sc_tokens.mean()
    # return (reward.item() - 3) / (19 - 3)
    return reward.item()

def entropy_reward(logits, mask_index, temperature=0):
    # H = -1/T * sum_i sum_j p_ij * log p_ij
    if logits.dim() == 3:
        logits = logits[0]
    if temperature != 0.:
        logits = logits / temperature
    log_probs = F.log_softmax(logits, dim=-1)  # (L, V)
    probs = torch.exp(log_probs)  # (L, V)
    entropy_per_pos = - (probs * log_probs).sum(dim=-1)  # (L,)
    if mask_index is not None:
        valid_pos = mask_index[0] if mask_index.dim() == 2 else mask_index
    if valid_pos.sum().item() == 0:
        return 0.0
    H = entropy_per_pos[valid_pos].mean().item()
    reward = -H
    return reward

def log_probs_reward(logits, tokens, mask_id, exponent=0.25, temperature=0.):
    if logits.dim() == 3:
        logits = logits[0]  # (L, V)
    if temperature != 0.:
        logits = logits / temperature
    log_probs = F.log_softmax(logits, dim=-1)  # (L, V)
    token_ids = tokens[0]  # (L,)
    seq_length = token_ids.size(0)
    positions = torch.arange(seq_length, device=token_ids.device)
    token_logp = log_probs[positions, token_ids]  # (L,)
    valid = (token_ids != mask_id)
    token_logp = token_logp[valid]
    T = token_logp.numel()
    if T == 0:
        return 0.0
    avg_logp_scaled = exponent * token_logp.mean()
    reward = torch.exp(avg_logp_scaled).item()
    return reward

def cal_log_probs(logits, x0, prompt_length, mask_index):
    log_probs = F.log_softmax(logits, dim=-1)   # [B, L, V]

    # 取出每个位置对应 token 的 log-prob
    token_log_probs = log_probs.gather(
        dim=-1, index=x0.unsqueeze(-1)
    ).squeeze(-1)  # [B, L]
    token_log_probs = token_log_probs[:, prompt_length:]  # [B, gen_length]
    print("token_log_probs shape:", token_log_probs.shape)
    # 只计算被 mask 的位置的 log-prob
    log_p = []
    for i in range(token_log_probs.size(0)):
        masked_token_log_probs = token_log_probs[i][mask_index[i][prompt_length:]]
        log_p.append(masked_token_log_probs.sum().item())
    # log_p = token_log_probs.sum(dim=-1)  # [B]
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
def substep_generate(
    model,
    prompt,
    x,
    steps,
    start_step,
    end_step,
    remain_steps,
    block_length,
    current_block,
    end_block,
    prompt_index,
    temperature=0.,
    cfg_scale=0., 
    remasking='low_confidence', 
    mask_id=126336,
):
    # Assume that guide length can be divided by block length or block length can be divided by guide length
    xt_logits = None
    # generate for the selected blocks
    for num_block in range(current_block, end_block):
        # recalculate num_transfer_tokens
        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length:] == mask_id)
        if (num_block == current_block) and (start_step > 0):
            num_transfer_tokens_short = get_num_transfer_tokens(block_mask_index, steps-start_step)
            pad_len = steps - num_transfer_tokens_short.size(1)  # == start_step
            if pad_len > 0:
                pad = num_transfer_tokens_short[:, :1].expand(-1, pad_len) 
                num_transfer_tokens = torch.cat([pad, num_transfer_tokens_short], dim=1)  # (B, steps)
            else:
                num_transfer_tokens = num_transfer_tokens_short
        else:
            num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        iterator = range(start_step, end_step) if num_block == current_block else range(steps)
        for i in iterator:
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
            # Record p(x_t|x_0) at the last step of the last block for calculating p_small, if necessary
            if (i == start_step + steps - 1) and (num_block == end_block - 1):
                xt_logits = logits.clone()

            x0 = gumbel_max_sample(logits, temperature=temperature, chunk_size=1)

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
    return x, xt_logits

@torch.no_grad()
def guide_generate_in_block(
    model, 
    prompt, 
    steps=128, 
    gen_length=128, 
    block_length=128, 
    temperature=0.,
    cfg_scale=0., 
    remasking='low_confidence', 
    mask_id=126336,
    guide_steps=8,
    num_candidates=3,
    monte_carlo_num=3,
    energy_weight=0.1,
    ground_truth=None,
    decode_fn=None,
    task="gsm8k",
    accumulate_x0=False,
    entry_point=None,
    accelerate=False,
):
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    prompt_index = (x != mask_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    in_block_guide_steps = block_length // (gen_length // guide_steps)
    x0_buffer = []
    extract_answers = []

    for num_block in range(num_blocks):
        # block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length:] == mask_id)
        # num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        for guide_step in range(in_block_guide_steps):
            x_batch = x.repeat(num_candidates, 1)
            # generate to x_{t-iB}
            start_time = time.time()
            x_batch, _ = substep_generate(
                model,
                prompt,
                x_batch,
                steps,
                start_step=guide_step * (steps // in_block_guide_steps),
                end_step=(guide_step + 1) * (steps // in_block_guide_steps),
                remain_steps=steps * num_blocks - steps * num_block - (guide_step * (steps // in_block_guide_steps)),
                block_length=block_length,
                current_block=num_block,
                end_block=num_block + 1,
                prompt_index=prompt_index.repeat(num_candidates, 1),
                temperature=temperature,
                cfg_scale=cfg_scale,
                remasking=remasking,
                mask_id=mask_id,
            )
            end_time = time.time()
            during_time = end_time - start_time
            x_batch_mc = x_batch.repeat_interleave(monte_carlo_num, dim=0)
            # generate to x0 for mc estimation
            mask_index = (x_batch_mc == mask_id)
            mc_start_step = (guide_step + 1) * (steps // in_block_guide_steps) if (guide_step + 1) * (steps // in_block_guide_steps) < steps else 0
            mc_current_block = num_block if mc_start_step != 0 else num_block + 1
            start_time = time.time()

            x0_batch_mc, _ = substep_generate(
                model,
                prompt,
                x_batch_mc,
                steps,
                start_step=mc_start_step,
                end_step=steps,
                remain_steps=steps * num_blocks - steps * num_block - ((guide_step + 1) * (steps // in_block_guide_steps)),
                block_length=block_length,
                current_block=mc_current_block,
                end_block=num_blocks,
                prompt_index=prompt_index.repeat(num_candidates * monte_carlo_num, 1),
                temperature=temperature,
                cfg_scale=cfg_scale,
                remasking=remasking,
                mask_id=mask_id,
            )
            end_time = time.time()
            during_time = end_time - start_time
            log_p_sub = [0.0 for _ in range(num_candidates * monte_carlo_num)]
            seqs = []
            answers = []
            EOS_ID = 126081
            for i in range(num_candidates * monte_carlo_num):
                tokens = x0_batch_mc[i][prompt.shape[1]:].tolist()
                seqs.append(tokens)
                x0_buffer.append(tokens)
                if decode_fn is not None:
                    generate_text = decode_fn(tokens, skip_special_tokens=False)
                    generate_text = generate_text.split('<|endoftext|>')[0].strip()
                    if task == "gsm8k":
                        extract_answer = gsm8k_extract(generate_text, method="flexible")[0]
                        if extract_answer == "":
                            extract_answer = None
                    elif task == "humaneval":
                        extract_answer = humaneval_extract(generate_text, entry_point)
                    elif task == "math500":
                        extract_answer = math_extract(generate_text)
                        if extract_answer == "[invalidanswer]":
                            extract_answer = None
                    elif task == "gpqa":
                        extract_answer = parse_answer_gpqa(generate_text)
                    else:
                        NotImplementedError()
                    answers.append(extract_answer)
                    extract_answers.append(extract_answer)
            if accumulate_x0:
                valid_answers = [ans for ans in extract_answers if ans is not None]
                if len(valid_answers) == 0:
                    best_ans, best_count = None, 0
                    rewards = [0.0 for _ in range(num_candidates * monte_carlo_num)]
                else:
                    counter = Counter(valid_answers)
                    best_ans, best_count = counter.most_common(1)[0]
                    if best_count == 1:
                        rewards = [0.0 for _ in range(num_candidates * monte_carlo_num)]
                    else:
                        rewards = [1.0 if ans == best_ans else 0.0 for ans in extract_answers]
                current_batch_rewards = rewards[-(num_candidates * monte_carlo_num):]
                corrected_rewards = []
                for k in range(num_candidates * monte_carlo_num):
                    corrected_reward = np.exp(np.clip(log_p_sub[k], -10, 10) + current_batch_rewards[k] / energy_weight)
                    corrected_rewards.append(corrected_reward)
                energies = [np.mean(corrected_rewards[k*monte_carlo_num:(k+1)*monte_carlo_num]) for k in range(num_candidates)]
            else:
                valid_answers = [ans for ans in answers if ans is not None]
                if len(valid_answers) == 0:
                    best_ans, best_count = None, 0
                    rewards = [0.0 for _ in range(num_candidates * monte_carlo_num)]
                else:
                    counter = Counter(valid_answers)
                    best_ans, best_count = counter.most_common(1)[0]
                    if best_count == 1:
                        rewards = [0.0 for _ in range(num_candidates * monte_carlo_num)]
                    else:
                        rewards = [1.0 if ans == best_ans else 0.0 for ans in answers]
                corrected_rewards = []
                for k in range(num_candidates * monte_carlo_num):
                    corrected_reward = np.exp(np.clip(log_p_sub[k], -10, 10) + rewards[k] / energy_weight)
                    corrected_rewards.append(corrected_reward)
                energies = [np.mean(corrected_rewards[k*monte_carlo_num:(k+1)*monte_carlo_num]) for k in range(num_candidates)]

            if num_candidates > 1 and any(e != 0 for e in energies):
                if len(set(energies)) == 1 and energies[0] == 0.0:
                    selected_idx = 0
                    selected_x = x_batch[selected_idx:selected_idx+1]
                else:
                    energies_tensor = torch.tensor(energies, device=x.device, dtype=torch.float32)
                    importance_weights = energies_tensor / energies_tensor.sum()
                    # probs = F.softmax(importance_weights, dim=0)
                    probs = importance_weights
                    selected_idx = torch.multinomial(probs, 1).item()
                    selected_x = x_batch[selected_idx:selected_idx+1]
            else:
                selected_x = x_batch[0:1]
            
            x = selected_x.clone()
    if accumulate_x0:
        valid_answers = [ans for ans in extract_answers if ans is not None]
        if len(valid_answers) == 0:
            best_ans, best_count = None, 0
        else:
            counter = Counter(valid_answers)
            best_ans, best_count = counter.most_common(1)[0]
        best_x = None
        if best_count == 1:
            best_x = torch.tensor(x0_buffer[0], dtype=torch.long, device=x.device).unsqueeze(0)
            x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
            x[:, :prompt.shape[1]] = prompt.clone()
            x[0, prompt.shape[1]:] = best_x
        else:
            for n in range(len(x0_buffer)):
                if extract_answers[n] == best_ans:
                    best_x = torch.tensor(x0_buffer[n], dtype=torch.long, device=x.device).unsqueeze(0)
                    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
                    x[:, :prompt.shape[1]] = prompt.clone()
                    x[0, prompt.shape[1]:] = best_x
                    break  
    return x

@torch.no_grad()
def guide_generate_cross_block(
    model, 
    prompt, 
    steps=128, 
    gen_length=128, 
    block_length=128, 
    temperature=0.,
    cfg_scale=0., 
    remasking='low_confidence', 
    mask_id=126336,
    guide_steps=8,
    num_candidates=3,
    monte_carlo_num=3,
    energy_weight=0.1,
    ground_truth=None,
    decode_fn=None,
    task="gsm8k",
    accumulate_x0=False,
    entry_point=None,
    accelerate=False,
    reward_type="ground_truth",
):
    dir_name = f"./logs/ablation/reward_design/{reward_type}/"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    log_file = os.path.join(dir_name, f"guide_generate_{time.strftime('%Y%m%d_%H%M%S')}.log")
    log = []
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    prompt_index = (x != mask_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    guide_step_cross_blocks = (gen_length // guide_steps) // block_length
    x0_buffer = []
    extract_answers = [] 

    for guide_step in range(guide_steps):
        x_batch = x.repeat(num_candidates, 1)
        start_time = time.time()
        x_batch, _ = substep_generate(
            model,
            prompt,
            x_batch,
            steps,
            start_step=0,
            end_step=steps,
            remain_steps=steps * num_blocks - (guide_step * (gen_length // guide_steps)),
            block_length=block_length,
            current_block=guide_step * guide_step_cross_blocks,
            end_block=(guide_step + 1) * guide_step_cross_blocks,
            prompt_index=prompt_index.repeat(num_candidates, 1),
            temperature=temperature,
            cfg_scale=cfg_scale,
            remasking=remasking,
            mask_id=mask_id,
        )
        end_time = time.time()
        during_time = end_time - start_time

        x_batch_mc = x_batch.repeat_interleave(monte_carlo_num, dim=0)
        mask_index = (x_batch_mc == mask_id)
        start_time = time.time()
        x0_batch_mc, x0_logits_batch = substep_generate(
            model,
            prompt,
            x_batch_mc,
            steps,
            start_step=0,
            end_step=steps,
            remain_steps=steps * num_blocks - (guide_step * (gen_length // guide_steps)),
            block_length=block_length,
            current_block=(guide_step + 1) * guide_step_cross_blocks,
            end_block=num_blocks,
            prompt_index=prompt_index.repeat(num_candidates, 1),
            temperature=temperature,
            cfg_scale=cfg_scale,
            remasking=remasking,
            mask_id=mask_id,
        )
        end_time = time.time()
        during_time = end_time - start_time

        log_p_sub = [0.0 for _ in range(num_candidates * monte_carlo_num)]
        seqs = []
        answers = []
        EOS_ID = 126081
        gt_rewards = []
        self_certainty_rewards = []
        entropy_rewards = []
        log_probs_rewards = []
        self_consistency_rewards = []
        for i in range(num_candidates * monte_carlo_num):
            tokens = x0_batch_mc[i:i+1][0][prompt.shape[1]:].tolist()
            seqs.append(tokens)
            x0_buffer.append(tokens)
            if decode_fn is not None:
                generate_text = decode_fn(tokens, skip_special_tokens=False)
                generate_text = generate_text.split('<|eot_id|>')[0].strip()
                if task == "gsm8k":
                    extract_answer = gsm8k_extract(generate_text, method="flexible")
                    if extract_answer[0] == "":
                        if extract_answer[1] != "":
                            extract_answer = extract_answer[1]
                        else:
                            extract_answer = None
                    else:
                        extract_answer = extract_answer[0]
                elif task == "humaneval":
                    extract_answer = humaneval_extract(generate_text, entry_point)
                elif task == "math500":
                    extract_answer = math_extract(generate_text)
                    if extract_answer == "[invalidanswer]":
                        extract_answer = None
                elif task == "gpqa":
                    extract_answer = parse_answer_gpqa(generate_text)
                else:
                    NotImplementedError()
                answers.append(extract_answer)
                extract_answers.append(extract_answer)
                if ground_truth is not None:
                    gt_rewards.append(compute_score(generate_text, ground_truth, method="flexible"))
                self_certainty_rewards.append(self_certainty_reward(x0_logits_batch[i:i+1], mask_index[i:i+1], temperature=temperature))
                entropy_rewards.append(entropy_reward(x0_logits_batch[i:i+1], mask_index[i:i+1], temperature=temperature))
                log_probs_rewards.append(log_probs_reward(x0_logits_batch[i:i+1], x0_batch_mc[i:i+1], mask_id, exponent=0.25, temperature=temperature))
        if accumulate_x0:
            valid_answers = [ans for ans in extract_answers if ans is not None]
            if len(valid_answers) == 0:
                best_ans, best_count = None, 0
                self_consistency_rewards = [0.0 for _ in range(num_candidates * monte_carlo_num)]
            else:
                counter = Counter(valid_answers)
                best_ans, best_count = counter.most_common(1)[0]
                if best_count == 1:
                    self_consistency_rewards = [0.0 for _ in range(num_candidates * monte_carlo_num)]
                else:
                    self_consistency_rewards = [1.0 if ans == best_ans else 0.0 for ans in extract_answers]
            current_batch_rewards = self_consistency_rewards[-(num_candidates * monte_carlo_num):]
            corrected_self_consistency_rewards = []
            for k in range(num_candidates * monte_carlo_num):
                corrected_self_consistency_reward = np.exp(np.clip(log_p_sub[k], -10, 10) + current_batch_rewards[k] / energy_weight)
                corrected_self_consistency_rewards.append(corrected_self_consistency_reward)
            self_consistency_energies = [np.mean(corrected_self_consistency_rewards[k*monte_carlo_num:(k+1)*monte_carlo_num]) for k in range(num_candidates)]
        else:
            valid_answers = [ans for ans in answers if ans is not None]
            if len(valid_answers) == 0:
                best_ans, best_count = None, 0
                self_consistency_rewards = [0.0 for _ in range(num_candidates * monte_carlo_num)]
            else:
                counter = Counter(valid_answers)
                best_ans, best_count = counter.most_common(1)[0]
                if best_count == 1:
                    self_consistency_rewards = [0.0 for _ in range(num_candidates * monte_carlo_num)]
                else:
                    self_consistency_rewards = [1.0 if ans == best_ans else 0.0 for ans in answers]
            corrected_self_consistency_rewards = []
            for k in range(num_candidates * monte_carlo_num):
                corrected_self_consistency_reward = np.exp(np.clip(log_p_sub[k], -10, 10) + self_consistency_rewards[k] / energy_weight)
                corrected_self_consistency_rewards.append(corrected_self_consistency_reward)
            self_consistency_energies = [np.mean(corrected_self_consistency_rewards[k*monte_carlo_num:(k+1)*monte_carlo_num]) for k in range(num_candidates)]

        for i in range(num_candidates * monte_carlo_num):
            log.append({
                "guide_step": guide_step + 1,
                "candidate_index": i//monte_carlo_num,
                "mc_index": i % monte_carlo_num,
                "generated_sequence": seqs[i],
                "extracted_answer": answers[i],
                "energy": energies[i//monte_carlo_num],
                "self_consistency_reward": self_consistency_rewards[i],
                "ground_truth_reward": gt_rewards[i] if len(gt_rewards) > 0 else None,
                "self_certainty_reward": self_certainty_rewards[i],
                "entropy_reward": entropy_rewards[i],
                "log_probs_reward": log_probs_rewards[i],
            })
        
        if reward_type == "ground_truth":
            rewards = gt_rewards
            energies = [np.mean(rewards[k*monte_carlo_num:(k+1)*monte_carlo_num]) for k in range(num_candidates)]
        elif reward_type == "self_certainty":
            rewards = self_certainty_rewards
            energies = [np.mean(rewards[k*monte_carlo_num:(k+1)*monte_carlo_num]) for k in range(num_candidates)]
        elif reward_type == "entropy":
            rewards = entropy_rewards
            energies = [np.mean(rewards[k*monte_carlo_num:(k+1)*monte_carlo_num]) for k in range(num_candidates)]
        elif reward_type == "log_probs":
            rewards = log_probs_rewards
            energies = [np.mean(rewards[k*monte_carlo_num:(k+1)*monte_carlo_num]) for k in range(num_candidates)]
        elif reward_type == "self_consistency":
            energies = self_consistency_energies

        print(f"Guide step {guide_step+1}/{guide_steps} candidate energies: {energies}")

        if num_candidates > 1 and any(e != 0 for e in energies):
            if len(set(energies)) == 1 and energies[0] == 0.0:
                selected_idx = 0
                selected_x = x_batch[selected_idx:selected_idx+1]
            else:
                energies_tensor = torch.tensor(energies, device=x.device, dtype=torch.float32)
                importance_weights = energies_tensor / energies_tensor.sum()
                # probs = F.softmax(importance_weights, dim=0)
                probs = importance_weights
                selected_idx = torch.multinomial(probs, 1).item()
                selected_x = x_batch[selected_idx:selected_idx+1]

        else:
            selected_x = x_batch[0:1]
        
        x = selected_x.clone()
    if accumulate_x0:
        valid_answers = [ans for ans in extract_answers if ans is not None]
        if len(valid_answers) == 0:
            best_ans, best_count = None, 0
        else:
            counter = Counter(valid_answers)
            best_ans, best_count = counter.most_common(1)[0]
        best_x = None
        if best_count == 1:
            best_x = torch.tensor(x0_buffer[0], dtype=torch.long, device=x.device).unsqueeze(0)
            x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
            x[:, :prompt.shape[1]] = prompt.clone()
            x[0, prompt.shape[1]:] = best_x
        else:
            for n in range(len(x0_buffer)):
                if extract_answers[n] == best_ans:
                    best_x = torch.tensor(x0_buffer[n], dtype=torch.long, device=x.device).unsqueeze(0)
                    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
                    x[:, :prompt.shape[1]] = prompt.clone()
                    x[0, prompt.shape[1]:] = best_x
                    break  
    log.append({
        "gound_truth": ground_truth
    })
    # with open(log_file, 'w') as f:
    #     for entry in log:
    #         f.write(json.dumps(entry) + '\n')
    return x

@torch.no_grad()
def generate_vanilla_mc_reward(
    model, 
    prompt, 
    steps=128, 
    gen_length=128, 
    block_length=128, 
    temperature=0.,
    cfg_scale=0., 
    remasking='low_confidence', 
    mask_id=126336,
    guide_steps=8,
    num_candidates=3,
    monte_carlo_num=3,
    energy_weight=0.1,
    ground_truth=None,
    decode_fn=None,
    task="gsm8k",
    accumulate_x0=False,
    entry_point=None,
    accelerate=False,
    reward_type="ground_truth",
):
    if gen_length // guide_steps <= block_length:
        return guide_generate_in_block(
            model,
            prompt,
            steps=steps,
            gen_length=gen_length,
            block_length=block_length,
            temperature=temperature,
            cfg_scale=cfg_scale,
            remasking=remasking,
            mask_id=mask_id,
            guide_steps=guide_steps,
            num_candidates=num_candidates,
            monte_carlo_num=monte_carlo_num,
            energy_weight=energy_weight,
            ground_truth=ground_truth,
            decode_fn=decode_fn,
            task=task,
            accumulate_x0=accumulate_x0,
            entry_point=entry_point,
            accelerate=accelerate
        )
    else:
        return guide_generate_cross_block(
            model,
            prompt,
            steps=steps,
            gen_length=gen_length,
            block_length=block_length,
            temperature=temperature,
            cfg_scale=cfg_scale,
            remasking=remasking,
            mask_id=mask_id,
            guide_steps=guide_steps,
            num_candidates=num_candidates,
            monte_carlo_num=monte_carlo_num,
            energy_weight=energy_weight,
            ground_truth=ground_truth,
            decode_fn=decode_fn,
            task=task,
            accumulate_x0=accumulate_x0,
            entry_point=entry_point,
            accelerate=accelerate,
            reward_type=reward_type,
        )

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
    prompt = question_raw
    answer = '''If she gave the craftsman four $20 bills, the total amount of money she gave to the craftsman is 4*$20 = $<<4*20=80>>80.
Since the hat was worth $70, the craftsman gave her change equal to $80-$70=$10
#### 10'''

    # Add special tokens for the Instruct model. The Base model does not require the following two lines.
    m = [{"role": "user", "content": prompt}, ]
    prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
    input_ids = tokenizer(prompt)['input_ids']
    input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)
    # input_ids = input_ids.repeat(10, 1)
    # warm up
    
    # print(tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=False)[0])
    start_time = time.time()
    out = generate_vanilla_mc_reward(
        model,
        input_ids,
        steps=256,
        gen_length=256,
        block_length=8,
        temperature=0.5,
        cfg_scale=0.,
        remasking='low_confidence',
        guide_steps=8,
        num_candidates=5,
        monte_carlo_num=3,
        energy_weight=0.1,
        ground_truth=answer,
        decode_fn=tokenizer.decode,
        task="gsm8k",
        accumulate_x0=True,
        reward_type="ground_truth",
    )
    end_time = time.time()
    print(f"Vanilla MC Generation time: {end_time - start_time} seconds")
    print(tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=False)[0])
if __name__ == '__main__':
    main()