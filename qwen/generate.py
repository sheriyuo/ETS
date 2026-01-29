import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict
import re

from utils.reward import compute_srt_reward, compute_sequence_log_probability

def generate_candidate_blocks(model, block_size, m_candidates, tokenizer, current_tokens, temperature):
    candidate_blocks = model.generate(
        current_tokens.repeat(m_candidates, 1),
        max_new_tokens=block_size,
        do_sample=True,
        temperature=temperature,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        num_return_sequences=1,
        use_cache=True,
    )
    
    original_length = current_tokens.shape[1]
    candidate_tokens_list = []
    
    for i in range(m_candidates):
        block_tokens = candidate_blocks[i, original_length:]
        candidate_tokens_list.append(block_tokens)
    
    return candidate_tokens_list

def importance_sampling_guidance(
    model,
    extract_fn,
    tokenizer,
    current_tokens,
    max_length,
    original_length,
    block_size,
    m_candidates,
    k_monte_carlo,
    temperature,
    lambda_param,
    until=[],
    all_candidate_answers=[],
    use_importance_sampling=False,
    small_model=None,
):
    candidate_blocks = generate_candidate_blocks(
        model, block_size, m_candidates, tokenizer, current_tokens, temperature
    )
    
    candidate_rewards = []
    all_candidate_texts = []
    candidate_block_texts_list = []
    all_full_sequence_objects = []
    batched_inputs = []
    block_indices = []
    
    for block_idx, block_tokens in enumerate(candidate_blocks):
        candidate_sequence = torch.cat([
            current_tokens[0], 
            block_tokens
        ]).unsqueeze(0)
        
        repeated_sequence = candidate_sequence.repeat(k_monte_carlo, 1)
        batched_inputs.append(repeated_sequence)
        block_indices.extend([block_idx] * k_monte_carlo)
    
    if batched_inputs:
        all_inputs = torch.cat(batched_inputs, dim=0)
        total_sequences = all_inputs.shape[0]
        max_new_tokens = max_length - (current_tokens.shape[1] - original_length)
        
        generate_model = small_model if use_importance_sampling else model
        
        generate_kwargs = {
            'max_new_tokens': max_new_tokens,
            'do_sample': True,
            'temperature': temperature,
            'pad_token_id': tokenizer.pad_token_id,
            'eos_token_id': tokenizer.eos_token_id,
            'num_return_sequences': 1,
            'use_cache': True,
        }
        
        if not use_importance_sampling:
            all_full_sequences = generate_model.generate(
                all_inputs,
                **generate_kwargs
            )
        else:
            all_full_sequences = small_model.generate(
                all_inputs,
                **generate_kwargs
            )
        
        for block_idx in range(m_candidates):
            block_seq_indices = [i for i, idx in enumerate(block_indices) if idx == block_idx]
            
            candidate_block_texts = []
            full_sequence_objects = []
            
            for seq_idx, global_idx in enumerate(block_seq_indices):
                full_sequence = all_full_sequences[global_idx]
                
                pad_positions = (full_sequence == tokenizer.pad_token_id).nonzero(as_tuple=True)[0]
                if len(pad_positions) > 0:
                    first_pad = pad_positions[0].item()
                    full_sequence = full_sequence[:first_pad]
                
                candidate_text = tokenizer.decode(
                    full_sequence[original_length:], 
                    skip_special_tokens=True
                )
                
                for stop_seq in until:
                    if stop_seq in candidate_text:
                        candidate_text = candidate_text.split(stop_seq)[0]
                        break
                
                candidate_block_texts.append(candidate_text)
                full_sequence_objects.append(full_sequence)
            
            all_candidate_texts.extend(candidate_block_texts)
            candidate_block_texts_list.append(candidate_block_texts)
            all_full_sequence_objects.append(full_sequence_objects)
                
    else:
        print("No valid inputs for batch generation")
        return None, all_candidate_answers
    
    srt_rewards_list, all_candidate_answers = compute_srt_reward(
        extract_fn, all_candidate_texts, candidate_block_texts_list, all_candidate_answers
    )

    if use_importance_sampling:
        all_inputs_for_logits = all_full_sequences[:, :-1]
        batch_size = 100
        all_logits_ref = []
        all_logits_small = []
        seq_len = all_full_sequences.shape[1] - original_length
        all_logits_ref = model(all_inputs_for_logits, return_dict=True, logits_to_keep=seq_len).logits
        all_logits_ref, _ = compute_sequence_log_probability(model, all_inputs_for_logits, original_length, all_logits_ref)
        all_logits_small = small_model(all_inputs_for_logits, return_dict=True, logits_to_keep=seq_len).logits
        all_logits_small, _ = compute_sequence_log_probability(small_model, all_inputs_for_logits, original_length, all_logits_small)
    
    for block_idx, (candidate_block_texts, full_sequence_objects, srt_rewards) in enumerate(zip(
        candidate_block_texts_list, all_full_sequence_objects, srt_rewards_list)):
        
        sequence_rewards = []
        
        for k, (candidate_text, full_sequence) in enumerate(zip(candidate_block_texts, full_sequence_objects)):
            if k >= len(full_sequence_objects):
                continue

            global_idx = block_idx * k_monte_carlo + k
            
            if use_importance_sampling:
                log_p_ref = all_logits_ref[global_idx]
                log_p_small = all_logits_small[global_idx]

            srt_reward = torch.tensor(srt_rewards[k] if k < len(srt_rewards) else 0.0).cuda()
            
            if use_importance_sampling:
                energy = torch.exp(
                    (log_p_ref - log_p_small).clip(min=-10, max=10) + 
                    srt_reward / max(lambda_param, 1e-8)
                ).item()
            else:
                energy = torch.exp(srt_reward / max(lambda_param, 1e-8)).item()
            
            sequence_rewards.append(energy)
        
        if sequence_rewards:
            avg_reward = sum(sequence_rewards) / len(sequence_rewards)
            candidate_rewards.append(avg_reward)
        else:
            candidate_rewards.append(0.0)
    
    if candidate_rewards and max(candidate_rewards) > 0:
        rewards_tensor = torch.tensor(candidate_rewards, device=current_tokens.device)
        total_weight = rewards_tensor.sum()
        
        if total_weight > 0:
            normalized_weights = rewards_tensor / total_weight
            selected_idx = torch.multinomial(normalized_weights, 1).item()
            selected_block = candidate_blocks[selected_idx]
            return selected_block.unsqueeze(0), all_candidate_answers
    
    return None, all_candidate_answers

@torch.no_grad()
def generate(
    model,
    extract_fn,
    input_ids,
    tokenizer,
    max_length,
    block_size,
    m_candidates,
    k_monte_carlo,
    until,
    temperature,
    lambda_param,
    use_importance_sampling=False,
    small_model=None,
):
    assert extract_fn is not None, "generate() need extract_fn to get answer from generation."

    original_length = input_ids.shape[1]

    total_blocks = (max_length + block_size - 1) // block_size

    current_tokens = input_ids
    all_candidate_answers = []

    for block_step in range(total_blocks):
        current_token_position = current_tokens.shape[1] - original_length
        
        next_block, all_candidate_answers = importance_sampling_guidance(
            model, extract_fn, tokenizer, current_tokens, max_length, original_length, 
            block_size, m_candidates, k_monte_carlo, temperature, lambda_param,
            until, all_candidate_answers, use_importance_sampling, small_model
        )
        
        if next_block is not None:
            current_tokens = torch.cat([current_tokens, next_block], dim=1)
        else:
            max_new_tokens = min(block_size, max_length - current_token_position)
            
            if max_new_tokens <= 0:
                break
            
            generated_block = model.generate(
                current_tokens,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                num_return_sequences=1,
                use_cache=True,
            )
            
            new_tokens = generated_block[0, current_tokens.shape[1]:]
            current_tokens = torch.cat([current_tokens, new_tokens.unsqueeze(0)], dim=1)

        current_generated = current_tokens[0, original_length:]
        
        if tokenizer.eos_token_id in current_generated:
            eos_pos = (current_generated == tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
            if len(eos_pos) > 0:
                first_eos = eos_pos[0].item()
                current_tokens = torch.cat([
                    current_tokens[0, :original_length + first_eos + 1].unsqueeze(0)
                ], dim=1)
                break
        
        generated_text = tokenizer.decode(
            current_generated, 
            skip_special_tokens=False
        )
        
        should_stop = False
        if until is not None:
            stop_seqs = [until] if isinstance(until, str) else until
            for stop_seq in stop_seqs:
                if stop_seq in generated_text:
                    stop_pos = generated_text.find(stop_seq)
                    if stop_pos != -1:
                        truncated_text = generated_text[:stop_pos]
                        truncated_tokens = tokenizer.encode(truncated_text, add_special_tokens=False)
                        current_tokens = torch.cat([
                            input_ids,
                            torch.tensor(truncated_tokens, device=current_tokens.device).unsqueeze(0)
                        ], dim=1)
                    should_stop = True
                    break
            
            if should_stop:
                break

        if current_tokens.shape[1] >= original_length + max_length:
            break

    return current_tokens