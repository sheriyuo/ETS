import torch
import torch.nn.functional as F
from collections import Counter

def compute_entropy(logits):
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1)
    return entropy.mean().item()

def compute_self_certainty_reward(logits):
    V = logits.shape[-1]
    lsm = F.log_softmax(logits, dim=-1)
    logV = torch.log(torch.tensor(V, device=logits.device))

    n_positions = logits.shape[0]
    if n_positions == 0:
        return 0.0
    
    token_term = (logV + lsm).sum(dim=-1)
    sc_tokens = -token_term / V
    reward = sc_tokens.mean()
    return reward.item()

def compute_entropy_reward(logits, original_length):
    generated_logits = logits[:, original_length-1:, :]  # [1, T, vocab_size]
    if generated_logits.shape[1] == 0:
        return 0.0
    
    probs = F.softmax(generated_logits, dim=-1)
    log_probs = F.log_softmax(generated_logits, dim=-1)
    # H(p_t) = -sum_{v in V} p_t(v) log p_t(v)
    entropy_per_token = -(probs * log_probs).sum(dim=-1)  # [1, T]
    total_entropy = entropy_per_token.mean()
    reward = -total_entropy
    return reward.item()

def compute_sequence_log_probability(model, full_sequence, original_length, logits=None):
    if logits is None:
        outputs = model(
            input_ids=full_sequence[:, :-1],
            return_dict=True
        )
        logits = outputs.logits

    if logits.shape[1] == full_sequence.shape[1] - 1:
        logits = logits[:, original_length-1:]

    
    log_probs = F.log_softmax(logits, dim=-1)  # (1, seq_len-1, vocab_size)
    target_tokens = full_sequence[:, original_length:]  # (1, seq_len-1)
    target_tokens_expanded = target_tokens.unsqueeze(-1)  # (1, seq_len-1, 1)
    
    token_log_probs = torch.gather(log_probs, dim=-1, index=target_tokens_expanded)
    token_log_probs = token_log_probs.squeeze(-1)  # (1, seq_len-1)
    
    total_log_prob = token_log_probs.sum(dim=1)
    mean_log_prob = token_log_probs.mean(dim=1)
    
    return total_log_prob, mean_log_prob

def is_valid(ans):
    # For HumanEval
    if ans is None or 'pass' in ans or ans.count('#') >= 10 or ans == ' ':
        return False
    # For MATH500
    if ans == '[invalidanswer]':
        return False
    
    return True

def compute_srt_reward(extract_fn, all_candidate_texts, candidate_block_texts_list, all_candidate_answers=None):
    """
    r(y) = 1[answer(y) = y_pseudo]
    """
    if all_candidate_answers is None:
        all_candidate_answers = []

    for candidate_text in all_candidate_texts:
        candidate_answer = extract_fn(candidate_text)
        if is_valid(candidate_answer):
            all_candidate_answers.append(candidate_answer)

    if not all_candidate_answers:
        return [[0.0] * len(texts) for texts in candidate_block_texts_list], all_candidate_answers
    
    answer_counter = Counter(all_candidate_answers)
    majority_answer = answer_counter.most_common(1)[0][0]
    print(f"SRT: Total {len(all_candidate_answers)} valid answers from {len(all_candidate_texts)} sequences, majority: {majority_answer} with {answer_counter[majority_answer]}")

    block_rewards_list = []
    for block_idx, candidate_block_texts in enumerate(candidate_block_texts_list):
        block_rewards = []
        for candidate_text in candidate_block_texts:
            candidate_answer = extract_fn(candidate_text)
            if candidate_answer is not None and candidate_answer == majority_answer:
                reward = 1.0
            else:
                reward = 0.0
            block_rewards.append(reward)
        block_rewards_list.append(block_rewards)
    
    return block_rewards_list, all_candidate_answers
