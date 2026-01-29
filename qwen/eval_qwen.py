import accelerate
import torch
from pathlib import Path
import random
import numpy as np
import torch.nn.functional as F
from datasets import Dataset
from lm_eval.__main__ import cli_evaluate
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from tqdm import tqdm
import os
import pickle
from typing import List, Optional, Tuple, Union
from collections import defaultdict

from transformers import AutoTokenizer, AutoModelForCausalLM
from datetime import timedelta

from lm_eval.tasks.minerva_math.utils import get_unnormalized_answer, normalize_final_answer
from utils.tasks.gsm8k import extract_solution
from utils.tasks.gpqa import parse_answer_gpqa
from utils.tasks.humaneval import extract_code
from generate import generate

import matplotlib.pyplot as plt
from utils.reward import (
    compute_entropy_reward, 
    compute_self_certainty_reward,
    compute_srt_reward,
    is_valid,
    compute_sequence_log_probability
)
from datetime import datetime

def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@register_model("qwen")
class QwenEvalHarness(LM):
    def __init__(
        self,
        model_path='',
        max_length=512,
        batch_size=1,
        device="cuda",
        dtype=torch.bfloat16,
        is_rl=False,
        **kwargs,
    ):
        '''
        Args:
            model_path: model path
            max_length: max sequence length
            batch_size: batch size for evaluation
            device: device to run on
            dtype: data type for model
        '''
        super().__init__()
        self.is_rl = is_rl

        init_process_group_kwargs = accelerate.InitProcessGroupKwargs(timeout=timedelta(seconds=7200))  # 2 hrs 
        accelerator = accelerate.Accelerator(kwargs_handlers=[init_process_group_kwargs])
        if accelerator.num_processes > 1:
            self.accelerator = accelerator
        else:
            self.accelerator = None
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            trust_remote_code=True,
            padding_side="left"
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs = {
            "torch_dtype": dtype,
            "trust_remote_code": True,
            "device_map": "auto" if self.accelerator is None else None
        }

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            **model_kwargs
        )
        self.model.eval()

        self.device = torch.device(device)
        if self.accelerator is not None:
            self.model = self.accelerator.prepare(self.model)
            self.device = torch.device(f'{self.accelerator.device}')
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else: 
            self.model = self.model.to(self.device)

        if hasattr(self.model, 'module'):
            self.model = self.model.module

        self.max_length = max_length
        self.batch_size = int(batch_size)

    @property
    def rank(self):
        return self._rank
    
    @property
    def world_size(self):
        return self._world_size

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def prefix_token_id(self):
        return self.eot_token_id

    def tok_encode(self, string: str, **kwargs) -> list[int]:
        return self.tokenizer.encode(string, **kwargs)

    @torch.no_grad()
    def generate_until(self, requests):
        """Generate text until stopping condition"""
        res = []
        
        for req in tqdm(requests, desc="Generating"):
            context, gen_kwargs = req.args
            
            max_length = gen_kwargs.get("max_length", self.max_length)
            temperature = gen_kwargs.get("temperature", 0.0)
            until = gen_kwargs.get("until", None)
            do_sample = gen_kwargs.get("do_sample", False)
            
            input_ids = self.tok_encode(context, return_tensors="pt").to(self.device)
            
            if self.is_rl:
                output_ids = self.model.generate(
                    input_ids,
                    max_new_tokens=max_length,
                    num_return_sequences=1,
                    do_sample=False,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                    use_cache=True
                )
            else:
                # Use non-thinking settings from Qwen3 Technical Report
                output_ids = self.model.generate(
                    input_ids,
                    max_new_tokens=max_length,
                    num_return_sequences=1,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.8,
                    top_k=20,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                    use_cache=True
                )

            generated = output_ids[0][input_ids.shape[1]:]
            generated_text = self.tokenizer.decode(generated, skip_special_tokens=True)
            
            if until is not None:
                if isinstance(until, str):
                    until = [until]
                
                for stop_seq in until:
                    if stop_seq in generated_text:
                        generated_text = generated_text.split(stop_seq)[0]
            
            res.append(generated_text)

        if self.accelerator is not None:
            self.accelerator.wait_for_everyone()
        
        return res

    def apply_chat_template(self, chat_history, add_generation_prompt=True):
        """Apply chat template for few-shot examples"""
        conversation = []
        for msg in chat_history:
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                conversation.append(f"<|im_start|>user\n{content}<|im_end|>")
            elif role == "assistant":
                conversation.append(f"<|im_start|>assistant\n{content}<|im_end|>")
        
        if add_generation_prompt:
            conversation.append("<|im_start|>assistant\n")
        
        return "\n".join(conversation)

    def loglikelihood(self, requests):
        raise NotImplementedError("loglikelihood not implemented for qwen")

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError("loglikelihood_rolling not implemented for qwen")


@register_model("qwen-beam")
class QwenEvalHarnessBeam(LM):
    def __init__(
        self,
        model_path='',
        max_length=512,
        num_beams=5,
        temperature=0.7,
        batch_size=1,
        device="cuda",
        dtype=torch.bfloat16,
        **kwargs,
    ):
        '''
        Args:
            model_path: model path
            max_length: max sequence length
            num_beams: the number of max beams
            batch_size: batch size for evaluation
            device: device to run on
            dtype: data type for model
        '''
        super().__init__()

        self.num_beams = num_beams
        assert num_beams > 1, "num_beams must > 1"
        self.temperature = temperature

        init_process_group_kwargs = accelerate.InitProcessGroupKwargs(timeout=timedelta(seconds=7200))  # 2 hrs 
        accelerator = accelerate.Accelerator(kwargs_handlers=[init_process_group_kwargs])
        if accelerator.num_processes > 1:
            self.accelerator = accelerator
        else:
            self.accelerator = None
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            trust_remote_code=True,
            padding_side="left"
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs = {
            "torch_dtype": dtype,
            "trust_remote_code": True,
            "device_map": "auto" if self.accelerator is None else None
        }
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            **model_kwargs
        )
        self.model.eval()

        self.device = torch.device(device)
        if self.accelerator is not None:
            self.model = self.accelerator.prepare(self.model)
            self.device = torch.device(f'{self.accelerator.device}')
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else: 
            self.model = self.model.to(self.device)

        if hasattr(self.model, 'module'):
            self.model = self.model.module

        self.max_length = max_length
        self.batch_size = int(batch_size)

    @property
    def rank(self):
        return self._rank
    
    @property
    def world_size(self):
        return self._world_size

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def prefix_token_id(self):
        return self.eot_token_id

    def tok_encode(self, string: str, **kwargs) -> List[int]:
        return self.tokenizer.encode(string, **kwargs)

    def tok_decode(self, tokens: List[int], **kwargs) -> str:
        return self.tokenizer.decode(tokens, **kwargs)

    def _transformers_beam_search(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        num_beams: int = 5,
        temperature: float = 1.0,
        length_penalty: float = 1.0,
        early_stopping: bool = False,
        no_repeat_ngram_size: int = 0,
        **kwargs
    ) -> torch.Tensor:
        """Use transformers built-in beam search"""
        
        gen_kwargs = {
            "input_ids": input_ids,
            "max_new_tokens": max_new_tokens,
            "num_beams": num_beams,
            "temperature": temperature,
            "length_penalty": length_penalty,
            "early_stopping": early_stopping,
            "no_repeat_ngram_size": no_repeat_ngram_size,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
            "use_cache": True,
        }
        
        for key, value in kwargs.items():
            if key not in gen_kwargs:
                gen_kwargs[key] = value
        
        with torch.no_grad():
            output = self.model.generate(**gen_kwargs)
        
        return output

    @torch.no_grad()
    def generate_until(self, requests):
        res = []
        
        for req in tqdm(requests, desc="Generating"):
            context, gen_kwargs = req.args
            
            max_new_tokens = self.max_length
            until = gen_kwargs.get("until", None)
            
            # Beam search parameters
            use_beam_search = True
            num_beams = self.num_beams
            length_penalty = 1.0
            early_stopping = False
            no_repeat_ngram_size = 0

            input_ids = self.tok_encode(context, return_tensors="pt").to(self.device)
            
            output_ids = self._transformers_beam_search(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                temperature=self.temperature,
                length_penalty=length_penalty,
                early_stopping=early_stopping,
                no_repeat_ngram_size=no_repeat_ngram_size,
            )

            generated = output_ids[0][input_ids.shape[1]:]
            generated_text = self.tok_decode(generated, skip_special_tokens=True)
            
            if until is not None:
                if isinstance(until, str):
                    until = [until]
                
                for stop_seq in until:
                    if stop_seq in generated_text:
                        generated_text = generated_text.split(stop_seq)[0]
            
            res.append(generated_text)

        if self.accelerator is not None:
            self.accelerator.wait_for_everyone()
        
        return res

    def apply_chat_template(self, chat_history, add_generation_prompt=True):
        """Apply chat template for few-shot examples"""
        conversation = []
        for msg in chat_history:
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                conversation.append(f"<|im_start|>user\n{content}<|im_end|>")
            elif role == "assistant":
                conversation.append(f"<|im_start|>assistant\n{content}<|im_end|>")
        
        if add_generation_prompt:
            conversation.append("<|im_start|>assistant\n")
        
        return "\n".join(conversation)

    def loglikelihood(self, requests):
        raise NotImplementedError("loglikelihood not implemented for qwen")

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError("loglikelihood_rolling not implemented for qwen")


@register_model("qwen-ets")
class QwenEvalHarnessETS(LM):
    def __init__(
        self,
        dataset='',
        model_path='',
        max_length=512,
        batch_size=1,
        device="cuda",
        dtype=torch.bfloat16,
        m_candidates=3,
        k_monte_carlo=3,
        lambda_param=0.1,
        block_size=64,
        temperature=0.7,
        small_model_path='',
        use_importance_sampling=False,
        **kwargs,
    ):
        super().__init__()
        
        self.dataset = dataset
        self.use_importance_sampling = use_importance_sampling

        self.m_candidates = m_candidates
        self.k_monte_carlo = k_monte_carlo
        self.lambda_param = lambda_param
        self.block_size = block_size

        init_process_group_kwargs = accelerate.InitProcessGroupKwargs(timeout=timedelta(seconds=7200))  # 2 hrs 
        accelerator = accelerate.Accelerator(kwargs_handlers=[init_process_group_kwargs])
        if accelerator.num_processes > 1:
            self.accelerator = accelerator
        else:
            self.accelerator = None
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            trust_remote_code=True,
            padding_side="left"
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs = {
            "torch_dtype": dtype,
            "trust_remote_code": True,
            "device_map": "auto" if self.accelerator is None else None
        }
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            **model_kwargs
        )
        self.model.eval()

        self.device = torch.device(device)
        if use_importance_sampling:
            self.small_model = AutoModelForCausalLM.from_pretrained(
                small_model_path,
                **model_kwargs
            )
            self.small_model.eval()
            if self.accelerator is not None:
                self.small_model = self.accelerator.prepare(self.small_model)
            else:
                self.small_model = self.small_model.to(self.device)
            
            if hasattr(self.small_model, 'module'):
                self.small_model = self.small_model.module
        else:
            self.small_model = None

        if self.accelerator is not None:
            self.model = self.accelerator.prepare(self.model)
            self.device = torch.device(f'{self.accelerator.device}')
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else: 
            self.model = self.model.to(self.device)

        if hasattr(self.model, 'module'):
            self.model = self.model.module

        self.max_length = max_length
        self.batch_size = int(batch_size)
        self.temperature = temperature

    @torch.no_grad()
    def generate_until(self, requests, checkpoint_interval=50):
        res = []
        
        checkpoint_file = f"{self.dataset}-{self.m_candidates}-{self.k_monte_carlo}-{self.block_size}-{self.max_length}-{self.temperature}-ckpt_rank{self.rank}-{self.world_size}.pkl"
        start_idx = 0
        
        if os.path.exists(checkpoint_file):
            try:
                with open(checkpoint_file, 'rb') as f:
                    checkpoint_data = pickle.load(f)
                    res = checkpoint_data['results']
                    start_idx = checkpoint_data['start_index']
                    print(f"[Rank {self.rank}] Resumed from checkpoint, starting from index {start_idx}")
            except:
                print(f"[Rank {self.rank}] Failed to load checkpoint, starting from scratch")

        for req in tqdm(requests[start_idx:], desc=f"[Rank {self.rank}] Generating with Acceleration-based Method, p_small {self.use_importance_sampling}"):
            context, gen_kwargs = req.args
            
            max_length = gen_kwargs.get("max_length", self.max_length)
            until = gen_kwargs.get("until", None)
            do_sample = gen_kwargs.get("do_sample", True)

            extract_fn = None
            if self.dataset == "gsm8k":
                extract_fn = extract_solution
            if self.dataset == "math500":
                extract_fn = lambda x: normalize_final_answer(get_unnormalized_answer(x))
            if self.dataset == "humaneval":
                extract_fn = lambda x: extract_code(x, req.doc["entry_point"])
            if self.dataset == 'gpqa':
                extract_fn = parse_answer_gpqa
            
            input_ids = self.tok_encode(context, return_tensors="pt").to(self.device)

            output_ids = generate(
                self.model, extract_fn, input_ids, self.tokenizer,
                self.max_length, self.block_size, self.m_candidates, self.k_monte_carlo,
                until, self.temperature, self.lambda_param,
                self.use_importance_sampling, self.small_model
            )
            generated_answer = output_ids[0, input_ids.shape[1]:]
            generated_text = self.tokenizer.decode(generated_answer, skip_special_tokens=True)
            
            res.append(generated_text)

            if len(res) % checkpoint_interval == 0 and self.accelerator is not None:
                self.accelerator.wait_for_everyone()
                checkpoint_data = {'results': res, 'start_index': len(res)}
                with open(checkpoint_file, 'wb') as f:
                    pickle.dump(checkpoint_data, f)
                print(f"[Rank {self.rank}] Saved checkpoint at index {len(res)}")
                torch.cuda.empty_cache()
            
        if self.accelerator is not None:
            self.accelerator.wait_for_everyone()

        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)
        
        return res

    @property
    def rank(self):
        return self._rank
    
    @property
    def world_size(self):
        return self._world_size

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def prefix_token_id(self):
        return self.eot_token_id

    def tok_encode(self, string: str, **kwargs) -> list[int]:
        return self.tokenizer.encode(string, **kwargs)

    def loglikelihood(self, requests):
        raise NotImplementedError("loglikelihood not implemented for qwen-accelerate")

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError("loglikelihood_rolling not implemented for qwen-accelerate")


@register_model("qwen-draw")
class QwenEvalHarnessDraw(LM):
    def __init__(
        self,
        dataset='',
        model_path='',
        max_length=512,
        batch_size=1,
        device="cuda",
        dtype=torch.bfloat16,
        **kwargs,
    ):
        super().__init__()
        
        init_process_group_kwargs = accelerate.InitProcessGroupKwargs(timeout=timedelta(seconds=7200))
        accelerator = accelerate.Accelerator(kwargs_handlers=[init_process_group_kwargs])
        if accelerator.num_processes > 1:
            self.accelerator = accelerator
        else:
            self.accelerator = None
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            trust_remote_code=True,
            padding_side="left"
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs = {
            "torch_dtype": dtype,
            "trust_remote_code": True,
            "device_map": "auto" if self.accelerator is None else None
        }
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            **model_kwargs
        )
        self.model.eval()

        self.device = torch.device(device)
        if self.accelerator is not None:
            self.model = self.accelerator.prepare(self.model)
            self.device = torch.device(f'{self.accelerator.device}')
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else: 
            self.model = self.model.to(self.device)

        if hasattr(self.model, 'module'):
            self.model = self.model.module

        self.max_length = max_length
        self.batch_size = int(batch_size)

        self.true_rewards_list = []
        self.logits_rewards_list = []
        self.entropy_rewards_list = []
        self.self_certainty_rewards_list = []
        self.srt_rewards_list = []

    @property
    def rank(self):
        return self._rank
    
    @property
    def world_size(self):
        return self._world_size

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def prefix_token_id(self):
        return self.eot_token_id

    def tok_encode(self, string: str, **kwargs) -> list[int]:
        return self.tokenizer.encode(string, **kwargs)

    def plot_reward_distributions(self):
        if self.accelerator is not None and self.accelerator.local_process_index != 0:
            return

        true_rewards = np.array(self.true_rewards_list)

        def filter_invalid(values):
            values_array = np.array(values)
            valid_mask = np.isfinite(values_array)
            return values_array[valid_mask]
        
        logits_rewards = filter_invalid(self.logits_rewards_list)
        entropy_rewards = filter_invalid(self.entropy_rewards_list)
        self_certainty_rewards = filter_invalid(self.self_certainty_rewards_list)
        srt_rewards = filter_invalid(self.srt_rewards_list)

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        reward_types = [
            ('Logits Reward', logits_rewards),
            ('Entropy Reward', entropy_rewards),
            ('Self-Certainty Reward', self_certainty_rewards),
            ('Self-Consistency Reward', srt_rewards),
        ]

        for idx, (reward_name, reward_values) in enumerate(reward_types):
            ax = axes[idx // 2, idx % 2]
            reward_0 = reward_values[true_rewards == 0]
            reward_1 = reward_values[true_rewards == 1]

            if idx < 3:
                from scipy.stats import gaussian_kde
                if len(reward_0) > 1:
                    kde_0 = gaussian_kde(reward_0)
                    x_vals_0 = np.linspace(reward_0.min(), reward_0.max(), 200)
                    y_vals_0 = kde_0(x_vals_0)
                    ax.plot(x_vals_0, y_vals_0, label='True Reward=0', color='red', linewidth=2, alpha=0.8)
                if len(reward_1) > 1:
                    kde_1 = gaussian_kde(reward_1)
                    x_vals_1 = np.linspace(reward_1.min(), reward_1.max(), 200)
                    y_vals_1 = kde_1(x_vals_1)
                    ax.plot(x_vals_1, y_vals_1, label='True Reward=1', color='blue', linewidth=2, alpha=0.8)
                if len(reward_0) > 1 and len(reward_1) > 1:
                    ax.fill_between(x_vals_0, y_vals_0, alpha=0.2, color='red')
                    ax.fill_between(x_vals_1, y_vals_1, alpha=0.2, color='blue')
                
                ax.set_xlabel(f'{reward_name} Value')
                ax.set_ylabel('Density')
                ax.set_title(f'Distribution of {reward_name}')
                ax.legend()
                
            else:
                if len(reward_0) > 0:
                    ax.hist(reward_0, bins=30, alpha=0.5, label='True Reward=0', color='red', density=True)
                if len(reward_1) > 0:
                    ax.hist(reward_1, bins=30, alpha=0.5, label='True Reward=1', color='blue', density=True)
                
                ax.set_xlabel(f'{reward_name} Value')
                ax.set_ylabel('Density')
                ax.set_title(f'Distribution of {reward_name}')
                ax.legend()
        
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'reward_distributions_{timestamp}.pdf'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        

    @torch.no_grad()
    def generate_until(self, requests):
        """Generate text until stopping condition"""
        res = []
        
        for req_idx, req in enumerate(tqdm(requests, desc="Generating")):
            context, gen_kwargs = req.args
            ground_truth = extract_solution(req.doc.get("answer", ""))
            
            max_length = gen_kwargs.get("max_length", self.max_length)
            temperature = 0.7
            until = gen_kwargs.get("until", None)
            
            input_ids = self.tok_encode(context, return_tensors="pt").to(self.device)
            original_length = input_ids.shape[1]
            
            num_samples = 20
            input_ids_repeated = input_ids.repeat(num_samples, 1)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids_repeated,
                    max_new_tokens=max_length,
                    num_return_sequences=1,
                    do_sample=True,
                    top_p=0.8,
                    top_k=20,
                    temperature=temperature,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                    use_cache=True
                )
            
            all_logits = self.model(outputs, return_dict=True).logits

            generated_texts = []
            for i in range(num_samples):
                generated_ids = outputs[i, original_length:]
                generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                
                if until is not None:
                    if isinstance(until, str):
                        until = [until]
                    for stop_seq in until:
                        if stop_seq in generated_text:
                            generated_text = generated_text.split(stop_seq)[0]
                
                generated_texts.append(generated_text)
            
            extracted_answers = []
            for text in generated_texts:
                answer = extract_solution(text)
                extracted_answers.append(answer)
            
            true_rewards = []
            for ans in extracted_answers:
                if ans is not None and ans == ground_truth:
                    true_rewards.append(1.0)
                else:
                    true_rewards.append(0.0)
            
            logits_rewards = []
            entropy_rewards = []
            self_certainty_rewards = []

            for i in range(num_samples):
                sample_logits = all_logits[i].unsqueeze(0)
                _, logit_reward = compute_sequence_log_probability(
                    self.model, outputs[i].unsqueeze(0), original_length, sample_logits
                )
                logits_rewards.append(logit_reward.item())
                
                entropy_reward = compute_entropy_reward(sample_logits, original_length)
                entropy_rewards.append(entropy_reward)
                
                self_certainty_reward = compute_self_certainty_reward(
                    sample_logits.view(-1, sample_logits.shape[-1])
                )
                self_certainty_rewards.append(self_certainty_reward)
            
            srt_rewards_list, _ = compute_srt_reward(
                extract_fn=extract_solution,
                all_candidate_texts=generated_texts,
                candidate_block_texts_list=[generated_texts],
            )
            srt_rewards = srt_rewards_list[0]
            
            logits_rewards = torch.tensor(logits_rewards)
            logits_rewards = ((logits_rewards - logits_rewards.min()) / (logits_rewards.max() - logits_rewards.min())).tolist()
            self_certainty_rewards = torch.tensor(self_certainty_rewards)
            self_certainty_rewards = ((self_certainty_rewards - self_certainty_rewards.min()) / (self_certainty_rewards.max() - self_certainty_rewards.min())).tolist()
            entropy_rewards = torch.tensor(entropy_rewards)
            entropy_rewards = ((entropy_rewards - entropy_rewards.min()) / (entropy_rewards.max() - entropy_rewards.min())).tolist()
            
            print(true_rewards)
            print(logits_rewards)
            print(self_certainty_rewards)
            print(entropy_rewards)
            print(srt_rewards)
            
            self.logits_rewards_list.extend(logits_rewards)
            self.true_rewards_list.extend(true_rewards)
            self.entropy_rewards_list.extend(entropy_rewards)
            self.self_certainty_rewards_list.extend(self_certainty_rewards)
            self.srt_rewards_list.extend(srt_rewards)
            
            res.append(generated_texts[0])
        
        self.plot_reward_distributions()
        
        if self.accelerator is not None:
            self.accelerator.wait_for_everyone()
        
        return res

    def apply_chat_template(self, chat_history, add_generation_prompt=True):
        """Apply chat template for few-shot examples"""
        conversation = []
        for msg in chat_history:
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                conversation.append(f"<|im_start|>user\n{content}<|im_end|>")
            elif role == "assistant":
                conversation.append(f"<|im_start|>assistant\n{content}<|im_end|>")
        
        if add_generation_prompt:
            conversation.append("<|im_start|>assistant\n")
        
        return "\n".join(conversation)

    def loglikelihood(self, requests):
        raise NotImplementedError("loglikelihood not implemented for qwen")

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError("loglikelihood_rolling not implemented for qwen")


if __name__ == "__main__":
    set_seed(1234)
    cli_evaluate()