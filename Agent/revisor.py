"""
Revisor Agent: Review and Revise Event Extraction Results.

The Revisor agent reviews extraction results from the Extractor and revises
them to improve quality. It uses an LLM (Llama/Qwen) with LoRA fine-tuning
to identify and correct extraction errors such as:
  - Incorrect argument spans
  - Missing arguments that should have been extracted
  - Spurious arguments that don't belong

The Revisor also provides a confidence score (revision_nll) that indicates
how much revision was needed, serving as an additional quality signal in
the multi-agent reward computation.

Input:  Context + Trigger + Event type + Extracted arguments
Output: Revised arguments + revision confidence score
"""

import json
import random
import math
import re
import numpy as np
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Dict, Tuple

import torch
import pandas as pd
from torch.nn import CrossEntropyLoss
from pydantic.main import BaseModel
from tqdm import tqdm
from peft import PeftModel, LoraConfig, TaskType
from transformers import AutoTokenizer

from modeling_rl import BaseEAEModel, select_model
from utils import delete_checkpoints
from Dataset import Dataset
from transformer_base import run_clm_rl


class Revisor(BaseModel):
    """
    Revisor Agent for reviewing and revising event extraction results.

    Uses an LLM (Llama/Qwen) with LoRA fine-tuning. The Revisor is trained
    to take extraction results as input and produce corrected/revised results.
    It acts as a quality control layer in the 3-agent pipeline.

    Workflow:
        1. fit():      Train on labeled data to learn revision patterns
        2. revise():   Review and revise extraction results
        3. estimate(): Compute revision confidence for reward scoring
    """
    load_dir: str
    save_dir: str
    num_gen_per_label: int = 250
    model_name: str = "generate"  # reuses CLM model infrastructure
    encoder_name: str = "generate"
    lora_dir: Optional[str] = None
    block_size: int = 512
    model_kwargs: dict = {}

    def get_model(self) -> BaseEAEModel:
        model = select_model(
            name=self.model_name,
            encoder_name=self.encoder_name,
            model_dir=str(Path(self.save_dir) / "model"),
            model_name=self.load_dir,
            lora_dir=self.lora_dir,
            data_dir=str(Path(self.save_dir) / "data"),
            do_pretrain=False,
            **self.model_kwargs,
        )
        return model

    def write_data(self, data: list, name: str) -> str:
        model = self.get_model()
        path_out = Path(model.data_dir) / f"{name}.json"
        path_out.parent.mkdir(exist_ok=True, parents=True)
        random.shuffle(data)
        for d in data:
            with open(path_out, "a") as f:
                json.dump(d, f, ensure_ascii=False)
                f.write('\n')
        return str(path_out)

    @staticmethod
    def _format_arguments(arguments: Dict[str, Optional[str]]) -> str:
        """Format arguments dict into a readable string."""
        parts = []
        for role, span in arguments.items():
            if span:
                parts.append(f"{role}: {span}")
            else:
                parts.append(f"{role}: none")
        return ', '.join(parts)

    @staticmethod
    def _build_revision_input(context: str, trigger: str, event_type: str,
                              arguments: Dict[str, Optional[str]]) -> str:
        """
        Build the input prompt for the Revisor.

        The prompt presents the extraction results and asks the LLM to review
        and produce corrected results.
        """
        all_roles = list(arguments.keys())
        roles_str = ', '.join(all_roles)
        args_str = Revisor._format_arguments(arguments)

        prompt = (
            f"Review and revise the following event extraction result. "
            f"Given the event type: '{event_type}' with roles: {roles_str}. "
            f"Context: {context}. "
            f"Detected trigger: {trigger}. "
            f"Extracted arguments: {args_str}. "
            f"Please verify each argument against the context and produce "
            f"the corrected extraction result."
        )
        return prompt

    @staticmethod
    def _build_revision_output(context: str, trigger: str, event_type: str,
                               arguments: Dict[str, Optional[str]]) -> str:
        """
        Build the expected output for the Revisor (the gold/correct result).
        """
        args_str = Revisor._format_arguments(arguments)
        output = (
            f"Revised extraction: trigger: {trigger}, "
            f"event type: {event_type}, "
            f"arguments: {args_str}."
        )
        return output

    @staticmethod
    def trans_revisor_data(task_name: str, meta_path: str, data_path: str) -> list:
        """
        Transform raw RAMS-format data into Revisor training format.

        Creates training pairs where:
        - Input: Context + trigger + event_type + (possibly noisy) arguments
        - Output: Corrected arguments

        For initial training (iter=0), we use gold arguments as both input
        and output. During later iterations, the Revisor learns to correct
        errors from synthetic data.

        Args:
            task_name: Dataset name
            meta_path: Path to meta.json
            data_path: Path to raw data file

        Returns:
            List of dicts with 'input', 'label', 'reward' keys.
        """
        with open(data_path) as f:
            data = [json.loads(line) for line in f]

        meta = json.load(open(meta_path, 'r', encoding='utf8'))
        type2roles = {}
        for rams_m in meta:
            rams_event_type, role_types = rams_m
            role_types = [
                r[11:].lower() if 'evt' in r else r.lower()
                for r in role_types
            ]
            type2roles[rams_event_type] = role_types

        all_data = []
        for d in tqdm(data, desc="Preparing revisor data"):
            sentences = d['sentences']
            all_tokens = []
            for sentence in sentences:
                all_tokens.extend(sentence)
            doc_text = ' '.join(all_tokens)

            event_type = d['evt_triggers'][0][-1][0][0]
            trigger_s, trigger_e = d['evt_triggers'][0][0], d['evt_triggers'][0][1]
            trigger = ' '.join(all_tokens[trigger_s:trigger_e + 1])

            gold_evt_links = d['gold_evt_links']
            all_roles = type2roles[event_type]

            role2span = {}
            for role in all_roles:
                role2span[role.lower()] = None

            for gold_evt_link in gold_evt_links:
                current_role = gold_evt_link[-1].lower()
                if 'evt' in current_role:
                    current_role = current_role[11:]
                s, e = gold_evt_link[1]
                span = ' '.join(all_tokens[s:e + 1])
                role2span[current_role] = span

            # Build revision input and output
            revision_input = Revisor._build_revision_input(
                doc_text, trigger, event_type, role2span
            )
            revision_output = Revisor._build_revision_output(
                doc_text, trigger, event_type, role2span
            )

            new_d = {
                'input': revision_input,
                'label': revision_output,
                'reward': 1.0
            }
            all_data.append(new_d)

            # Data augmentation: create noisy versions for training robustness
            # Randomly drop some arguments to simulate extraction errors
            if len(gold_evt_links) > 0:
                noisy_role2span = dict(role2span)
                roles_with_spans = [r for r, s in noisy_role2span.items() if s]
                if roles_with_spans:
                    # Drop a random argument
                    drop_role = random.choice(roles_with_spans)
                    noisy_role2span[drop_role] = None

                    noisy_input = Revisor._build_revision_input(
                        doc_text, trigger, event_type, noisy_role2span
                    )
                    # Output should still be the gold (complete) result
                    noisy_output = Revisor._build_revision_output(
                        doc_text, trigger, event_type, role2span
                    )
                    all_data.append({
                        'input': noisy_input,
                        'label': noisy_output,
                        'reward': 1.0
                    })

        return all_data

    @staticmethod
    def trans_revisor_synthetic(data: list) -> list:
        """
        Convert synthetic/filtered data into Revisor training format.

        Args:
            data: List of dicts with Context, Trigger, Event type, arguments, reward.

        Returns:
            List of dicts with 'input', 'label', 'reward' keys.
        """
        all_data = []
        for d in tqdm(data, desc="Preparing synthetic revisor data"):
            context = d['Context']
            trigger = d['Trigger']
            event_type = d['Event type']
            arguments = d['arguments']
            reward = d.get('reward', 1.0)

            revision_input = Revisor._build_revision_input(
                context, trigger, event_type, arguments
            )
            revision_output = Revisor._build_revision_output(
                context, trigger, event_type, arguments
            )

            new_d = {
                'input': revision_input,
                'label': revision_output,
                'reward': reward
            }
            all_data.append(new_d)

        return all_data

    def fit(self, path_train: str, path_dev: str, task_name: str,
            meta_path: str, iter: int):
        """
        Train the Revisor model.

        Args:
            path_train: Path to training data
            path_dev:   Path to dev data
            task_name:  Dataset name
            meta_path:  Path to meta.json
            iter:       Current iteration (0 = initial training)
        """
        model = self.get_model()
        if Path(model.model_dir).exists():
            return

        if iter == 0:
            data_train = self.trans_revisor_data(
                task_name=task_name, meta_path=meta_path, data_path=path_train
            )
            data_dev = self.trans_revisor_data(
                task_name=task_name, meta_path=meta_path, data_path=path_dev
            )
        else:
            data_train = Dataset().load(path_train)
            data_train = self.trans_revisor_synthetic(data_train)
            data_dev = self.trans_revisor_data(
                task_name=task_name, meta_path=meta_path, data_path=path_dev
            )

        path_train = self.write_data(data_train, "train")
        path_dev = self.write_data(data_dev, "dev")
        model.fit(path_train=path_train, path_dev=path_dev)
        delete_checkpoints(model.model_dir)

    def revise(self, path_in: str, path_out: str):
        """
        Revise extraction results using the trained Revisor model.

        Takes extraction results (Context, Trigger, Event type, arguments)
        and produces revised/corrected versions.

        Args:
            path_in:  Path to extraction results (jsonl)
            path_out: Path to write revised results
        """
        if Path(path_out).exists():
            return

        revisor_model = self.get_model()

        if 'llama' in revisor_model.model_name.lower():
            base_model = run_clm_rl.Model_Llama.from_pretrained(
                revisor_model.model_name, device_map="auto", torch_dtype=torch.float16
            )
        elif 'qwen' in revisor_model.model_name.lower():
            base_model = run_clm_rl.Model_Qwen.from_pretrained(
                revisor_model.model_name, device_map="auto", torch_dtype=torch.float16
            )
        else:
            raise ValueError(f"Unsupported model: {revisor_model.model_name}")

        model = PeftModel.from_pretrained(base_model, revisor_model.model_dir)
        tokenizer = AutoTokenizer.from_pretrained(
            revisor_model.model_name, use_fast=False
        )
        tokenizer.pad_token = tokenizer.eos_token

        data_in = Dataset.load(path_in)
        all_revised = []

        for ins in tqdm(data_in, desc="Revising"):
            context = ins['Context']
            trigger = ins['Trigger']
            event_type = ins['Event type']
            arguments = ins['arguments']

            revision_input = self._build_revision_input(
                context, trigger, event_type, arguments
            )

            inputs = tokenizer(revision_input, return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=self.block_size,
                    do_sample=False,
                    top_k=50,
                    top_p=0.95,
                    temperature=0.3,
                    num_return_sequences=1,
                    pad_token_id=tokenizer.pad_token_id,
                )

            result_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            revised_args = self._parse_revision(result_text, list(arguments.keys()))

            # If revision produced valid results, use them; otherwise keep original
            if revised_args:
                revised_ins = dict(ins)
                revised_ins['arguments'] = revised_args
                revised_ins['revised'] = True
                all_revised.append(revised_ins)
            else:
                ins['revised'] = False
                all_revised.append(ins)

        Path(path_out).parent.mkdir(exist_ok=True, parents=True)
        with open(path_out, "w") as f:
            for d in all_revised:
                json.dump(d, f, ensure_ascii=False)
                f.write('\n')

    @staticmethod
    def _parse_revision(text: str, expected_roles: List[str]) -> Optional[Dict[str, Optional[str]]]:
        """
        Parse the Revisor's output text into structured arguments.

        Expected format in the output:
            "... arguments: role1: span1, role2: span2, ..."

        Args:
            text:           Raw output text from Revisor
            expected_roles: List of expected role names

        Returns:
            Dict mapping role names to spans, or None if parsing fails.
        """
        try:
            args_text = text.split("arguments:")[1].strip().rstrip('.')
        except (IndexError, AttributeError):
            return None

        role2span = {}
        for chunk in args_text.split(','):
            if ':' not in chunk:
                continue
            role, val = chunk.strip().split(':', 1)
            role = role.strip().lower()
            val = val.strip().strip('"').rstrip('.')

            if role in [r.lower() for r in expected_roles]:
                if val.lower() == "none":
                    val = None
                role2span[role] = val

        if not role2span:
            return None

        return role2span

    def estimate(self, path_in: str, path_out: str, path_train: str,
                 task_name: str, meta_path: str):
        """
        Compute revision-based quality scores for synthetic data.

        Evaluates each synthetic instance by computing:
        1. revision_nll: NLL of the Revisor reproducing the extraction result
           (lower = more consistent with the Revisor's learned patterns)
        2. consistency_penalty: Penalty for arguments not found in context

        These scores are incorporated into the multi-agent reward computation.

        Args:
            path_in:     Path to input synthetic data
            path_out:    Path to write scored data
            path_train:  Path to original training data (for statistics)
            task_name:   Dataset name
            meta_path:   Path to meta.json
        """
        data_in = Dataset.load(path_in)

        for ins in data_in:
            if ins.get('revision_penalty') is not None:
                return

            context = ins['Context']
            trigger = ins['Trigger']
            arguments = ins['arguments']

            # Compute consistency penalty:
            # Check that each non-None argument actually appears in the context
            penalty = 0.0
            num_args = 0
            num_inconsistent = 0

            context_tokens = re.findall(r"\w+|[^\w\s]", context)

            for role, span in arguments.items():
                num_args += 1
                if not span:
                    continue
                span_tokens = re.findall(r"\w+|[^\w\s]", span)
                n, m = len(context_tokens), len(span_tokens)

                found = False
                if m > 0 and m <= n:
                    for i in range(n - m + 1):
                        if context_tokens[i:i + m] == span_tokens:
                            found = True
                            break

                if not found:
                    num_inconsistent += 1

            if num_args > 0:
                penalty = num_inconsistent / num_args

            # Also compute none_ratio penalty (similar to Generator)
            num_none = sum(1 for v in arguments.values() if not v)
            none_ratio = num_none / max(num_args, 1)

            # Combined penalty
            ins['revision_penalty'] = penalty + max(0, none_ratio - 0.7) * 0.5

        with open(path_out, "w") as f:
            for d in data_in:
                json.dump(d, f, ensure_ascii=False)
                f.write('\n')

    def compute_revision_nll(self, path_in: str, path_out: str):
        """
        Compute revision NLL using the Revisor's LLM.

        For each instance, computes how likely the Revisor model would
        generate the current extraction result. This serves as an
        additional quality signal: if the Revisor "agrees" with the
        extraction (low NLL), the result is likely high quality.

        Args:
            path_in:  Path to input data
            path_out: Path to write data with 'revision_nll' field
        """
        revisor_model = self.get_model()

        if 'llama' in revisor_model.model_name.lower():
            base_model = run_clm_rl.Model_Llama.from_pretrained(
                revisor_model.model_name, device_map="auto", torch_dtype=torch.float16
            )
        elif 'qwen' in revisor_model.model_name.lower():
            base_model = run_clm_rl.Model_Qwen.from_pretrained(
                revisor_model.model_name, device_map="auto", torch_dtype=torch.float16
            )
        else:
            raise ValueError(f"Unsupported model: {revisor_model.model_name}")

        model = PeftModel.from_pretrained(base_model, revisor_model.model_dir)
        tokenizer = AutoTokenizer.from_pretrained(
            revisor_model.model_name, use_fast=False
        )
        tokenizer.pad_token = tokenizer.eos_token

        data_in = Dataset.load(path_in)

        for ins in tqdm(data_in, desc="Computing revision NLL"):
            if ins.get('revision_nll') is not None:
                continue

            context = ins['Context']
            trigger = ins['Trigger']
            event_type = ins['Event type']
            arguments = ins['arguments']

            revision_input = self._build_revision_input(
                context, trigger, event_type, arguments
            )
            revision_output = self._build_revision_output(
                context, trigger, event_type, arguments
            )

            full_text = revision_input + " " + revision_output
            inputs = tokenizer(full_text, return_tensors="pt",
                               truncation=True, max_length=self.block_size).to(model.device)

            # Compute NLL for the output portion only
            input_len = len(tokenizer(revision_input)['input_ids'])
            labels = inputs['input_ids'].clone()
            labels[0, :input_len] = -100  # mask input portion

            with torch.no_grad():
                outputs = model(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    labels=labels
                )
                ins['revision_nll'] = outputs.loss.item()

        with open(path_out, "w") as f:
            for d in data_in:
                json.dump(d, f, ensure_ascii=False)
                f.write('\n')
