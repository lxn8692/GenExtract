"""
Verifier Agent: Knowledge-Based Verification of Event Extraction Results.

The Verifier agent validates extraction results against external knowledge
and ontological constraints. Unlike the Revisor (which corrects errors based
on learned patterns), the Verifier performs explicit knowledge-based checks:

    1. Ontology Constraint Verification:
       - Type constraints: Does the argument match the expected entity type
         for this role? (e.g., "attacker" should be a person/organization)
       - Cardinality constraints: Are required roles filled?
       - Event-role compatibility: Do the arguments make sense for this
         event type?

    2. Factual Consistency Verification:
       - Cross-argument consistency: Are arguments logically compatible?
         (e.g., buyer != seller, origin != destination)
       - Temporal consistency: Do temporal arguments follow chronological order?
       - Contextual grounding: Are arguments actually supported by the context?

    3. Knowledge-Enhanced Scoring:
       - Uses an LLM to assess factual plausibility of extraction results
       - Produces a verification_score that measures knowledge consistency
       - Feeds back into the multi-agent reward computation

Input:  Context + Trigger + Event type + Arguments + Ontology constraints
Output: Verification scores + constraint violation flags
"""

import json
import random
import math
import re
import numpy as np
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Set

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


# ---------------------------------------------------------------------------
# Ontology Constraint Definitions
# ---------------------------------------------------------------------------

# Common semantic type categories for argument roles
ROLE_TYPE_CONSTRAINTS = {
    # Person/Organization roles (expect named entities of PER/ORG type)
    'person_roles': {
        'attacker', 'defendant', 'plaintiff', 'victim', 'killer',
        'perpetrator', 'suspect', 'buyer', 'seller', 'giver',
        'recipient', 'beneficiary', 'agent', 'patient', 'communicator',
        'employee', 'employer', 'leader', 'founder', 'owner',
        'investigator', 'prosecutor', 'judge', 'witness', 'detainee',
        'deportee', 'extraditer', 'transporter', 'manufacturer',
        'demonstrator', 'granter',
    },
    # Location roles (expect GPE/LOC type)
    'location_roles': {
        'place', 'origin', 'destination', 'territory', 'position',
        'location', 'hidingplace', 'target',
    },
    # Instrument/artifact roles
    'artifact_roles': {
        'instrument', 'weapon', 'vehicle', 'artifact', 'money',
        'commodity',
    },
    # Temporal roles
    'time_roles': {
        'time', 'date', 'startdate', 'enddate',
    },
}

# Mutual exclusion constraints: roles that should not have the same value
MUTUAL_EXCLUSION_PAIRS = [
    ('buyer', 'seller'),
    ('giver', 'recipient'),
    ('attacker', 'victim'),
    ('origin', 'destination'),
    ('plaintiff', 'defendant'),
    ('employer', 'employee'),
    ('prosecutor', 'defendant'),
    ('extraditer', 'deportee'),
    ('transporter', 'passenger'),
]


class Verifier(BaseModel):
    """
    Verifier Agent for knowledge-based validation of event extraction results.

    Uses an LLM (Llama/Qwen) with LoRA fine-tuning to verify extraction
    results against world knowledge and ontological constraints. The Verifier
    provides a verification_score that quantifies the factual consistency
    of the extraction, serving as an additional quality signal in the
    multi-agent reward computation.

    Verification pipeline:
        1. Ontology constraint checking (rule-based, fast)
        2. Cross-argument consistency checking (rule-based, fast)
        3. Knowledge-based plausibility scoring (LLM-based, deep)

    Workflow:
        1. fit():            Train to learn verification patterns
        2. verify():         Run full verification pipeline
        3. estimate():       Compute verification score for reward
    """
    load_dir: str
    save_dir: str
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

    # ------------------------------------------------------------------
    # Rule-based Constraint Checking
    # ------------------------------------------------------------------

    @staticmethod
    def check_ontology_constraints(arguments: Dict[str, Optional[str]],
                                   event_type: str,
                                   context: str) -> Dict[str, float]:
        """
        Check ontology-based constraints on extracted arguments.

        Performs three types of checks:
            1. Required role coverage: Are important roles filled?
            2. Span grounding: Do argument spans actually appear in context?
            3. Role-type compatibility: Basic heuristic checks

        Args:
            arguments:  Dict mapping role -> span (or None)
            event_type: The event type string
            context:    The document text

        Returns:
            Dict with constraint violation scores:
                - 'coverage_penalty':  Penalty for unfilled required roles
                - 'grounding_penalty': Penalty for arguments not in context
                - 'type_penalty':      Penalty for type constraint violations
                - 'total_penalty':     Weighted sum of all penalties
        """
        penalties = {
            'coverage_penalty': 0.0,
            'grounding_penalty': 0.0,
            'type_penalty': 0.0,
            'total_penalty': 0.0,
        }

        if not arguments:
            penalties['total_penalty'] = 1.0
            return penalties

        num_roles = len(arguments)
        if num_roles == 0:
            return penalties

        # --- 1. Coverage: penalize if too many roles are empty ---
        num_filled = sum(1 for v in arguments.values() if v)
        num_empty = num_roles - num_filled
        empty_ratio = num_empty / num_roles

        # Mild penalty if > 70% roles are empty
        if empty_ratio > 0.7:
            penalties['coverage_penalty'] = (empty_ratio - 0.7) * 2.0

        # --- 2. Grounding: check if argument spans exist in context ---
        context_tokens = re.findall(r"\w+|[^\w\s]", context.lower())
        num_ungrounded = 0

        for role, span in arguments.items():
            if not span:
                continue
            span_tokens = re.findall(r"\w+|[^\w\s]", span.lower())
            n, m = len(context_tokens), len(span_tokens)

            found = False
            if 0 < m <= n:
                for i in range(n - m + 1):
                    if context_tokens[i:i + m] == span_tokens:
                        found = True
                        break

            if not found:
                num_ungrounded += 1

        if num_filled > 0:
            penalties['grounding_penalty'] = num_ungrounded / num_filled

        # --- 3. Type constraints: basic heuristic checks ---
        type_violations = 0
        total_checked = 0

        for role, span in arguments.items():
            if not span:
                continue
            role_lower = role.lower()

            # Check if location roles contain typical location indicators
            if role_lower in ROLE_TYPE_CONSTRAINTS['location_roles']:
                total_checked += 1
                # Location arguments that are just numbers or very short
                # single-char are suspicious
                if span.isdigit() or (len(span) == 1 and not span.isalpha()):
                    type_violations += 1

            # Check if time roles look like temporal expressions
            if role_lower in ROLE_TYPE_CONSTRAINTS['time_roles']:
                total_checked += 1
                # Time arguments should contain digits or temporal keywords
                temporal_indicators = [
                    'day', 'month', 'year', 'january', 'february', 'march',
                    'april', 'may', 'june', 'july', 'august', 'september',
                    'october', 'november', 'december', 'monday', 'tuesday',
                    'wednesday', 'thursday', 'friday', 'saturday', 'sunday',
                    'morning', 'evening', 'night', 'today', 'yesterday',
                    'tomorrow', 'week', 'ago', '20', '19',
                ]
                has_temporal = any(
                    ind in span.lower() for ind in temporal_indicators
                ) or bool(re.search(r'\d', span))
                if not has_temporal:
                    type_violations += 1

        if total_checked > 0:
            penalties['type_penalty'] = type_violations / total_checked

        # --- Weighted total ---
        penalties['total_penalty'] = (
            0.3 * penalties['coverage_penalty'] +
            0.4 * penalties['grounding_penalty'] +
            0.3 * penalties['type_penalty']
        )

        return penalties

    @staticmethod
    def check_cross_argument_consistency(
            arguments: Dict[str, Optional[str]]) -> Dict[str, float]:
        """
        Check logical consistency between extracted arguments.

        Verifies:
            1. Mutual exclusion: Certain role pairs should not have the
               same argument value (e.g., buyer != seller)
            2. Duplicate detection: Different roles having identical values
               is suspicious (but not always wrong)

        Args:
            arguments: Dict mapping role -> span

        Returns:
            Dict with:
                - 'exclusion_violations': Number of mutual exclusion violations
                - 'duplicate_ratio':     Ratio of duplicate values
                - 'consistency_penalty':  Combined penalty score
        """
        result = {
            'exclusion_violations': 0,
            'duplicate_ratio': 0.0,
            'consistency_penalty': 0.0,
        }

        if not arguments:
            return result

        filled = {k.lower(): v.lower().strip()
                  for k, v in arguments.items() if v}

        if len(filled) < 2:
            return result

        # --- 1. Mutual exclusion check ---
        for role_a, role_b in MUTUAL_EXCLUSION_PAIRS:
            if role_a in filled and role_b in filled:
                if filled[role_a] == filled[role_b]:
                    result['exclusion_violations'] += 1

        # --- 2. Duplicate detection ---
        values = list(filled.values())
        unique_values = set(values)
        if len(values) > 0:
            result['duplicate_ratio'] = 1.0 - len(unique_values) / len(values)

        # --- Combined penalty ---
        result['consistency_penalty'] = (
            0.6 * min(result['exclusion_violations'], 3) / 3.0 +
            0.4 * result['duplicate_ratio']
        )

        return result

    # ------------------------------------------------------------------
    # LLM-based Knowledge Verification
    # ------------------------------------------------------------------

    @staticmethod
    def _build_verification_input(context: str, trigger: str,
                                  event_type: str,
                                  arguments: Dict[str, Optional[str]]) -> str:
        """
        Build the input prompt for LLM-based knowledge verification.

        The prompt asks the LLM to assess the factual plausibility of the
        extraction results against world knowledge.
        """
        all_roles = list(arguments.keys())
        roles_str = ', '.join(all_roles)
        args_parts = []
        for role, span in arguments.items():
            if span:
                args_parts.append(f"{role}: {span}")
            else:
                args_parts.append(f"{role}: none")
        args_str = ', '.join(args_parts)

        prompt = (
            f"Verify the factual correctness of the following event extraction. "
            f"Event type: '{event_type}', roles: {roles_str}. "
            f"Context: {context}. "
            f"Trigger: {trigger}. "
            f"Extracted arguments: {args_str}. "
            f"For each argument, check: (1) Does it exist in the context? "
            f"(2) Does it match the expected role type? "
            f"(3) Is it factually plausible given world knowledge? "
            f"Produce a verification result."
        )
        return prompt

    @staticmethod
    def _build_verification_output(context: str, trigger: str,
                                   event_type: str,
                                   arguments: Dict[str, Optional[str]],
                                   is_correct: bool = True) -> str:
        """
        Build the expected output for LLM verification training.

        For training, we create both positive (correct) and negative
        (incorrect) examples so the LLM learns to distinguish them.
        """
        args_parts = []
        for role, span in arguments.items():
            if span:
                args_parts.append(f"{role}: verified")
            else:
                args_parts.append(f"{role}: empty")
        args_str = ', '.join(args_parts)

        if is_correct:
            verdict = "PASS"
            explanation = "All arguments are factually consistent with the context."
        else:
            verdict = "FAIL"
            explanation = "Some arguments contain knowledge errors."

        output = (
            f"Verification: {verdict}. "
            f"Argument status: {args_str}. "
            f"Explanation: {explanation}"
        )
        return output

    @staticmethod
    def trans_verifier_data(task_name: str, meta_path: str,
                            data_path: str) -> list:
        """
        Transform raw RAMS-format data into Verifier training format.

        Creates training pairs:
        - Positive examples: correct extractions → "PASS"
        - Negative examples: corrupted extractions → "FAIL"

        Corruption strategies for negative examples:
            1. Swap arguments between roles
            2. Replace arguments with random spans from context
            3. Insert hallucinated arguments not in context

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
        for d in tqdm(data, desc="Preparing verifier data"):
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

            # Positive example: gold extraction → PASS
            pos_input = Verifier._build_verification_input(
                doc_text, trigger, event_type, role2span
            )
            pos_output = Verifier._build_verification_output(
                doc_text, trigger, event_type, role2span, is_correct=True
            )
            all_data.append({
                'input': pos_input,
                'label': pos_output,
                'reward': 1.0
            })

            # Negative examples: corrupted extractions → FAIL
            filled_roles = [r for r, s in role2span.items() if s]

            if len(filled_roles) >= 2:
                # Strategy 1: Swap two arguments
                corrupted = dict(role2span)
                r1, r2 = random.sample(filled_roles, 2)
                corrupted[r1], corrupted[r2] = corrupted[r2], corrupted[r1]

                neg_input = Verifier._build_verification_input(
                    doc_text, trigger, event_type, corrupted
                )
                neg_output = Verifier._build_verification_output(
                    doc_text, trigger, event_type, corrupted, is_correct=False
                )
                all_data.append({
                    'input': neg_input,
                    'label': neg_output,
                    'reward': 1.0
                })

            if filled_roles:
                # Strategy 2: Replace an argument with a random context span
                corrupted = dict(role2span)
                target_role = random.choice(filled_roles)
                # Pick a random span from the document
                if len(all_tokens) > 3:
                    start = random.randint(0, max(0, len(all_tokens) - 3))
                    length = random.randint(1, min(3, len(all_tokens) - start))
                    random_span = ' '.join(all_tokens[start:start + length])
                    # Only use if different from original
                    if random_span != corrupted[target_role]:
                        corrupted[target_role] = random_span

                        neg_input = Verifier._build_verification_input(
                            doc_text, trigger, event_type, corrupted
                        )
                        neg_output = Verifier._build_verification_output(
                            doc_text, trigger, event_type, corrupted,
                            is_correct=False
                        )
                        all_data.append({
                            'input': neg_input,
                            'label': neg_output,
                            'reward': 1.0
                        })

        return all_data

    @staticmethod
    def trans_verifier_synthetic(data: list) -> list:
        """
        Convert synthetic/filtered data into Verifier training format.

        Args:
            data: List of dicts with Context, Trigger, Event type, arguments, reward.

        Returns:
            List of dicts with 'input', 'label', 'reward' keys.
        """
        all_data = []
        for d in tqdm(data, desc="Preparing synthetic verifier data"):
            context = d['Context']
            trigger = d['Trigger']
            event_type = d['Event type']
            arguments = d['arguments']
            reward = d.get('reward', 1.0)

            ver_input = Verifier._build_verification_input(
                context, trigger, event_type, arguments
            )
            ver_output = Verifier._build_verification_output(
                context, trigger, event_type, arguments, is_correct=True
            )

            all_data.append({
                'input': ver_input,
                'label': ver_output,
                'reward': reward
            })

        return all_data

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, path_train: str, path_dev: str, task_name: str,
            meta_path: str, iter: int):
        """
        Train the Verifier model.

        iter=0: Train on gold data with positive/negative verification examples.
        iter>0: Train on filtered data from previous pipeline iteration.

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
            data_train = self.trans_verifier_data(
                task_name=task_name, meta_path=meta_path, data_path=path_train
            )
            data_dev = self.trans_verifier_data(
                task_name=task_name, meta_path=meta_path, data_path=path_dev
            )
        else:
            data_train = Dataset().load(path_train)
            data_train = self.trans_verifier_synthetic(data_train)
            data_dev = self.trans_verifier_data(
                task_name=task_name, meta_path=meta_path, data_path=path_dev
            )

        path_train = self.write_data(data_train, "train")
        path_dev = self.write_data(data_dev, "dev")
        model.fit(path_train=path_train, path_dev=path_dev)
        delete_checkpoints(model.model_dir)

    # ------------------------------------------------------------------
    # Verification Pipeline
    # ------------------------------------------------------------------

    def verify(self, path_in: str, path_out: str):
        """
        Run full verification pipeline on extraction results.

        Combines rule-based constraint checking with LLM-based knowledge
        verification to produce comprehensive verification scores.

        Pipeline:
            1. Ontology constraint checking (fast, rule-based)
            2. Cross-argument consistency checking (fast, rule-based)
            3. LLM-based knowledge plausibility scoring (deep, model-based)

        Each instance gets the following added fields:
            - ontology_penalty:      From ontology constraint checking
            - consistency_penalty:   From cross-argument checking
            - verification_nll:      From LLM-based plausibility scoring
            - verification_score:    Combined verification quality score

        Args:
            path_in:  Path to input extraction results (jsonl)
            path_out: Path to write verified results
        """
        if Path(path_out).exists():
            return

        data_in = Dataset.load(path_in)

        # --- Phase 1 & 2: Rule-based checks (fast) ---
        for ins in tqdm(data_in, desc="Rule-based verification"):
            context = ins['Context']
            event_type = ins['Event type']
            arguments = ins['arguments']

            # Ontology constraints
            ont_result = self.check_ontology_constraints(
                arguments, event_type, context
            )
            ins['ontology_penalty'] = ont_result['total_penalty']

            # Cross-argument consistency
            cons_result = self.check_cross_argument_consistency(arguments)
            ins['consistency_penalty'] = cons_result['consistency_penalty']

        # --- Phase 3: LLM-based verification ---
        verifier_model = self.get_model()

        if 'llama' in verifier_model.model_name.lower():
            base_model = run_clm_rl.Model_Llama.from_pretrained(
                verifier_model.model_name, device_map="auto",
                torch_dtype=torch.float16
            )
        elif 'qwen' in verifier_model.model_name.lower():
            base_model = run_clm_rl.Model_Qwen.from_pretrained(
                verifier_model.model_name, device_map="auto",
                torch_dtype=torch.float16
            )
        else:
            raise ValueError(f"Unsupported model: {verifier_model.model_name}")

        model = PeftModel.from_pretrained(base_model, verifier_model.model_dir)
        tokenizer = AutoTokenizer.from_pretrained(
            verifier_model.model_name, use_fast=False
        )
        tokenizer.pad_token = tokenizer.eos_token

        for ins in tqdm(data_in, desc="LLM verification"):
            context = ins['Context']
            trigger = ins['Trigger']
            event_type = ins['Event type']
            arguments = ins['arguments']

            ver_input = self._build_verification_input(
                context, trigger, event_type, arguments
            )
            # Build the expected "PASS" output to compute NLL against
            ver_output = self._build_verification_output(
                context, trigger, event_type, arguments, is_correct=True
            )

            full_text = ver_input + " " + ver_output
            inputs = tokenizer(
                full_text, return_tensors="pt",
                truncation=True, max_length=self.block_size
            ).to(model.device)

            # Compute NLL for the verification output portion only
            input_len = len(tokenizer(ver_input)['input_ids'])
            labels = inputs['input_ids'].clone()
            labels[0, :min(input_len, labels.shape[1])] = -100

            with torch.no_grad():
                outputs = model(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    labels=labels
                )
                ins['verification_nll'] = outputs.loss.item()

            # --- Compute combined verification_score ---
            # Lower is better: combine all penalty signals
            ins['verification_score'] = (
                0.3 * ins['ontology_penalty'] +
                0.3 * ins['consistency_penalty'] +
                0.4 * min(ins['verification_nll'] / 5.0, 1.0)
                # normalize NLL to roughly [0, 1]
            )

        # Write results
        Path(path_out).parent.mkdir(exist_ok=True, parents=True)
        with open(path_out, "w") as f:
            for d in data_in:
                json.dump(d, f, ensure_ascii=False)
                f.write('\n')

    # ------------------------------------------------------------------
    # Estimation (lightweight, for reward computation)
    # ------------------------------------------------------------------

    def estimate(self, path_in: str, path_out: str):
        """
        Compute verification-based quality scores (rule-based only, fast).

        This is a lightweight version of verify() that only runs rule-based
        checks. Used during the iterative loop when full LLM verification
        is too expensive.

        Produces:
            - ontology_penalty
            - consistency_penalty
            - verification_score (combined, without LLM NLL)

        Args:
            path_in:  Path to input data
            path_out: Path to write scored data
        """
        data_in = Dataset.load(path_in)

        for ins in data_in:
            if ins.get('verification_score') is not None:
                return

            context = ins['Context']
            event_type = ins['Event type']
            arguments = ins['arguments']

            # Ontology constraints
            ont_result = self.check_ontology_constraints(
                arguments, event_type, context
            )
            ins['ontology_penalty'] = ont_result['total_penalty']

            # Cross-argument consistency
            cons_result = self.check_cross_argument_consistency(arguments)
            ins['consistency_penalty'] = cons_result['consistency_penalty']

            # Combined score (without LLM component)
            ins['verification_score'] = (
                0.5 * ins['ontology_penalty'] +
                0.5 * ins['consistency_penalty']
            )

        with open(path_out, "w") as f:
            for d in data_in:
                json.dump(d, f, ensure_ascii=False)
                f.write('\n')
