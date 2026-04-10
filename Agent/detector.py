"""
Detector Agent: Event Trigger Detection and Event Type Classification.

The Detector agent is responsible for identifying event triggers in documents
and classifying their event types. It uses a BART-based seq2seq model that
takes raw document text as input and generates structured output containing
detected triggers and their event types.

Input:  Raw document text
Output: List of (trigger, event_type) pairs
"""

import json
import random
import math
import re
import numpy as np
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Dict

import torch
import pandas as pd
from torch.nn import CrossEntropyLoss
from pydantic.main import BaseModel
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from modeling_rl import EAE_Extractor, BaseEAEModel, select_model
from utils import delete_checkpoints
from Dataset import Dataset


class Detector(BaseModel):
    """
    Detector Agent for event trigger detection and type classification.

    Uses a BART-based encoder-decoder model to detect event triggers
    in document text and classify their event types.

    Workflow:
        1. fit():      Train on labeled data to learn trigger detection
        2. predict():  Detect triggers and event types from raw documents
        3. estimate(): Compute detection confidence (NLL) for quality scoring
    """
    load_dir: str
    save_dir: str
    model_name: str = "extractor"  # reuses BART-based model infrastructure
    encoder_name: str = "extract"
    model_kwargs: dict = {}

    def get_model(self) -> BaseEAEModel:
        model = select_model(
            name=self.model_name,
            encoder_name=self.encoder_name,
            model_dir=str(Path(self.save_dir) / "model"),
            model_name=self.load_dir,
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
    def trans_detector_data(task_name: str, meta_path: str, data_path: str,
                            ontology_dict_path: str, task: str = 'train') -> list:
        """
        Transform raw RAMS-format data into Detector training format.

        The Detector learns to generate a structured detection string from
        document context. The input is the raw document text, and the output
        is a formatted string: "trigger: <trigger_word> ; event type: <type>"

        Args:
            task_name:         Dataset name (e.g. 'rams')
            meta_path:         Path to meta.json with event type -> role mappings
            data_path:         Path to raw data file (jsonl)
            ontology_dict_path: Path to ontology CSV
            task:              'train' or 'eval'

        Returns:
            List of dicts with keys:
                - input_context: tokenized document with no trigger markers
                - summary:       "trigger: X ; event type: Y"
                - reward:        1.0 (for training data)
                - doc_key:       (only for eval)
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
        for d in tqdm(data, desc="Preparing detector data"):
            sentences = d['sentences']
            all_tokens = []
            for sentence in sentences:
                all_tokens.extend(sentence)

            event_type = d['evt_triggers'][0][-1][0][0]
            trigger_s, trigger_e = d['evt_triggers'][0][0], d['evt_triggers'][0][1]
            trigger = ' '.join(all_tokens[trigger_s:trigger_e + 1])

            # Input: raw document tokens (no trigger markers)
            input_context = all_tokens

            # Output: structured detection result
            summary = f"trigger: {trigger} ; event type: {event_type}"

            new_d = {
                "input_context": input_context,
                "summary": summary,
                "reward": 1.0
            }
            if task == 'eval':
                new_d['doc_key'] = d['doc_key']

            all_data.append(new_d)

        return all_data

    def fit(self, path_train: str, path_dev: str, task_name: str,
            meta_path: str, ontology_dict_path: str, iter: int):
        """
        Train the Detector model.

        Args:
            path_train:         Path to training data
            path_dev:           Path to dev data
            task_name:          Dataset name
            meta_path:          Path to meta.json
            ontology_dict_path: Path to ontology CSV
            iter:               Current iteration (0 = initial training)
        """
        model = self.get_model()
        if Path(model.model_dir).exists():
            return

        if iter == 0:
            data_train = self.trans_detector_data(
                task_name=task_name, meta_path=meta_path,
                data_path=path_train, ontology_dict_path=ontology_dict_path,
                task='train'
            )
            data_dev = self.trans_detector_data(
                task_name=task_name, meta_path=meta_path,
                data_path=path_dev, ontology_dict_path=ontology_dict_path,
                task='train'
            )
        else:
            data_train = Dataset().load(path_train)
            data_train = self._trans_detector_synthetic(data_train)
            data_dev = self.trans_detector_data(
                task_name=task_name, meta_path=meta_path,
                data_path=path_dev, ontology_dict_path=ontology_dict_path,
                task='train'
            )

        path_train = self.write_data(data_train, "train")
        path_dev = self.write_data(data_dev, "dev")
        model.fit(path_train=path_train, path_dev=path_dev)
        delete_checkpoints(model.model_dir)

    @staticmethod
    def _trans_detector_synthetic(data: list) -> list:
        """
        Convert synthetic/filtered data back into Detector training format.

        Args:
            data: List of dicts with Context, Trigger, Event type, reward keys.

        Returns:
            List of dicts in Detector training format.
        """
        all_data = []
        for d in tqdm(data, desc="Preparing synthetic detector data"):
            context = d['Context']
            trigger = d['Trigger']
            event_type = d['Event type']
            reward = d.get('reward', 1.0)

            context_tokens = context.split(' ')
            summary = f"trigger: {trigger} ; event type: {event_type}"

            new_d = {
                "input_context": context_tokens,
                "summary": summary,
                "reward": reward
            }
            all_data.append(new_d)

        return all_data

    def predict(self, path_in: str, path_out: str):
        """
        Run trigger detection on input documents.

        Args:
            path_in:  Path to input data (jsonl with input_context fields)
            path_out: Path to write predictions

        Returns:
            Writes predictions with detected trigger and event_type.
        """
        if Path(path_out).exists():
            return

        data = Dataset.load(path_in)
        model = self.get_model()
        gen = model.load_generator(torch.device("cuda"))
        out_data = []

        for i in tqdm(range(0, len(data), model.batch_size), desc="Detecting"):
            batch = data[i: i + model.batch_size]
            contexts = [' '.join(t['input_context']) if isinstance(t['input_context'], list)
                        else t['input_context'] for t in batch]
            targets = [t.get('summary', '') for t in batch]
            doc_keys = [t.get('doc_key', '') for t in batch]

            # Encode context as both template and context for BART input
            # For detection, we use a fixed prompt as template
            detection_prompt = "detect event trigger and type"
            batch_texts = [
                (detection_prompt, ctx) for ctx in contexts
            ]
            model_inputs = gen.tokenizer.batch_encode_plus(
                batch_texts,
                add_special_tokens=True,
                max_length=512,
                truncation='only_second',
                padding='max_length',
                return_tensors="pt"
            ).to(gen.model.device)

            outputs = gen.model.generate(
                **model_inputs,
                do_sample=False,
                num_return_sequences=1,
                max_length=128,
            )
            decoded = gen.tokenizer.batch_decode(outputs, skip_special_tokens=True)

            for output, target, doc_key in zip(decoded, targets, doc_keys):
                out_data.append({
                    "doc_key": doc_key,
                    "predicted": output,
                    "gold": target
                })

        Path(path_out).parent.mkdir(exist_ok=True, parents=True)
        with open(path_out, "w") as f:
            for s in out_data:
                json.dump(s, f)
                f.write('\n')

    @staticmethod
    def parse_detection(text: str) -> Optional[Dict[str, str]]:
        """
        Parse the Detector's output text into structured trigger + event_type.

        Expected format: "trigger: <trigger_word> ; event type: <type>"

        Returns:
            Dict with 'trigger' and 'event_type' keys, or None if parsing fails.
        """
        try:
            trigger = text.split("trigger:")[1].split(";")[0].strip()
            event_type = text.split("event type:")[1].strip()
            return {"trigger": trigger, "event_type": event_type}
        except (IndexError, AttributeError):
            return None

    def estimate(self, path_in: str, path_out: str):
        """
        Compute detection confidence scores (NLL) for each instance.

        For each input instance, computes the negative log-likelihood of the
        Detector model producing the gold detection output. Lower NLL means
        the Detector is more confident about this detection.

        Args:
            path_in:  Path to input data with Context, Trigger, Event type
            path_out: Path to write data with added 'detector_nll' field
        """
        model = self.get_model()
        bart_model = AutoModelForSeq2SeqLM.from_pretrained(model.model_dir).to('cuda')
        bart_tokenizer = AutoTokenizer.from_pretrained(model.model_dir)

        data_in = Dataset().load(path_in)
        dataset = []

        for ins_i, ins in enumerate(data_in):
            if ins.get('detector_nll') is not None:
                return
            context = ins['Context']
            trigger = ins['Trigger']
            event_type = ins['Event type']

            context_tokens = context.split(' ') if isinstance(context, str) else context
            input_text = ' '.join(context_tokens)
            output_text = f"trigger: {trigger} ; event type: {event_type}"

            dataset.append({
                'ins_i': ins_i,
                'input': input_text,
                'output': output_text
            })

        num_batch = math.ceil(len(dataset) / model.batch_size)
        bsz = model.batch_size
        dataset_counter = 0

        detection_prompt = "detect event trigger and type"

        for i in tqdm(range(num_batch), desc='Detector estimating'):
            batch = dataset[i * bsz: (i + 1) * bsz]
            batch_input = [detection_prompt + ' </s> </s> ' + t['input'] for t in batch]
            batch_output = [t['output'] for t in batch]

            inputs = bart_tokenizer(batch_input, return_tensors='pt', padding=True,
                                    truncation=True, max_length=512)
            input_ids = inputs['input_ids'].to('cuda')
            attention_mask = inputs['attention_mask'].to('cuda')
            labels = bart_tokenizer(batch_output, return_tensors='pt', padding=True)
            labels = labels['input_ids'].to('cuda')
            labels = torch.where(labels == bart_tokenizer.pad_token_id, -100, labels).to('cuda')

            with torch.no_grad():
                outputs = bart_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                ).logits

            loss_fct = CrossEntropyLoss()
            for o, l in zip(outputs, labels):
                loss = loss_fct(o.view(-1, o.shape[-1]), l.view(-1))
                data_in[dataset_counter]['detector_nll'] = loss.item()
                dataset_counter += 1

        assert dataset_counter == len(dataset)

        with open(path_out, "w") as f:
            for d in data_in:
                json.dump(d, f, ensure_ascii=False)
                f.write('\n')
