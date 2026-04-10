"""
Extractor Agent: Event Argument Extraction via Template Filling.

The Extractor agent extracts event arguments from documents given detected
triggers and event types. It uses a BART-based seq2seq model with a
template-filling approach: the model takes a template with <arg> placeholders
and a context with <tgr> trigger markers as input, and generates the template
with placeholders filled by actual argument spans.

Input:  (template_with_<arg>_slots, context_with_<tgr>_markers)
Output: template_with_filled_arguments
"""

import json
import random
import math
import re
import numpy as np
from collections import Counter, defaultdict
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


class Extractor(BaseModel):
    """
    Extractor Agent for event argument extraction.

    Uses a BART-based encoder-decoder model with template-filling approach.
    The input template contains <arg> placeholders for each argument role,
    and the model learns to fill them with text spans from the document.

    Workflow:
        1. fit():      Train on labeled data for template-filling
        2. predict():  Extract arguments from documents
        3. estimate(): Compute extraction confidence (NLL) for quality scoring
    """
    load_dir: str
    save_dir: str
    model_name: str = "extractor"
    encoder_name: str = "extract"
    search_threshold: float = -0.9906
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

    def fit(self, path_train: str, path_dev: str, task_name: str,
            meta_path: str, ontology_dict_path: str, iter: int):
        """
        Train the Extractor model.

        iter=0: Train on real annotated data using template-filling format.
        iter>0: Train on filtered synthetic data from the pipeline.
        """
        model = self.get_model()
        if Path(model.model_dir).exists():
            return

        if iter == 0:
            data_train = Dataset().trans_extractor(
                task_name=task_name, meta_path=meta_path,
                ontology_dict_path=ontology_dict_path,
                data_path=path_train, task='train'
            )
            data_dev = Dataset().trans_extractor(
                task_name=task_name, meta_path=meta_path,
                ontology_dict_path=ontology_dict_path,
                data_path=path_dev, task='train'
            )
        else:
            data_train = Dataset().load(path_train)
            data_train = Dataset().trans_extractor_synthetic(
                data_train, ontology_dict_path=ontology_dict_path
            )
            data_dev = Dataset().trans_extractor(
                task_name=task_name, meta_path=meta_path,
                ontology_dict_path=ontology_dict_path,
                data_path=path_dev, task='train'
            )

        path_train = self.write_data(data_train, "train")
        path_dev = self.write_data(data_dev, "dev")
        model.fit(path_train=path_train, path_dev=path_dev)
        delete_checkpoints(model.model_dir)

    def predict(self, path_in: str, path_out: str, use_label_constraint: bool = True):
        """
        Run argument extraction on input data.

        Reads data with (input_templete, input_context, summary) fields,
        generates template-filled output using beam search.
        """
        if Path(path_out).exists():
            return

        data = Dataset.load(path_in)
        model = self.get_model()
        gen = model.load_generator(torch.device("cuda"))
        out_data = []

        for i in tqdm(range(0, len(data), model.batch_size), desc="Extracting"):
            batch = data[i: i + model.batch_size]
            doc_keys = [t['doc_key'] for t in batch]
            templete = [t['input_templete'] for t in batch]
            context = [t['input_context'] for t in batch]
            targets = [t['summary'] for t in batch]

            outputs = model.gen_texts(
                [templete, context], gen, num_beams=1,
                save_scores=use_label_constraint
            )
            assert len(outputs) == len(context) == len(templete)

            for output, target, doc_key in zip(outputs, targets, doc_keys):
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

    def estimate(self, path_in: str, path_out: str, ontology_dict_path: str):
        """
        Compute extraction confidence scores (NLL) for synthetic data.

        For each synthetic instance, computes the cross-entropy loss of the
        Extractor model producing the expected filled template. Lower NLL
        means higher confidence in the extraction quality.

        This score is used in the reward computation for data filtering.
        """

        def load_ontology(ontology_path):
            ontology_dict = {}
            df = pd.read_csv(ontology_path, quotechar='"')
            for _, row in df.iterrows():
                evt_type = row['event_type']
                template = row['template']
                ontology_dict[evt_type] = {'template': template}
                for i in range(len(row) - 2):
                    col_name = f'<arg{i + 1}>'
                    if col_name in row:
                        arg = row[col_name]
                        if pd.notna(arg):
                            if isinstance(arg, str) and 'evt' in arg:
                                role = arg[11:].lower()
                            else:
                                role = arg.lower()
                            ontology_dict[evt_type][f'arg{i + 1}'] = role
                            ontology_dict[evt_type][role] = f'arg{i + 1}'
            return ontology_dict

        ontology_dict = load_ontology(ontology_dict_path)
        for evt, roles in ontology_dict.items():
            new_d = {}
            for i, (key, value) in enumerate(roles.items()):
                if i == 0:
                    new_d['template'] = value
                    continue
                if i % 2 == 0:
                    key = key
                else:
                    value = value
                new_d[key] = value
            ontology_dict[evt] = new_d

        model = self.get_model()
        bart_model = AutoModelForSeq2SeqLM.from_pretrained(model.model_dir).to('cuda')
        bart_tokenizer = AutoTokenizer.from_pretrained(model.model_dir)

        data_in = Dataset().load(path_in)
        dataset = []

        for ins_i, ins in enumerate(data_in):
            if ins.get('extractor_nll'):
                return
            context = ins['Context']
            trigger = ins['Trigger']
            event_type = ins['Event type']
            arguments = ins['arguments']

            template = ontology_dict[event_type.replace('n/a', 'unspecified')]['template']
            input_template = re.sub(r'<arg\d>', '<arg>', template)

            # Find trigger position and add <tgr> markers
            context_tokens = re.findall(r"\w+|[^\w\s]", context)
            trigger_token = re.findall(r"\w+|[^\w\s]", trigger)

            trigger_s = None
            n, m = len(context_tokens), len(trigger_token)
            for idx in range(n - m + 1):
                if context_tokens[idx:idx + m] == trigger_token:
                    trigger_s = idx
                    trigger_e = idx + m - 1
                    break

            if trigger_s is None:
                continue

            prefix = context_tokens[:trigger_s]
            tgt = context_tokens[trigger_s:trigger_e + 1]
            suffix = context_tokens[trigger_e + 1:]
            input_context = prefix + ['<tgr>'] + tgt + ['<tgr>'] + suffix

            # Build output template with filled arguments
            output_template = template
            for role, span in arguments.items():
                role = role.strip()
                if not span:
                    continue
                arg_num = ontology_dict[event_type.replace('n/a', 'unspecified')].get(role)
                if arg_num is None:
                    continue
                output_template = re.sub(f'<{arg_num}>', span, output_template)
            output_template = re.sub(r'<arg\d>', '<arg>', output_template)

            dataset.append({
                'ins_i': ins_i,
                'template': input_template,
                'context': input_context,
                'output': output_template,
                'Event type': event_type
            })

        num_batch = math.ceil(len(dataset) / model.batch_size)
        bsz = model.batch_size
        dataset_counter = 0

        for i in tqdm(range(num_batch), desc='Extractor estimating'):
            batch = dataset[i * bsz: (i + 1) * bsz]
            batch_input_context = [' '.join(t['context']) for t in batch]
            batch_input_template = [t['template'] for t in batch]
            batch_output = [t['output'] for t in batch]
            batch_input_combined = [
                template + ' </s> </s> ' + context
                for template, context in zip(batch_input_template, batch_input_context)
            ]

            inputs = bart_tokenizer(batch_input_combined, return_tensors='pt', padding=True)
            input_ids = inputs['input_ids'].to('cuda')
            attention_mask = inputs['attention_mask'].to('cuda')
            labels = bart_tokenizer(batch_output, return_tensors='pt', padding=True)
            labels = labels['input_ids'].to('cuda')
            labels = torch.where(
                labels == bart_tokenizer.pad_token_id, -100, labels
            ).to('cuda')

            with torch.no_grad():
                outputs = bart_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                ).logits

            loss_fct = CrossEntropyLoss()
            for o, l in zip(outputs, labels):
                loss = loss_fct(o.view(-1, o.shape[-1]), l.view(-1))
                data_in[dataset_counter]['extractor_nll'] = loss.item()
                data_in[dataset_counter]['cond_generator_nll'] = 0.0
                data_in[dataset_counter]['generator_nll'] = 0.0
                dataset_counter += 1

        assert dataset_counter == len(dataset)

        with open(path_out, "w") as f:
            for d in data_in:
                json.dump(d, f, ensure_ascii=False)
                f.write('\n')

    @staticmethod
    def parse_extraction(predicted: str, template: str,
                         ontology_dict: dict, event_type: str) -> Dict[str, Optional[str]]:
        """
        Parse the Extractor's output into structured role -> span mappings.

        Compares the predicted filled template against the original template
        to extract which argument spans were placed into which <arg> slots.

        Args:
            predicted:     The filled template from Extractor
            template:      The original template with <argN> placeholders
            ontology_dict: Event ontology mapping
            event_type:    The event type

        Returns:
            Dict mapping role names to extracted spans (or None).
        """
        arguments = {}
        evt_key = event_type.replace('n/a', 'unspecified')
        if evt_key not in ontology_dict:
            return arguments

        evt_ont = ontology_dict[evt_key]

        # Find <argN> positions in original template
        arg_pattern = re.compile(r'<arg(\d+)>')
        for match in arg_pattern.finditer(template):
            arg_num = f'arg{match.group(1)}'
            if arg_num in evt_ont:
                role = evt_ont[arg_num]
                arguments[role] = None

        # Compare predicted with template to extract filled values
        pred_clean = predicted.strip()
        template_clean = re.sub(r'<arg\d>', '<arg>', template)

        # Split by <arg> placeholder to identify filled spans
        template_parts = template_clean.split('<arg>')
        if len(template_parts) < 2:
            return arguments

        remaining = pred_clean
        filled_values = []
        for j, part in enumerate(template_parts[:-1]):
            part = part.strip()
            if part and part in remaining:
                idx = remaining.index(part)
                remaining = remaining[idx + len(part):]
            next_part = template_parts[j + 1].strip()
            if next_part and next_part in remaining:
                idx = remaining.index(next_part)
                value = remaining[:idx].strip()
                filled_values.append(value if value and value != '<arg>' else None)
                remaining = remaining[idx:]
            else:
                filled_values.append(remaining.strip() if remaining.strip() else None)
                remaining = ''

        # Map filled values back to roles
        role_names = list(arguments.keys())
        for idx, role in enumerate(role_names):
            if idx < len(filled_values):
                val = filled_values[idx]
                if val and val.lower() != 'none' and val != '<arg>':
                    arguments[role] = val

        return arguments
