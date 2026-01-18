import json
import random
import math 
import numpy as np
from collections import Counter
from pathlib import Path
from typing import List
from torch.nn import CrossEntropyLoss
import re 
from collections import defaultdict
import torch
import pandas as pd
from transformers import set_seed as hf_set_seed

import torch.nn.functional as F
# from fire import Fire
from pydantic.main import BaseModel
from tqdm import tqdm


from modeling_rl import (EAE_Extractor,  BaseEAEModel,
                      select_model)
from utils import delete_checkpoints

from Dataset import Dataset

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

class Extractor(BaseModel):
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

    def write_data(self, data: Dataset, name: str) -> str:
       
        model = self.get_model()
        path_out = Path(model.data_dir) / f"{name}.json"
        path_out.parent.mkdir(exist_ok=True, parents=True)
        
        # random.seed(model.random_seed)
        random.shuffle(data)
        for d in data:
            with open(path_out, "a") as f:
                json.dump(d,f,ensure_ascii=False)
                f.write('\n')
        return str(path_out)

    def fit(self, path_train: str, path_dev: str,task_name:str, meta_path:str, ontology_dict_path:str,iter:int):
        model = self.get_model()
        if Path(model.model_dir).exists():
            return
        if iter == 0:
            data_train = Dataset().trans_extractor(task_name = task_name,meta_path=meta_path, ontology_dict_path=ontology_dict_path,data_path=path_train,task='train')
            data_dev = Dataset().trans_extractor(task_name = task_name,meta_path=meta_path, ontology_dict_path=ontology_dict_path, data_path=path_dev,task='train')
        else:
            data_train = Dataset().load(path_train)
            data_train = Dataset().trans_extractor_synthetic(data_train,ontology_dict_path=ontology_dict_path)
            data_dev = Dataset().trans_extractor(task_name = task_name,meta_path=meta_path, ontology_dict_path=ontology_dict_path, data_path=path_dev,task='train')
        path_train = self.write_data(data_train, "train")
        path_dev = self.write_data(data_dev, "dev")

        model.fit(path_train=path_train, path_dev=path_dev)
        delete_checkpoints(model.model_dir)

    def predict(self, path_in: str, path_out: str, use_label_constraint: bool = True):
        if Path(path_out).exists():
            return
        
        data = Dataset.load(path_in)
  
        model = self.get_model()
     
        gen = model.load_generator(torch.device("cuda"))
        encoder = model.get_encoder()
        out_data = []

        for i in tqdm(range(0, len(data), model.batch_size)):
            batch = data[i : i + model.batch_size]
            doc_keys = [t['doc_key'] for t in batch]
            templete = [t['input_templete'] for t in batch]
            context = [t['input_context'] for t in batch]
            targets =  [t['summary'] for t in batch]
            outputs = model.gen_texts(
                [templete,context], gen, num_beams=1, save_scores=use_label_constraint
            )
            assert len(outputs) == len(context)== len(templete)

            for output, target,doc_key in zip(outputs,targets,doc_keys):
                out_data.append( {"doc_key":doc_key,"predicted":output,"gold":target})
                # if use_label_constraint:
                #     assert gen.scores is not None
                #     triplet = constraint.run(triplet, gen.scores[i])
                # sents.append(Sentence(triplets=[triplet]))
        Path(path_out).parent.mkdir(exist_ok=True, parents=True)
        with open(path_out, "w") as f:
            for s in out_data:
                json.dump(s,f)
                f.write('\n')
    @staticmethod
    def score(path_pred: str, path_gold: str) -> dict:
        pred = Dataset().load(path_pred)
        gold = Dataset().load(path_gold)
        assert len(pred.sents) == len(gold.sents)
        num_pred = 0
        num_gold = 0
        num_correct = 0

        for i in range(len(gold.sents)):
            num_pred += len(pred.sents[i].triplets)
            num_gold += len(gold.sents[i].triplets)
            for p in pred.sents[i].triplets:
                for g in gold.sents[i].triplets:
                    if len(p.head) == 0 or len(p.tail) == 0:
                        num_pred -= 1 
                        continue
                    if (p.head, p.tail, p.label) == (g.head, g.tail, g.label):
                        num_correct += 1

        precision = safe_divide(num_correct, num_pred)
        recall = safe_divide(num_correct, num_gold)

        info = dict(
            path_pred=path_pred,
            path_gold=path_gold,
            precision=precision,
            recall=recall,
            score=safe_divide(2 * precision * recall, precision + recall),
        )
        return info
    def estimate(self, path_in, path_out,ontology_dict_path):

        def load_ontology(ontology_path):
            ontology_dict ={} 
            df =pd.read_csv(ontology_path, quotechar='"')
            for _, row in df.iterrows():
                evt_type = row['event_type']
                template = row['template']
                ontology_dict[evt_type] = {
                        'template': template
                    }
                for i in range(len(row) - 2):  # 头两列是 event_type, template
                    col_name = f'<arg{i+1}>'
                    if col_name in row:
                        arg = row[col_name]
                        if pd.notna(arg):  # ✅ 正确判断是否为 NaN
                            if isinstance(arg, str) and 'evt' in arg:
                                role = arg[11:].lower()
                                ontology_dict[evt_type][f'arg{i+1}'] = role
                                ontology_dict[evt_type][role] = f'arg{i+1}'
                            else:
                                role = arg.lower()
                                ontology_dict[evt_type][f'arg{i+1}'] = role
                                ontology_dict[evt_type][role] = f'arg{i+1}'

            return ontology_dict 

            
        ontology_dict = load_ontology(ontology_dict_path)
        for evt,roles in ontology_dict.items():
            new_d = {}
            for i,(key,value) in enumerate(roles.items()):
                if i == 0:
                    new_d['template'] = value
                    continue
                if i%2==0:
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

            template = ontology_dict[event_type.replace('n/a','unspecified')]['template']
            input_template = re.sub(r'<arg\d>', '<arg>', template) 
            trigger_s = None
            trigger_e = None
            context_tokens  = re.findall(r"\w+|[^\w\s]", context)
            trigger_token = re.findall(r"\w+|[^\w\s]", trigger)

            n, m = len(context_tokens), len(trigger_token)
            for i in range(n - m + 1):
                if context_tokens[i:i + m] == trigger_token:
                    trigger_s = i
                    trigger_e = i + m - 1  
                    break 

            prefix = context_tokens[: trigger_s]
            tgt = context_tokens[trigger_s:trigger_e+1]
            
            suffix = context_tokens[trigger_e+1:]
            input_context = prefix + ['<tgr>', ] + tgt + ['<tgr>', ] + suffix 
            output_template = template

     

            for role,span in arguments.items():
                role = role.strip()
                if not span:
                    continue
                arg_num = ontology_dict[event_type.replace('n/a','unspecified')][role]
                arg_text = span
                output_template = re.sub('<{}>'.format(arg_num),arg_text , output_template)
                
            output_template = re.sub(r'<arg\d>','<arg>', output_template ) 
            dataset.append({'ins_i': ins_i, "template":input_template, 'context': input_context, 'output':output_template,'Event type':event_type})
            
        num_batch = math.ceil(len(dataset) / model.batch_size)
        bsz = model.batch_size
        dataset_counter = 0 


        for i in tqdm(range(num_batch), desc='extractor estimating'):
            batch = dataset[i * bsz: (i + 1) * bsz]
            batch_input_context = [' '.join(t['context']) for t in batch]
            batch_input_template= [t['template'] for t in batch]
            batch_output = [t['output'] for t in batch]
            batch_input_combined = [
                    template + ' </s> </s> ' + context
                    for template, context in zip(batch_input_template, batch_input_context)
                ]
            
            inputs = bart_tokenizer(batch_input_combined, return_tensors='pt', padding=True)
            input_ids, attention_mask = inputs['input_ids'].to('cuda'), inputs['attention_mask'].to('cuda')
            labels = bart_tokenizer(batch_output, return_tensors='pt', padding=True)
            labels = labels['input_ids'].to('cuda')
            labels = torch.where(labels == bart_tokenizer.pad_token_id, -100, labels).to('cuda')
            outputs = bart_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).logits
            # no shift for bart model 

            loss_fct = CrossEntropyLoss()
            for o, l in zip(outputs, labels):
                loss = loss_fct(o.view(-1, o.shape[-1]), l.view(-1), )
                data_in[dataset_counter]['extractor_nll'] = loss.item()
                data_in[dataset_counter]['cond_generator_nll'] = 0.0
                data_in[dataset_counter]['generator_nll'] = 0.0
                dataset_counter += 1
        assert dataset_counter == len(dataset) 

        
  
        with open(path_out, "w") as f:
            for d in data_in:
                json.dump(d,f,ensure_ascii=False)
                f.write('\n')
        # for data in dataset:
        #     ins_i, tri_i, e_nll = data['ins_i'], data['tri_i'], data['extractor_nll']
        #     assert ins_i < len(data_in.sents) and tri_i < len(data_in.sents[ins_i].triplets)
        #     tri = data_in.sents[ins_i].triplets[tri_i]
        #     tri.extractor_nll = e_nll
        # data_in.save(path_out)

