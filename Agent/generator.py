
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
from peft import get_peft_model, LoraConfig, TaskType,PeftModel, PeftConfig
import torch.nn.functional as F

from pydantic.main import BaseModel
from tqdm import tqdm

from typing import Optional
from modeling_rl import (EAE_Extractor,  BaseEAEModel,
                      select_model)
from utils import delete_checkpoints

from Dataset import Dataset
from transformer_base import run_clm_rl
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

class Generator(BaseModel):
    load_dir: str
    save_dir: str
    num_gen_per_label: int = 250
    model_name: str = "generate"
    encoder_name: str = "generate"
    lora_dir: Optional[str]  = None
    block_size : int = 512
    model_kwargs: dict = {}

    def get_model(self) -> BaseEAEModel:
        model = select_model(
            name=self.model_name,
            encoder_name=self.encoder_name,
            model_dir=str(Path(self.save_dir) / "model"),
            model_name=self.load_dir,
            lora_dir =self.lora_dir,
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

    def fit(self, path_train: str, path_dev: str,task_name:str, meta_path:str,iter:int):
        model = self.get_model()
        if Path(model.model_dir).exists():
            return
        
        if iter == 0 :
            data_train = Dataset().trans_generator(task_name = task_name,meta_path=meta_path, data_path=path_train)
            data_dev = Dataset().trans_generator(task_name = task_name,meta_path=meta_path, data_path=path_dev)
        else:
            data_train = Dataset().load(path_train)
            data_train = Dataset().trans_generator_synthetic(data_train)
            data_dev = Dataset().trans_generator(task_name = task_name,meta_path=meta_path, data_path=path_dev)

        path_train = self.write_data(data_train, "train")
        path_dev = self.write_data(data_dev, "dev")
        model.fit(path_train=path_train, path_dev=path_dev)
        delete_checkpoints(model.model_dir)

    def decode_result(self, results: list,  event_type: str):
        decode_results = []
        for text in results:
            try:
                role_list = text.split('following roles: ')[1].split(", please generate")[0].split(',')
                role_list = [role.strip().lower() for role in role_list] 
               
                Context = text.split("Context: ")[1].split("The trigger of the event in the context is : ")[0]

                Trigger = text.split("The trigger of the event in the context is :")[1].split(". Role-Argument Pairs")[0].strip()
                Role_Argument = text.split(". Role-Argument Pairs: ")[1]
            except (IndexError, AttributeError):
                continue

            # role2arg = {role:None for role in role_list}
            role2arg = {}
            result = {}

            for chunk in Role_Argument.split(','):
                if ':' not in chunk:
                    continue
                role, val = chunk.strip().split(':', 1)
                role = role.strip().lower()
                val = val.strip().strip('"').rstrip('.') 
                if role in role_list:
                    if re.fullmatch(r'[A-Z](?:\.[A-Z])+\.?', val): 
                        val = val.rstrip('.') + '.'  
                    if val.lower() == "none":
                        val = None
                    role2arg[role] = val
            if len(role2arg)==0 :
                continue
            result['Context']= Context
            result['Trigger']= Trigger
            result['Event type'] = event_type
            result["arguments"] = role2arg
            decode_results.append(result)
        return decode_results

    def generate(self, labels: List[str], path_out: str):
        if Path(path_out).exists():
            return

        generator_model = self.get_model()
       
        if 'llama' in generator_model.model_name.lower():
            base_model = run_clm_rl.Model_Llama.from_pretrained(generator_model.model_name, device_map="auto", torch_dtype=torch.float16)
        elif 'qwen' in generator_model.model_name.lower():
            base_model = run_clm_rl.Model_Qwen.from_pretrained(generator_model.model_name, device_map="auto", torch_dtype=torch.float16)
        model = PeftModel.from_pretrained(base_model, generator_model.model_dir)  
        tokenizer = AutoTokenizer.from_pretrained(generator_model.model_name, use_fast=False)
        tokenizer.pad_token = tokenizer.eos_token  
    

        groups = {}
   
        all_data = []
        for label in tqdm(labels, desc='Generating'):
            current_events = []
            current_num = len(current_events)
            
            event_type = list(label.keys())[0]
            inputs = tokenizer(list(label.values())[0], return_tensors="pt").to(model.device)
            total = 0
            while current_num<self.num_gen_per_label:
            
                outputs = model.generate(
                            **inputs,
                            max_length=512,
                            do_sample=True,          
                            top_k=50,               
                            top_p=0.95,              
                            temperature=0.7,         
                            num_return_sequences=100,  
                            pad_token_id=tokenizer.pad_token_id,
                        )
                  
                results  = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                total +=len(results)
                results = self.decode_result(results,  event_type)
                results = self.data_trans(results)
                current_events.extend(results)
                current_num = len(current_events)
                print(f"success_num:{current_num},total:{total}")
                
            all_data.extend(current_events)
           

        Path(path_out).parent.mkdir(exist_ok=True, parents=True)
        with open(path_out, "w") as f:
            for s in all_data:
                json.dump(s,f)
                f.write('\n')


    def data_trans(self,results):
        flag_trigger_exist = False
        flag_arg = False
        fin_data = []
        for data in results:
            flag_arg = False 
            context = data['Context']
            trigger = data['Trigger']
            event_type = data['Event type']
            arguments = data['arguments']

            context_tokens  = re.findall(r"\w+|[^\w\s]", context)
            for role, argument in arguments.items():
                flag_arg = False 
                if not argument:
                    continue
                argument_tokens  = re.findall(r"\w+|[^\w\s]", argument)
                n, m = len(context_tokens), len(argument_tokens)
                for i in range(n - m + 1):
                    if context_tokens[i:i + m] == argument_tokens:    
                        flag_arg=True 
                        break 
                if not flag_arg:
                    arguments[role] = None
            
            data['arguments'] = arguments

            
            trigger_token = re.findall(r"\w+|[^\w\s]", trigger)
            trigger_s = None
            trigger_e = None
            n, m = len(context_tokens), len(trigger_token)
            for i in range(n - m + 1):
                if context_tokens[i:i + m] == trigger_token:
                    trigger_s = i
                    trigger_e = i + m - 1 
                    flag_trigger_exist=True 
                    break 

            # prefix = context_tokens[: trigger_s]
            # tgt = context_tokens[trigger_s:trigger_e+1]
            # suffix = context_tokens[trigger_e+1:]

            if flag_trigger_exist:
                fin_data.append(data)
            flag_trigger_exist = False
        return fin_data
 
    def estimate(self, path_in, path_out,path_train,task_name,meta_path):
        # if Path(path_out).exists():
        #     return
        # def none_penalty(none_ratio, target_range=(0.1, 0.3), tolerance=0.05):
        #     lower, upper = target_range
        #     if lower - tolerance <= none_ratio <= upper + tolerance:
        #         return 0 
        #     else:
        #         return abs(none_ratio - (lower + upper) / 2)
        
        def none_penalty(none_ratio,mean, std):
           
            if mean - std <= none_ratio <= mean + std:
                return 0  
            else:
                return abs(none_ratio - mean)
            
        
        mean, std  = Dataset().train_data_none_radio_compute(task_name, meta_path, path_train)

        data_in = Dataset.load(path_in)
        dataset = []
        
        for ins in data_in:
            if ins. get('penalty'):
                return
            num_none = 0
            num_roles = 0
            arguments = ins['arguments']
            for role,arg in arguments.items():
                if not arg:
                    num_none +=1
                num_roles+=1
            none_ratio =   num_none/num_roles
            penalty = none_penalty(none_ratio, mean, std)
            # penalty = none_penalty(none_ratio)
            ins['penalty'] = penalty
            dataset.append(ins)
        with open(path_out, "w") as f:
            for d in dataset:
                json.dump(d,f,ensure_ascii=False)
                f.write('\n')

