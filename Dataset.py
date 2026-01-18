import json

import numpy as np
from collections import Counter
from pathlib import Path
from typing import List

import re 

from transformers import set_seed as hf_set_seed

import torch.nn.functional as F
# from fire import Fire
from pydantic.main import BaseModel
from tqdm import tqdm


import pandas as pd
class Dataset(BaseModel):
    

    @classmethod
    def load(cls, path: str):
        with open(path) as f:
            sents = [json.loads(line) for line in f]
        return sents
    def load_ontology(self,ontology_path):
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

    
    def get_meta(self,task_name,meta_path):
            meta = json.load(open(meta_path,'r',encoding='utf8'))
          
            type2roles = {}
            for rams_m in meta:
                rams_event_type, role_types = rams_m
                role_types = [
                        rams_role[11:].lower() if 'evt' in rams_role else rams_role.lower()
                        for rams_role in role_types
                    ]
                type2roles[rams_event_type] = role_types
            return type2roles
        
 

    def trans_extractor_synthetic(self,data,ontology_dict_path):
        def find_sublist_indices(A, B):
            len_A = len(A)
            for i in range(len(B) - len_A + 1):
                if B[i:i + len_A] == A:
                    return i, i + len_A - 1
            return -1, -1   
        
     
        all_data = []
        ontology_dict = self.load_ontology(ontology_dict_path)
        for d in tqdm(data):
            event_type = d['Event type']
            Trigger = d['Trigger']
            Context = d['Context']
            arguments = d['arguments']
            reward =d['reward']
            
            template = ontology_dict[event_type.replace('n/a','unspecified')]['template']
            input_template = re.sub(r'<arg\d>', '<arg>', template) 
 
            Context_token = Context.split(' ')
            Trigger_token = Trigger.split(' ')

            t_s,t_e = find_sublist_indices(Trigger_token,Context_token)

      
            prefix = Context_token[: t_s]
            tgt = Context_token[t_s:t_e+1 ]
            suffix = Context_token[t_e+1:]
           
            input_context = prefix + ['<tgr>', ] + tgt + ['<tgr>', ] + suffix 
            output_template = template
            for role, args in arguments.items():
                if "evt" in role:
                    role = role[11:]
                role = role.lower()
                if not ontology_dict[event_type.replace('n/a','unspecified')].get(role):continue
                arg_num = ontology_dict[event_type.replace('n/a','unspecified')][role]
                arg_text = args
                if not arg_text:
                    continue
                output_template = re.sub('<{}>'.format(arg_num),arg_text , output_template)
            output_template = re.sub(r'<arg\d>','<arg>', output_template ) 

            new_d = {'input_templete':input_template,"input_context":input_context,"summary":output_template,"reward":reward}
            all_data.append(new_d)

        return all_data
   

    def trans_extractor(self,task_name, meta_path, data_path,ontology_dict_path,task='train'):
        def read_data(data_path):
            data = []
            with open(data_path) as f:
                    for line in f.readlines():
                        data.append(json.loads(line))
            return data
        type2roles = self.get_meta(task_name,meta_path)
        data = read_data(data_path)
        all_data = []
        ontology_dict = self.load_ontology(ontology_dict_path)
        for d in tqdm(data):
            doc_key = d['doc_key']
            sentences = d['sentences']
            all_tokens = []
            for sentence in sentences:
                all_tokens.extend(sentence)
            doc_text = ' '.join (all_tokens)
            event_type = d['evt_triggers'][0][-1][0][0]

            template = ontology_dict[event_type.replace('n/a','unspecified')]['template']
            input_template = re.sub(r'<arg\d>', '<arg>', template) 

            trigger_s,trigger_e = d['evt_triggers'][0][0],d['evt_triggers'][0][1]
            trigger = all_tokens[trigger_s:trigger_e+1]
            prefix = all_tokens[: trigger_s]
            tgt = all_tokens[ trigger_s:trigger_e+1]
            suffix = all_tokens[trigger_e+1:]

            input_context = prefix + ['<tgr>', ] + tgt + ['<tgr>', ] + suffix 
           
            gold_evt_links = d['gold_evt_links'] 
           
            all_roles = type2roles[event_type]
            output_template = template
            for triple in gold_evt_links:
                trigger_span, argument_span, arg_name = triple
                if "evt" in arg_name:
                    arg_name=arg_name[11:]
                arg_name = arg_name.lower()
                if not ontology_dict[event_type.replace('n/a','unspecified')].get(arg_name):continue
                arg_num = ontology_dict[event_type.replace('n/a','unspecified')][arg_name]
                
                arg_text = ' '.join(all_tokens[argument_span[0]:argument_span[1]+1])
                output_template = re.sub('<{}>'.format(arg_num),arg_text , output_template)
            output_template = re.sub(r'<arg\d>','<arg>', output_template ) 
            if task=='train':
                new_d = {'input_templete':input_template,"input_context":input_context,"summary":output_template,"reward":1.0}
            else:
                new_d = {'doc_key':doc_key,'input_templete':input_template,"input_context":input_context,"summary":output_template,"reward":1.0}
            all_data.append(new_d)

        return all_data
    
    def trans_generator_synthetic(self,data):
        all_data = []
        for d in tqdm(data):
            Context = d['Context']
            Trigger = d['Trigger']
            event_type = d['Event type']
            arguments = d['arguments']
            reward =d['reward']
            output_Context = f"Context: {Context}. "  
            output_trigger = f"The trigger of the event in the context is : {Trigger}. "
            all_roles = list(arguments.keys())
           
            all_roles = [
                rams_role[11:].lower() if 'evt' in rams_role else rams_role.lower()
                for rams_role in all_roles
            ]
            output_role = 'Role-Argument Pairs: '
            role_items = [f"{role.lower()}: {span}" for role, span in arguments.items()]
            output_role = "Role-Argument Pairs: " + ', '.join(role_items) + '.'
            inputs = f"Generate an event. Given the event type: '{event_type}' and the following roles: {', '.join(all_roles)}, please generate a coherent context that includes the event trigger and the role-argument pairs. Make sure the event trigger and each argument for the corresponding role appear clearly in the context, and the context reflects the relationship between the trigger and its arguments."
            outputs = output_Context+output_trigger+output_role.strip()
        
            new_d = {'input':inputs,"label":outputs,"reward":reward}
            all_data.append(new_d)
        return all_data
    
    def train_data_none_radio_compute(self,task_name, meta_path, data_path):
        def read_data(data_path):
            data = []
            with open(data_path) as f:
                    for line in f.readlines():
                        data.append(json.loads(line))
            return data
        def get_meta(task_name,meta_path):
            meta = json.load(open(meta_path,'r',encoding='utf8'))
            
            type2roles = {}
            for rams_m in meta:
                rams_event_type, role_types = rams_m
                role_types = [
                        rams_role[11:].lower() if 'evt' in rams_role else rams_role.lower()
                        for rams_role in role_types
                    ]
     
                type2roles[rams_event_type] = role_types
            return type2roles
        type2roles = get_meta(task_name,meta_path)
        data = read_data(data_path)
    
        none_radio_list = []
        for d in tqdm(data):
            none_num = 0
            all_num = 0
            sentences = d['sentences']
            all_tokens = []
            for sentence in sentences:
                all_tokens.extend(sentence)
        
            event_type = d['evt_triggers'][0][-1][0][0]
          
            gold_evt_links = d['gold_evt_links'] 
            
            all_roles = type2roles[event_type]
            
            role2span = {}
            all_num = len(all_roles)
            for role in all_roles:
                role2span[role] = None
            
            for gold_evt_link in gold_evt_links:
                current_role = gold_evt_link[-1]
                if "evt" in current_role:
                    current_role = current_role[11:]
                current_role=current_role.lower()
                s,e = gold_evt_link[1]
                span = ' '.join(all_tokens[s:e+1])
                role2span[current_role] = span
            
            for role,span in role2span.items():
                if not span:
                    none_num+=1
            none_radio_list.append(none_num/all_num)
        return np.mean(none_radio_list),np.std(none_radio_list) 



    def trans_generator(self,task_name, meta_path, data_path):
        def read_data(data_path):
            data = []
            with open(data_path) as f:
                    for line in f.readlines():
                        data.append(json.loads(line))
            return data
        
        type2roles = self.get_meta(task_name,meta_path)
        data = read_data(data_path)
        all_data = []
        for d in tqdm(data):
            sentences = d['sentences']
            all_tokens = []
            for sentence in sentences:
                all_tokens.extend(sentence)
            doc_text = ' '.join (all_tokens)
            event_type = d['evt_triggers'][0][-1][0][0]
            trigger_s,trigger_e = d['evt_triggers'][0][0],d['evt_triggers'][0][1]
            trigger = ' '.join(all_tokens[trigger_s:trigger_e+1])
            
            gold_evt_links = d['gold_evt_links'] 
            
            all_roles = type2roles[event_type]
            output_Context = f"Context: {doc_text}. "
            output_trigger = f"The trigger of the event in the context is : {trigger}. "
            context_role = ""
            role2span = {}
            
            for role in all_roles:
                role2span[role] = None
            
            for gold_evt_link in gold_evt_links:
                current_role = gold_evt_link[-1]

                if "evt" in current_role:
                    current_role = current_role[11:]
                current_role=current_role.lower()
                s,e = gold_evt_link[1]
                span = ' '.join(all_tokens[s:e+1])
                role2span[current_role] = span
            output_role = 'Role-Argument Pairs: '
            role_items = [f"{role}: {span}" for role, span in role2span.items()]
            output_role = "Role-Argument Pairs: " + ', '.join(role_items) + '.'

            inputs = f"Generate an event. Given the event type: '{event_type}' and the following roles: {', '.join(all_roles)}, please generate a coherent context that includes the event trigger and the role-argument pairs. Make sure the event trigger and each argument for the corresponding role appear clearly in the context, and the context reflects the relationship between the trigger and its arguments."
            outputs = output_Context+output_trigger+output_role.strip()
            new_d = {'input':inputs,"label":outputs,"reward":1.0}
            all_data.append(new_d)
        return all_data
    
    def save(self, path: str):
        Path(path).parent.mkdir(exist_ok=True, parents=True)
        with open(path, "w") as f:
            for s in self.sents:
                f.write(s.json() + "\n")

 