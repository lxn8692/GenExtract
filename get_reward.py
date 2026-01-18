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
# from fire import Fire
from pydantic.main import BaseModel
from tqdm import tqdm
from typing import Optional

import pandas as pd


from modeling_rl import (NewRelationExtractor, RelationGenerator, RelationModel,
                      select_model)
from utils import delete_checkpoints
from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer,AutoConfig
from peft import PeftModel

import math
from collections import Counter
import matplotlib.pyplot as plt
def safe_divide(a: float, b: float) -> float:
    if a == 0 or b == 0:
        return 0
    return a / b


class Sentence(BaseModel):
   

    @property
    def tokens(self) -> List[str]:
        return self.triplets[0].tokens

    @property
    def text(self) -> str:
        return " ".join(self.tokens)

    def assert_valid(self):
        assert len(self.tokens) > 0
        for t in self.triplets:
            assert t.text == self.text
            assert len(t.head) > 0
            assert len(t.tail) > 0
            assert len(t.label) > 0


class Dataset(BaseModel):
    sents: List[Sentence]

    def get_labels(self) -> List[str]:
        return sorted(set(t.label for s in self.sents for t in s.triplets))

    @classmethod
    def load(cls, path: str):
        with open(path) as f:
            sents = [json.loads(line) for line in f]
        return sents
    
    def trans_extractor_iter(data,ontology_dict_path):
        def find_sublist_indices(A, B):
            len_A = len(A)
            for i in range(len(B) - len_A + 1):
                if B[i:i + len_A] == A:
                    return i, i + len_A - 1
            return -1, -1
        
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
     
        all_data = []
        ontology_dict = load_ontology(ontology_dict_path)
        for d in tqdm(data):
            event_type = d['Event type']
            Trigger = d['Trigger']
            Context = d['Context']
            arguments = d['arguments']
            reward =d['reward']
            
            template = ontology_dict[event_type.replace('n/a','unspecified')]['template']
            input_template = re.sub(r'<arg\d>', '<arg>', template) 
            # input_Context = f"{doc_text}."
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
           
            # outputs = output_Context+output_trigger+output_role
            # out_lenth.append(len(gpt_tokenizer(outputs)['input_ids']))
            new_d = {'input_templete':input_template,"input_context":input_context,"summary":output_template,"reward":reward}
            all_data.append(new_d)

        return all_data
   

    def trans_extractor(task_name, meta_path, data_path,ontology_dict_path,task='train'):
        def read_data(data_path):
            data = []
            with open(data_path) as f:
                    for line in f.readlines():
                        data.append(json.loads(line))
            return data

        
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
        
        def get_meta(task_name,meta_path):
            meta = json.load(open(meta_path,'r',encoding='utf8'))
            # role2data = json.load(open(f"type2data_{task_name}.json",'r',encoding='utf8'))
            
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
        all_data = []
        ontology_dict = load_ontology(ontology_dict_path)
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
            # input_Context = f"{doc_text}."

            
            trigger_s,trigger_e = d['evt_triggers'][0][0],d['evt_triggers'][0][1]
            trigger = all_tokens[trigger_s:trigger_e+1]
        
            prefix = all_tokens[: trigger_s]
            tgt = all_tokens[ trigger_s:trigger_e+1]
            
            suffix = all_tokens[trigger_e+1:]
            input_context = prefix + ['<tgr>', ] + tgt + ['<tgr>', ] + suffix 
            # input_context = prefix + tgt + suffix 
            gold_evt_links = d['gold_evt_links'] 
            # if len(gold_evt_links)==0:
            #     continue
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
            
            # outputs = output_Context+output_trigger+output_role
            # out_lenth.append(len(gpt_tokenizer(outputs)['input_ids']))
            if task=='train':
                new_d = {'input_templete':input_template,"input_context":input_context,"summary":output_template,"reward":1.0}
            else:
                new_d = {'doc_key':doc_key,'input_templete':input_template,"input_context":input_context,"summary":output_template,"reward":1.0}
      
            all_data.append(new_d)

        return all_data
    
    def trans_generator_iter(data,model_type="decoder-only"):
        
        all_data = []
        for d in tqdm(data):
            Context = d['Context']
            Trigger = d['Trigger']
            event_type = d['Event type']
            arguments = d['arguments']
            reward =d['reward']
            # output_Context = f"Context: {Context}. "
            # output_trigger = f"Trigger: {Trigger}. "
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

            if model_type=="decoder-only":
                # inputs = f"Event type: {event_type}, Roles: {', '.join(all_roles)}. "
                # inputs =  f"Generate an event. Given the event type: {event_type}, and the following roles: {', '.join(all_roles)}. Please generate a coherent context and provide appropriate role-argument pairs."
                inputs = f"Generate an event. Given the event type: '{event_type}' and the following roles: {', '.join(all_roles)}, please generate a coherent context that includes the event trigger and the role-argument pairs. Make sure the event trigger and each argument for the corresponding role appear clearly in the context, and the context reflects the relationship between the trigger and its arguments."

                outputs = output_Context+output_trigger+output_role.strip()

            elif model_type=="encoder-decoder":
                inputs = f"generate event: Given the event type {event_type}, and  Roles: {', '.join(all_roles)}. "
                outputs = output_Context+output_trigger+output_role
        
            new_d = {'input':inputs,"label":outputs,"reward":reward}
            all_data.append(new_d)
        return all_data
    
    def trans_generator_special(task_name, meta_path, data_path,model_type="decoder-only"):
        
        def insert_special_tokens_by_traversal(words, trigger, arguments):
            # 记录每个位置是否是某些起点或终点
            start_map = {}  # idx -> [tags]
            end_map = {}    # idx -> [tags]

            # 添加触发词位置
            start_map.setdefault(trigger[0], []).append("<tgr>")
            end_map.setdefault(trigger[1] - 1, []).append("</tgr>")  # 在最后一个词后插入结束标记

            # 添加每个论元的位置
            for start, end in arguments:
                start_map.setdefault(start, []).append("<arg>")
                end_map.setdefault(end - 1, []).append("</arg>")

            # 遍历 token list 插入标记
            output = []
            for idx, token in enumerate(words):
                if idx in start_map:
                    output.extend(start_map[idx])
                output.append(token)
                if idx in end_map:
                    output.extend(end_map[idx])

            return " ".join(output)
        
        def read_data(data_path):
            data = []
            with open(data_path) as f:
                    for line in f.readlines():
                        data.append(json.loads(line))
            return data


        def get_meta(task_name,meta_path):
            meta = json.load(open(meta_path,'r',encoding='utf8'))
            # role2data = json.load(open(f"type2data_{task_name}.json",'r',encoding='utf8'))
            
            type2roles = {}
            for rams_m in meta:
                rams_event_type, role_types = rams_m
                if task_name=="rams":
                    role_types = [rams_role[11:].lower() for rams_role in role_types]
                else:
                    role_types = [rams_role.lower() for rams_role in role_types]
                type2roles[rams_event_type] = role_types
            return type2roles
        
        type2roles = get_meta(task_name,meta_path)
        data = read_data(data_path)
        all_data = []
        for d in tqdm(data):
            sentences = d['sentences']
            all_tokens = []
            for sentence in sentences:
                all_tokens.extend(sentence)
            # doc_text = ' '.join (all_tokens)
            event_type = d['evt_triggers'][0][-1][0][0]
            trigger_s,trigger_e = d['evt_triggers'][0][0],d['evt_triggers'][0][1]
         
            
            trigger = ' '.join(all_tokens[trigger_s:trigger_e+1])
            # tokens = pre +['<tgr>']+trigger_token+['<tgr>']+suffix
            # doc_text  = ' '.join (tokens)
            gold_evt_links = d['gold_evt_links'] 

            triggers_star_end=(trigger_s,trigger_e+1)

            all_roles = type2roles[event_type]
            
            context_role = ""
            role2span = {}
            inputs = f"Event type: {event_type}, Roles: {', '.join(all_roles)}. "
            for role in all_roles:
                role2span[role] = None
            args_star_end = []
            for gold_evt_link in gold_evt_links:
                current_role = gold_evt_link[-1]
                if task_name== 'rams':
                    current_role = current_role[11:]
                s,e = gold_evt_link[1]
                args_star_end.append((s,e+1))
                span = ' '.join(all_tokens[s:e+1])
                role2span[current_role] = span
            output_role = ''
            for role,span in role2span.items():
                output_role+=f"{role}: {span}, "

            context = insert_special_tokens_by_traversal(all_tokens,triggers_star_end,args_star_end)

            output_Context = f"Context: {context}. "
            output_trigger = f"Trigger: {trigger}. "

            if model_type=="decoder-only":
                outputs = inputs + output_Context+output_trigger+output_role
            elif model_type=="encoder-decoder":
                outputs = output_Context+output_trigger+output_role
            
            
            new_d = {'input':inputs,"label":outputs,"reward":1.0}
            all_data.append(new_d)
        return all_data
    
    
    def train_data_none_radio_compute(task_name, meta_path, data_path):
        def read_data(data_path):
            data = []
            with open(data_path) as f:
                    for line in f.readlines():
                        data.append(json.loads(line))
            return data
        def get_meta(task_name,meta_path):
            meta = json.load(open(meta_path,'r',encoding='utf8'))
            # role2data = json.load(open(f"type2data_{task_name}.json",'r',encoding='utf8'))
            
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


    def train_data_none_radio_compute_iter(data_path):
        def read_data(data_path):
            data = []
            with open(data_path) as f:
                    for line in f.readlines():
                        data.append(json.loads(line))
            return data
        
        data = read_data(data_path)
    
        none_radio_list = []

        for d in data:
            none_num = 0
            all_num = 0 
            
            all_num = len(d['arguments'])
            if  not d['reward']==1:
                continue
            for role ,span in d['arguments'].items():
                if not span:
                    none_num+=1
            none_radio_list.append(none_num/all_num) 
        
        return np.mean(none_radio_list),np.std(none_radio_list) 


    def trans_generator(task_name, meta_path, data_path,model_type="decoder-only"):
        def read_data(data_path):
            data = []
            with open(data_path) as f:
                    for line in f.readlines():
                        data.append(json.loads(line))
            return data
        def get_meta(task_name,meta_path):
            meta = json.load(open(meta_path,'r',encoding='utf8'))
            # role2data = json.load(open(f"type2data_{task_name}.json",'r',encoding='utf8'))
            
            type2roles = {}
            for rams_m in meta:
                rams_event_type, role_types = rams_m
                role_types = [
                        rams_role[11:].lower() if 'evt' in rams_role else rams_role.lower()
                        for rams_role in role_types
                    ]
     
                type2roles[rams_event_type] = role_types
            return type2roles
        
        def Filter_data(data,type2roles):
            none_data = []
            full_data = []
            for d in data :
                event_type = d['evt_triggers'][0][-1][0][0]
                gold_evt_links = d['gold_evt_links'] 
                arg_nums = len(gold_evt_links)
                all_roles_nums = len(type2roles[event_type])
                if arg_nums/all_roles_nums<0.3:
                    none_data.append(d)
                else:
                    full_data.append(d)
            return full_data,none_data
        
        type2roles = get_meta(task_name,meta_path)
        data = read_data(data_path)

        # if task_name=='wiki' and "dev" not in data_path:
        #     full_data,none_data = Filter_data(data,type2roles)
        #     max_none_empty = min(len(none_data), int(0.2 * len(full_data)))
        #     none_data = random.sample(none_data, max_none_empty)
        #     new_data = full_data+none_data

        #     data  = new_data
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
            
            if model_type=="decoder-only":
                inputs = f"Generate an event. Given the event type: '{event_type}' and the following roles: {', '.join(all_roles)}, please generate a coherent context that includes the event trigger and the role-argument pairs. Make sure the event trigger and each argument for the corresponding role appear clearly in the context, and the context reflects the relationship between the trigger and its arguments."

                outputs = output_Context+output_trigger+output_role.strip()

            elif model_type=="encoder-decoder":
                inputs = f"generate event: Given the event type {event_type}, and  Roles: {', '.join(all_roles)}. "
                outputs = output_Context+output_trigger+output_role
             
            
            
            new_d = {'input':inputs,"label":outputs,"reward":1.0}
            all_data.append(new_d)
        return all_data
    
    def save(self, path: str):
        Path(path).parent.mkdir(exist_ok=True, parents=True)
        with open(path, "w") as f:
            for s in self.sents:
                f.write(s.json() + "\n")


    def filter_labels(self, labels: List[str]):
        label_set = set(labels)
        sents = []
        for s in self.sents:
            triplets = [t for t in s.triplets if t.label in label_set]
            if triplets:
                s = s.copy(deep=True)
                s.triplets = triplets
                sents.append(s)
        return Dataset(sents=sents)

    def train_test_split(self, test_size: int, random_seed: int, by_label: bool):
        random.seed(random_seed)

        if by_label:
            labels = self.get_labels()
            labels_test = random.sample(labels, k=test_size)
            labels_train = sorted(set(labels) - set(labels_test))
            sents_train = self.filter_labels(labels_train).sents
            sents_test = self.filter_labels(labels_test).sents
        else:
            sents_train = [s for s in self.sents]
            sents_test = random.sample(self.sents, k=test_size)

        banned = set(s.text for s in sents_test)  # Prevent sentence overlap
        sents_train = [s for s in sents_train if s.text not in banned]
        assert len(self.sents) == len(sents_train) + len(sents_test)
        return Dataset(sents=sents_train), Dataset(sents=sents_test)

    def analyze(self):
        info = dict(
            sents=len(self.sents),
            unique_texts=len(set(s.triplets[0].text for s in self.sents)),
            lengths=str(Counter(len(s.triplets) for s in self.sents)),
            labels=len(self.get_labels()),
        )
        print(json.dumps(info, indent=2))


def write_data_splits(
    path_in: str,
    mode: str,
    folder_out: str = "outputs/data/splits/zero_rte",
    num_dev_labels: int = 5,
    num_test_labels: List[int] = [5, 10, 15],
    seeds: List[int] = [0, 1, 2, 3, 4],
):
    for n in num_test_labels:
        for s in seeds:
            if mode == "fewrel":
                data = Dataset.load_fewrel(path_in)
            elif mode == "wiki":
                data = Dataset.load_wiki(path_in)
            else:
                raise ValueError()

            train, test = data.train_test_split(
                test_size=n, random_seed=s, by_label=True
            )
            train, dev = train.train_test_split(
                test_size=num_dev_labels, random_seed=s, by_label=True
            )
            del data

            for key, data in dict(train=train, dev=dev, test=test).items():
                name = f"unseen_{n}_seed_{s}"
                path = Path(folder_out) / Path(path_in).stem / name / f"{key}.jsonl"
                data.save(str(path))
                print(dict(key=key, labels=len(data.get_labels()), path=path))


class Generator(BaseModel):
    load_dir: str
    save_dir: str
    num_gen_per_label: int = 250
    model_name: str = "generate"
    encoder_name: str = "generate"
    lora_dir: Optional[str]  = None
    model_kwargs: dict = {}

    def get_model(self) -> RelationModel:
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
        
        random.seed(model.random_seed)
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
            data_train = Dataset.trans_generator(task_name = task_name,meta_path=meta_path, data_path=path_train)
            data_dev = Dataset.trans_generator(task_name = task_name,meta_path=meta_path, data_path=path_dev)
        else:
            data_train = Dataset.load(path_train)
    
            data_train = Dataset.trans_generator_iter(data_train)
            data_dev = Dataset.trans_generator(task_name = task_name,meta_path=meta_path, data_path=path_dev)


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
                # 尝试解析 Context 和 Trigger，如果失败则跳过该样本
                Context = text.split("Context: ")[1].split("The trigger of the event in the context is : ")[0]
                # Trigger = text.split("Trigger: ")[1].split(". Role-Argument Pairs")[0]
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
                val = val.strip().strip('"').rstrip('.')  # 默认去掉结尾句点
                if role in role_list:
                    # 但如果 val 是缩写，如 U.N.、U.S.，不要去掉末尾句点
                    if re.fullmatch(r'[A-Z](?:\.[A-Z])+\.?', val):  # 匹配 U.N. 或 U.S.
                        val = val.rstrip('.') + '.'  # 保留末尾句点
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

    def generate_llama(self, labels: List[str], path_out: str):
        if Path(path_out).exists():
            return

        generator_model = self.get_model()
       

        base_model = run_clm_rl.Model.from_pretrained(generator_model.model_name, device_map="auto", torch_dtype=torch.float16)
        model = PeftModel.from_pretrained(base_model, generator_model.model_dir)  
        tokenizer = AutoTokenizer.from_pretrained(generator_model.model_name, use_fast=False)
        tokenizer.pad_token = tokenizer.eos_token 

        groups = {}
   
        all_data = []
        for relation in tqdm(labels, desc='Generating'):
            current_events = []
            current_num = len(current_events)
            
            event_type = list(relation.keys())[0]
            inputs = tokenizer(list(relation.values())[0], return_tensors="pt").to(model.device)
            total = 0
            while current_num<self.num_gen_per_label:
            
                outputs = model.generate(
                            **inputs,
                            max_length=512,
                            do_sample=True,          # 开启采样让结果多样
                            top_k=50,                # top-k 采样限制候选
                            top_p=0.95,              # nucleus 采样
                            temperature=0.7,         # 控制随机程度，越低越确定
                            num_return_sequences=100,  # 一次生成多条
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


    def generate_qwen(self, labels: List[str], path_out: str):
        if Path(path_out).exists():
            return

        generator_model = self.get_model()
       

        base_model = run_clm_rl.Model.from_pretrained(generator_model.model_name, device_map="auto",torch_dtype=torch.bfloat16)
        model = PeftModel.from_pretrained(base_model, generator_model.model_dir)  # 可选
        tokenizer = AutoTokenizer.from_pretrained(generator_model.model_name, use_fast=False,trust_remote_code=True )
        tokenizer.pad_token = tokenizer.eos_token  
   
        all_data = []
        for relation in tqdm(labels, desc='Generating'):
            current_events = []
            current_num = len(current_events)
            
            event_type = list(relation.keys())[0]
            inputs = tokenizer(list(relation.values())[0], return_tensors="pt").to(model.device)
            total = 0
            while current_num<self.num_gen_per_label:
            
                outputs = model.generate(
                            **inputs,
                            max_new_tokens=512,
                            do_sample=True,          # 开启采样让结果多样
                            top_k=50,                # top-k 采样限制候选
                            top_p=0.95,              # nucleus 采样
                            temperature=0.7,         # 控制随机程度，越低越确定
                            num_return_sequences=100,  # 一次生成多条
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
    
    def generate(self, labels: List[str], path_out: str):
        if Path(path_out).exists():
            return

        model = self.get_model()
        pipe = model.make_pipe()

        groups = {}
        assert isinstance(model, RelationGenerator)
        all_data = []
        for relation in tqdm(labels, desc='Generating'):
            generate_data, raw = model.generate(relation, self.num_gen_per_label, pipe=pipe)
            all_data.extend(generate_data)

        Path(path_out).parent.mkdir(exist_ok=True, parents=True)
        with open(path_out, "w") as f:
            for s in all_data:
                json.dump(s,f)
                f.write('\n')


    def estimate(self, path_in, path_out,path_train,iter,task_name,meta_path):
        # if Path(path_out).exists():
        #     return
        # def none_penalty(none_ratio, target_range=(0.1, 0.3), tolerance=0.05):
        #     lower, upper = target_range
        #     if lower - tolerance <= none_ratio <= upper + tolerance:
        #         return 0  # 在可接受范围内，不惩罚
        #     else:
        #         return abs(none_ratio - (lower + upper) / 2)
        
        def none_penalty(none_ratio,mean, std):
           
            if mean - std <= none_ratio <= mean + std:
                return 0  
            else:
                return abs(none_ratio - mean)
            
        if iter==0:
            mean, std  = Dataset.train_data_none_radio_compute(task_name, meta_path, data_path=path_train)
        else:
            mean, std = Dataset.train_data_none_radio_compute_iter(data_path=path_train)
        
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
            ins['penalty'] = penalty
            dataset.append(ins)
        with open(path_out, "w") as f:
            for d in dataset:
                json.dump(d,f,ensure_ascii=False)
                f.write('\n')

class Extractor(BaseModel):
    load_dir: str
    save_dir: str
    model_name: str = "new_extract"
    encoder_name: str = "extract"
    search_threshold: float = -0.9906
    model_kwargs: dict = {}

    def get_model(self) -> RelationModel:
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
        
        random.seed(model.random_seed)
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
            data_train = Dataset.trans_extractor(task_name = task_name,meta_path=meta_path, ontology_dict_path=ontology_dict_path,data_path=path_train,task='train')
            data_dev = Dataset.trans_extractor(task_name = task_name,meta_path=meta_path, ontology_dict_path=ontology_dict_path, data_path=path_dev,task='train')
        else:
            data_train = Dataset.load(path_train)
            data_train = Dataset.trans_extractor_iter(data_train,ontology_dict_path=ontology_dict_path)
            data_dev = Dataset.trans_extractor(task_name = task_name,meta_path=meta_path, ontology_dict_path=ontology_dict_path, data_path=path_dev,task='train')
        path_train = self.write_data(data_train, "train")
        path_dev = self.write_data(data_dev, "dev")

        model.fit(path_train=path_train, path_dev=path_dev)
        delete_checkpoints(model.model_dir)

    def predict(self, path_in: str, path_out: str, use_label_constraint: bool = True):
        if Path(path_out).exists():
            return
        
        data = Dataset.load(path_in)
  
        model = self.get_model()
        assert isinstance(model, NewRelationExtractor)
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
        pred = Dataset.load(path_pred)
        gold = Dataset.load(path_gold)
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
        bart_model = AutoModelForSeq2SeqLM.from_pretrained('experiments/rams2rams/output_llama_修正/extractor/iter1/model').to('cuda')
        bart_tokenizer = AutoTokenizer.from_pretrained('experiments/rams2rams/output_llama_修正/extractor/iter1/model')
        
        data_in = Dataset.load(path_in)
        dataset = []


        for ins_i, ins in enumerate(data_in):
            # if ins.get('extractor_nll'):
            #         return
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


def main(
    path_train: str,
    path_dev: str,
    path_test: str,
    save_dir: str,
    g_encoder_name: str,
):
    print(dict(main=locals()))
    generator = Generator(
        load_dir="gpt2",
        save_dir=str(Path(save_dir) / "generator"),
        encoder_name=g_encoder_name
    )
    extractor = Extractor(
        load_dir="facebook/bart-base",
        save_dir=str(Path(save_dir) / "extractor"),
    )

    generator.fit(path_train, path_dev)
    extractor.fit(path_train, path_dev)
    path_synthetic = str(Path(save_dir) / "synthetic.jsonl")
    labels_dev = Dataset.load(path_dev).get_labels()
    labels_test = Dataset.load(path_test).get_labels()
    generator.generate(labels_dev + labels_test, path_out=path_synthetic)

    extractor_final = Extractor(
        load_dir=str(Path(save_dir) / "extractor" / "model"),
        save_dir=str(Path(save_dir) / "extractor_final"),
    )
    extractor_final.fit(path_synthetic, path_dev)

    # path_pred = str(Path(save_dir) / "pred.jsonl")
    # extractor_final.predict(path_in=path_test, path_out=path_pred)
    # results = extractor_final.score(path_pred, path_test)
    # print(json.dumps(results, indent=2))
    # with open(Path(save_dir) / "results.json", "w") as f:
    #     json.dump(results, f, indent=2)
    run_eval(path_model=str(Path(save_dir) / "extractor_final"), path_test=path_test, mode='single', is_eval=False)
    run_eval(path_model=str(Path(save_dir) / "extractor_final"), path_test=path_dev, mode='single', is_eval=True)

    run_eval(path_model=str(Path(save_dir) / "extractor_final"), path_test=path_test, mode='multi', is_eval=False)

    # return results

def main_dpo(path_train: str,
    path_dev: str,
    path_test: str,
    save_dir: str,
):
    print(dict(main_dpo=locals()))
    generator = Generator(
        load_dir=str(Path(save_dir) / "generator_dpo" / 'model'),
        save_dir=str(Path(save_dir) / "generator_dpo"),
    )

    # generator.fit(path_train, path_dev)
    path_synthetic = str(Path(save_dir) / "generator_dpo" / "synthetic.jsonl")
    labels_dev = Dataset.load(path_dev).get_labels()
    labels_test = Dataset.load(path_test).get_labels()
    generator.generate(labels_dev + labels_test, path_out=path_synthetic)

    extractor_final = Extractor(
        load_dir=str(Path(save_dir) / "extractor" / "model"),
        save_dir=str(Path(save_dir) / "extractor_final_dpo"),
    )
    extractor_final.fit(path_synthetic, path_dev)

    # path_pred = str(Path(save_dir) / "pred.jsonl")
    # extractor_final.predict(path_in=path_test, path_out=path_pred)
    # results = extractor_final.score(path_pred, path_test)
    # print(json.dumps(results, indent=2))
    # with open(Path(save_dir) / "results.json", "w") as f:
    #     json.dump(results, f, indent=2)
    run_eval(path_model=str(Path(save_dir) / "extractor_final_dpo"), path_test=path_test, mode='single', is_eval=False)
    run_eval(path_model=str(Path(save_dir) / "extractor_final_dpo"), path_test=path_test, mode='multi', is_eval=False)

    # return results


def main_pseudo(
    path_train: str,
    path_dev: str,
    path_test: str,
    save_dir: str,
    num_iter: int
):
    print(dict(main=locals()))
    generator = Generator(
        load_dir="gpt2",
        save_dir=str(Path(save_dir) / "generator/iter0"),
    )
    extractor = Extractor(
        load_dir="facebook/bart-base",
        save_dir=str(Path(save_dir) / "extractor/iter0"),
    )

    generator.fit(path_train, path_dev)
    extractor.fit(path_train, path_dev)
    
    labels_dev = Dataset.load(path_dev).get_labels()
    labels_test = Dataset.load(path_test).get_labels()
    for i in range(num_iter):
        path_synthetic = str(Path(save_dir) / "synthetic" / f"{i}.jsonl")
        # path_synthetic_generator = str(Path(save_dir) / "synthetic" / f"{i}_gen.jsonl")
        # path_synthetic_extractor = str(Path(save_dir) / "synthetic" / f"{i}_ext.jsonl")
        generator.generate(labels_dev + labels_test, path_out=path_synthetic)
        # extractor.estimate(path_synthetic, path_synthetic_extractor)
        # generator.estimate(path_synthetic, path_synthetic_generator)
        path_filtered = path_synthetic
        extractor = Extractor(
            load_dir=str(Path(save_dir) / "extractor" / f'iter{i}' / "model"),
            save_dir=str(Path(save_dir) / "extractor" / f'iter{i+1}'),
        )
        generator = Generator(
            load_dir=str(Path(save_dir) / "generator" / f'iter{i}' / "model"),
            save_dir=str(Path(save_dir) / "generator" / f'iter{i+1}'),
        )
        extractor.fit(path_filtered, path_dev)
        generator.fit(path_filtered, path_dev)

        # path_pred_dev = str(Path(save_dir) / "pred_dev" / f"{i}.jsonl")
        # path_pred_test = str(Path(save_dir) / "pred_test" / f"{i}.jsonl")
        run_eval(path_model=str(Path(save_dir) / "extractor" / f'iter{i+1}'), 
                 path_test=path_test, mode='single', is_eval=False)
        run_eval(path_model=str(Path(save_dir) / "extractor" / f'iter{i+1}'), 
                 path_test=path_test, mode='multi', is_eval=False)
        run_eval(path_model=str(Path(save_dir) / "extractor" / f'iter{i+1}'), 
                 path_test=path_dev, mode='single', is_eval=True)
        run_eval(path_model=str(Path(save_dir) / "extractor" / f'iter{i+1}'), 
                 path_test=path_dev, mode='multi', is_eval=True)

def sort_data(data, version):
    groups = defaultdict(list)
    assert version in ['single', 'all']
    for item in data:
        event_type = item['Event type']
        groups[event_type].append(item)
  
    
    # version 1 tabby
    gen_list = []
    ext_list = []
    for k in groups:
        for tri in groups[k]:
            gen_list.append(tri['cond_generator_nll'])
            ext_list.append(tri['extractor_nll'])
    gen_mean, gen_std = np.mean(gen_list), np.std(gen_list)
    ext_mean, ext_std = np.mean(ext_list), np.std(ext_list)
    std_func = lambda x, mean, std: ((x - mean) / std) if std != 0 else (x - mean) 
    # nll_func = lambda x: std_func(x['extractor_nll'], ext_mean, ext_std) + std_func(x['cond_generator_nll'], gen_mean, gen_std)
    def nll_func(x):
        nll = std_func(x['extractor_nll'], ext_mean, ext_std) + std_func(x['cond_generator_nll'], gen_mean, gen_std)
        return nll + x.get('penalty', 0)  
    max_ll, min_ll = None, None
    for k in groups:
        groups[k] = sorted(groups[k], key=lambda x: nll_func(x))
        ll_list = [-nll_func(x) for x in groups[k]]
        max_ll = max(max_ll, max(ll_list)) if max_ll is not None else max(ll_list)
        min_ll = min(min_ll, min(ll_list)) if min_ll is not None else min(ll_list)
    for k in groups:
        scaler_func = lambda x: (x - min_ll) / (max_ll - min_ll) if max_ll != min_ll else 1.
        for tri in groups[k]:
            tri['reward'] = scaler_func(-nll_func(tri))
            # if tri.get('penalty'):
            #     tri['reward'] = tri['reward'] - tri['penalty']
    return groups 
          
def filter_data_nb_rel(path_pseudo, path_out):
 

    pseudo_data = Dataset.load(path_pseudo)
 
    # 按relation分类，并排序
    pseudo_data = sort_data(pseudo_data, version='all')


    Path(path_out).parent.mkdir(exist_ok=True, parents=True)
    # with open(path_out, "w") as f:
    #     for d in all_data:
    #         json.dump(d,f,ensure_ascii=False)
    #         f.write('\n')

def read_data(data_path):
    data = []
    with open(data_path) as f:
            for line in f.readlines():
                data.append(json.loads(line))
    return data

if __name__ == "__main__":
    match_extractor_nll_list = []
    missmatch_extractor_nll_list = []
    path_synthetic='experiments/rams2rams/output_llama_修正/synthetic/1.jsonl'
   
    synthetic_data = read_data(path_synthetic)
    pseudo_data = sort_data(synthetic_data, version='all')
    t = 1


    # main_dual(path_synthetic='mismatch/processed.jsonl',
    #     ontology_dict_path='/root/siton-data-Hzhang/ZGJ/2025/llm_zero_eae/data/rams.csv'
    #     )
    
  

   
  
