import json
import random
import math 
import numpy as np
from pathlib import Path
from typing import List
import torch
import re 
from collections import defaultdict
from tqdm import tqdm
from transformers import set_seed as hf_set_seed
import pandas as pd

import os
import argparse
from Agent.generator import Generator
from Agent.evaluator import Extractor

from Dataset import Dataset


def sort_data(data):
    groups = defaultdict(list)
    for item in data:
        event_type = item['Event type']
        groups[event_type].append(item)
  
    
    
    gen_list = []
    ext_list = []
    for k in groups:
        for tri in groups[k]:
            gen_list.append(tri['cond_generator_nll'])
            ext_list.append(tri['extractor_nll'])
    gen_mean, gen_std = np.mean(gen_list), np.std(gen_list)
    ext_mean, ext_std = np.mean(ext_list), np.std(ext_list)
    std_func = lambda x, mean, std: ((x - mean) / std) if std != 0 else (x - mean) 

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
         
    return groups 


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

def trans_format(data,meta_path,task_name):

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
  
        role2span = {}

        for role in all_roles:
            role2span[role.lower()] = None
        
        for gold_evt_link in gold_evt_links:
            current_role = gold_evt_link[-1]
            current_role = current_role.lower()
            
            if 'evt' in current_role:
                current_role = current_role[11:]

            s,e = gold_evt_link[1]
            span = ' '.join(all_tokens[s:e+1])
            role2span[current_role] = span

        output_role = {}
        for role,span in role2span.items():
            output_role[role]= span
    
    
        new_d = {'Context':doc_text,"Trigger":trigger,"Event type":event_type,'arguments':output_role,'extractor_nll':0,"reward": 1.0,"cond_generator_nll":0.0,"generator_nll":0.0}
        all_data.append(new_d)
    return all_data
            
def filter_data_nb_rel(path_pseudo, 
                        path_train,
                        path_out, 
                        total_pseudo_per_label, 
                        pseudo_ratio,
                        task_name,
                        unseen_meta_path,
                        meta_path,
                        Sampling_cfg):
    print(dict(filter_data_nb_rel=locals()))
    if Path(path_out).exists():
        return 
    def count_non_empty_args(tri):
        return sum(1 for v in tri['arguments'].values() if v and v.lower() != "none")
        

    train_data = Dataset().load(path_train)
    pseudo_data = Dataset().load(path_pseudo)
    train_data = trans_format(task_name = task_name,meta_path=meta_path, data=train_data)

    pseudo_labels = get_meta(task_name,unseen_meta_path)
    num_pseudo = int(total_pseudo_per_label * pseudo_ratio * len(pseudo_labels))

    num_pseudo_per_label = int(num_pseudo / len(pseudo_labels))

    # 按relation分类，并排序
    pseudo_data = sort_data(pseudo_data)
    train_data = sort_data(train_data)
    
    nb_rel_train_data = [x for k, v in train_data.items() for x in v]
    nb_rel_pseudo_data = [x for k, v in pseudo_data.items() for x in v]

    origin_pseudo_data = sorted(nb_rel_pseudo_data, key=lambda x: x['reward'], reverse=True)

    pseudo_empty_threshold = Sampling_cfg['pseudo_empty_threshold']
    pseudo_empty_ratio = Sampling_cfg['pseudo_empty_ratio']
    non_empty_data = [x for x in origin_pseudo_data if count_non_empty_args(x) >= pseudo_empty_threshold]
    empty_data = [x for x in origin_pseudo_data if count_non_empty_args(x) < pseudo_empty_threshold]
    nb_rel_pseudo_data = non_empty_data[:num_pseudo]
    max_empty = min(len(empty_data), int(pseudo_empty_ratio * len(nb_rel_pseudo_data)))
    nb_rel_pseudo_data += empty_data[:max_empty]


    for rel in pseudo_labels:
        rel_pseudo_data = [x for x in nb_rel_pseudo_data if x['Event type'] == rel]
        num = len(rel_pseudo_data)
        if num < num_pseudo_per_label / 5:
            rel_data = [x for x in origin_pseudo_data if x['Event type'] == rel and x not in rel_pseudo_data]
            nb_rel_pseudo_data.extend(rel_data[:num_pseudo_per_label // 5 - num])
    pseudo_reward = [x['reward'] for x in nb_rel_pseudo_data]
    mean_pseudo_reward = np.mean(pseudo_reward)
    
    all_data = nb_rel_train_data + nb_rel_pseudo_data
    
    Path(path_out).parent.mkdir(exist_ok=True, parents=True)
    with open(path_out, "w") as f:
        for d in all_data:
            json.dump(d,f,ensure_ascii=False)
            f.write('\n')

    print('num of filtered data:', len(all_data), 'mean pseudo reward:', mean_pseudo_reward)


def main_dual(
    path_train: str,
    path_dev: str,
    path_test: str,
    task_name:str,
    meta_path:str,
    unseen_meta:str,
    ontology_dict_path:str,
    ontology_test_dict_path:str,
    generator_unseen_label_path,
    save_dir: str,
    num_iter: int,
    limit: int = 5000,
    g_encoder_name: str = 'generate', 
    num_gen_per_label: int = 250, 
    diverse: bool = False, 
    gen_PLM_Path:str='GPT2',
    ext_PLM_Path:str='Bart-large',
    Sampling_cfg:dict= {}
):
    print(dict(main=locals()))
    

    generator = Generator(
            load_dir=gen_PLM_Path,
            save_dir=str(Path(save_dir) / "generator/iter0"),
            num_gen_per_label=num_gen_per_label + 100 if not diverse else num_gen_per_label * 2
        )
    
        
    extractor = Extractor(
        load_dir=ext_PLM_Path,
        save_dir=str(Path(save_dir) / "extractor/iter0")
    )
    
    task_train_data,task_test_data = task_name.split('2')


    generator.fit(path_train, path_dev,task_train_data,meta_path,  iter = 0)
    extractor.fit(path_train, path_dev,task_train_data,meta_path,ontology_dict_path,iter = 0)
    run_eval(path_model=str(Path(save_dir) / "extractor" / f'iter{0}'), 
                 path_test=path_test, mode='single', is_eval=True, limit=limit,meta_path=unseen_meta, ontology_dict_path=ontology_test_dict_path,task_name =task_test_data)
       

    labels_test = json.load(open(generator_unseen_label_path,'r'))
    for i in range(num_iter):
        
        path_synthetic = str(Path(save_dir) / "synthetic" / f"{i}.jsonl")
        path_synthetic_generator = str(Path(save_dir) / "synthetic" / f"{i}_gen.jsonl")
        path_synthetic_extractor = str(Path(save_dir) / "synthetic" / f"{i}_ext.jsonl")
  
        generator.generate(labels_test, path_out=path_synthetic)      
        generator.estimate(path_synthetic, path_synthetic,path_train,task_name=task_name,meta_path=meta_path)
        extractor.estimate(path_synthetic, path_synthetic,ontology_test_dict_path)
        

        # filter
        path_filtered = str(Path(save_dir) / "filtered" / f"{i}.jsonl")
        filter_data_nb_rel(path_synthetic, 
                        path_train, 
                        path_filtered, 
                        num_gen_per_label, 
                        (i + 1.)/ num_iter,
                        task_name=task_train_data,
                        unseen_meta_path=unseen_meta,
                        meta_path=meta_path,
                        Sampling_cfg=Sampling_cfg)

        
        extractor = Extractor(
            load_dir=str(Path(save_dir) / "extractor" / f'iter{i}' / "model"),
            save_dir=str(Path(save_dir) / "extractor" / f'iter{i+1}'),
        )

        generator = Generator(
            load_dir=gen_PLM_Path,
            lora_dir=str(Path(save_dir) / "generator" / f'iter{i}' / "model"),
            save_dir=str(Path(save_dir) / "generator" / f'iter{i+1}'),
            num_gen_per_label=num_gen_per_label + 100, 
            encoder_name=g_encoder_name,
        )
    
        extractor.fit(path_filtered, path_dev,task_train_data,meta_path,ontology_dict_path,iter = i+1)
        run_eval(path_model=str(Path(save_dir) / "extractor" / f'iter{i+1}'), 
                 path_test=path_test, mode='single', is_eval=True, limit=limit,meta_path=unseen_meta, ontology_dict_path=ontology_test_dict_path,task_name =task_test_data)
         
        generator.fit(path_filtered, path_dev,task_train_data,meta_path,  iter = i+1)
      
    

def run_eval(path_model: str, path_test: str, mode: str, is_eval: bool, ontology_dict_path:str,meta_path:str,task_name:str,limit: int = 0):
    print(dict(run_eval=locals()))
    is_eval = f'is_eval_{is_eval}'
    path_results = str(Path(path_model) / f"results_{mode}_{is_eval}.json") if limit == 0 else str(Path(path_model) / f"results_{mode}_{is_eval}_limit{limit}.json")
    if Path(path_results).exists():
        return 
    

    data = Dataset().trans_extractor(data_path=path_test,meta_path=meta_path,ontology_dict_path=ontology_dict_path,task_name=task_name,task='eval')
    model = Extractor(load_dir=str(Path(path_model) / "model"), save_dir=path_model)
    
  
    if limit > 0:
        random.seed(0)
        random.shuffle(data)
      
    path_in = str(Path(path_model) / f"pred_in_{mode}_{is_eval}.jsonl") if limit == 0 else str(Path(path_model) / f"pred_in_{mode}_{is_eval}_limit{limit}.jsonl")
    path_out = str(Path(path_model) / f"pred_out_{mode}_{is_eval}.jsonl") if limit == 0 else str(Path(path_model) / f"pred_out_{mode}_{is_eval}_limit{limit}.jsonl")
    
    Path(path_in).parent.mkdir(exist_ok=True, parents=True)
    with open(path_in, "w") as f:
        for d in data:
            json.dump(d,f,ensure_ascii=False)
            f.write('\n')


    model.predict(path_in, path_out)

def set_seed(seed=42):
    # 环境变量（cuBLAS 可复现设置，OpenMP/MKL 线程数）
    os.environ["PYTHONHASHSEED"] = str(seed)
    # cuBLAS 可复现（某些 CUDA 版本需要这个）
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    # 限制线程避免多线程引入的非确定性
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    # Python / numpy / random
    random.seed(seed)
    np.random.seed(seed)

    # PyTorch CPU / CUDA
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # 限制 PyTorch 使用的线程数
    try:
        torch.set_num_threads(1)
    except Exception:
        pass

    # 让 cudnn 更可复现（代价是性能）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 强制使用可复现算法（PyTorch >=1.8），若出现不支持的操作会抛错
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        # 在某些 PyTorch 版本或环境下可能不支持，继续但提示
        print("Warning: torch.use_deterministic_algorithms not available or failed")



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run main_dual generation-and-evaluation loop")
    parser.add_argument("--path-train", default="dataset/rams2rams/train.jsonl")
    parser.add_argument("--path-dev", default="dataset/rams2rams/dev.jsonl")
    parser.add_argument("--path-test", default="dataset/rams2rams/test.jsonl")
    parser.add_argument("--seen-meta", default="dataset/rams2rams/meta_seen.json")
    parser.add_argument("--unseen-meta", default="dataset/rams2rams/meta_unseen.json")
    parser.add_argument("--ontology-dict-path", default="dataset/rams2rams/ontology.csv")
    parser.add_argument("--ontology-test-dict-path", default="dataset/rams2rams/ontology.csv")
    parser.add_argument("--generator-unseen-label-path", default="data/rams2rams/generator_unseen_label_rams2rams_code.json")
    parser.add_argument("--save-dir", default="experiments/rams2rams/output_llama")
    parser.add_argument("--seed", default=42, type=int )
    parser.add_argument("--with-train", dest="with_train", action="store_true")
    parser.add_argument("--num-iter", type=int, default=5)
    parser.add_argument("--by-rel", action="store_true", default=False)
    parser.add_argument("--gen-plm-path", default="Meta-Llama-3-8B")
    parser.add_argument("--ext-plm-path", default="/root/siton-data-Hzhang/pretrained-model/English/bart-large")
    parser.add_argument("--block-size", default=512, type=int)
    parser.add_argument("--task-name", default='rams2rams', type=str)
    parser.add_argument("--num-gen-per-label", type=int, default=100)
    parser.add_argument("--pseudo-empty-ratio",default= 0.1, type=float)
    parser.add_argument("--pseudo-empty-threshold",default= 2, type=float)
 
    args = parser.parse_args()
    #rams2rams
    set_seed(args.seed)
    hf_set_seed(args.seed)
    main_dual(path_train=args.path_train,
            path_dev = args.path_dev,
            path_test = args.path_test,
            task_name = args.task_name,
            meta_path =  args.seen_meta,
            unseen_meta =  args.unseen_meta,
            ontology_dict_path= args.ontology_dict_path,
            ontology_test_dict_path =  args.ontology_dict_path,
            generator_unseen_label_path = args.ontology_test_dict_path,
            save_dir = args.save_dir,
            num_iter= args.num_iter,
            num_gen_per_label = args.num_gen_per_label,
            gen_PLM_Path = args.gen_plm_path,
            ext_PLM_Path = args.ext_plm_path,
            Sampling_cfg = {'pseudo_empty_ratio':  args.pseudo_empty_ratio,
                            "pseudo_empty_threshold": args.pseudo_empty_threshold})



    #  # llama
    # Sampling_cfg_wiki2wiki_llama ={'pseudo_empty_ratio':  0.1,
    #                          "pseudo_empty_threshold": 1,
    #                          'train_empty_ratio': 1,
    #                          'train_empty_threshold':2}
    # #qwen
    # Sampling_cfg_wiki2wiki_qwen ={'pseudo_empty_ratio':  0.1,
    #                          "pseudo_empty_threshold": 2,
    #                          'train_empty_ratio': 1,
    #                         'train_empty_threshold':2}

    # main_dual(path_train='/root/siton-data-Hzhang/ZGJ/2025/llm_zero_eae/data/wiki2wiki/train.jsonl',
    #         path_dev = '/root/siton-data-Hzhang/ZGJ/2025/llm_zero_eae/data/wiki2wiki/dev.jsonl',
    #         path_test = "/root/siton-data-Hzhang/ZGJ/2025/llm_zero_eae/data/wiki2wiki/test.jsonl",
    #         task_name = 'wiki2wiki',
    #         meta_path = '/root/siton-data-Hzhang/ZGJ/2025/llm_zero_eae/data/wikievents/meta.json',
    #         unseen_meta = '/root/siton-data-Hzhang/ZGJ/2025/llm_zero_eae/data/wiki2wiki/wiki2wiki_unseen_meta.json',
    #         ontology_dict_path='/root/siton-data-Hzhang/ZGJ/2025/llm_zero_eae/data/wiki.csv',
    #         ontology_test_dict_path = '/root/siton-data-Hzhang/ZGJ/2025/llm_zero_eae/data/wiki.csv',
    #         generator_unseen_label_path = '/root/siton-data-Hzhang/ZGJ/2025/llm_zero_eae/data/wiki2wiki/generator_unseen_label_wiki2wiki.json',
    #         save_dir = 'experiments/wiki2wiki/output_llama_seed_42',
    #         with_train = True,
    #         num_iter= 5,
    #         score_only_ext = False,
    #         by_rel = False,
    #         rl_version = "all",
    #         rescale_train = False,
    #         num_gen_per_label = 50,
    #         gen_PLM_Path = 'llama',
    #         Sampling_cfg = Sampling_cfg_wiki2wiki_llama)
   
   

    # Sampling_cfg_rams2wiki_qwen ={'pseudo_empty_ratio':  0.5,
    #                          "pseudo_empty_threshold": 1,
    #                          'train_empty_ratio': 0.1,
    #                          'train_empty_threshold':2}
    
    # #   训练集： 所有论元数大于2 的+50 %论元小于1
    # #   伪数据 所有论元数大于2 的+50 %论元小于2
    
    # Sampling_cfg_rams2wiki_llama ={'pseudo_empty_ratio':  0.5,
    #                         "pseudo_empty_threshold": 1,
    #                         'train_empty_ratio': 0.1,
    #                         'train_empty_threshold':2}

 

    # main_dual(path_train='/root/siton-data-Hzhang/ZGJ/2025/llm_zero_eae/data/rams2wiki/train.jsonl',
    #         path_dev = '/root/siton-data-Hzhang/ZGJ/2025/llm_zero_eae/data/rams2wiki/dev.jsonl',
    #         path_test = "/root/siton-data-Hzhang/ZGJ/2025/llm_zero_eae/data/rams2wiki/test.jsonl",
    #         task_name = 'rams2wiki',
    #         meta_path = '/root/siton-data-Hzhang/ZGJ/2025/llm_zero_eae/data/rams2wiki/merged_meta.json',
    #         unseen_meta = '/root/siton-data-Hzhang/ZGJ/2025/llm_zero_eae/data/rams2wiki/rams2wiki_meta.json',
    #         ontology_dict_path='/root/siton-data-Hzhang/ZGJ/2025/llm_zero_eae/data/rams2wiki/merged_ontology.csv',
    #         ontology_test_dict_path = '/root/siton-data-Hzhang/ZGJ/2025/llm_zero_eae/data/rams2wiki/merged_ontology.csv',
    #         generator_unseen_label_path = '/root/siton-data-Hzhang/ZGJ/2025/llm_zero_eae/data/rams2wiki/generator_unseen_label_rams2wiki.json',
    #         save_dir = 'experiments/rams2wiki/output_llama_seed_42',
    #         with_train = True,
    #         num_iter= 5,
    #         score_only_ext = False,
    #         by_rel = False,
    #         rl_version = "all",
    #         rescale_train = False,
    #         num_gen_per_label = 500,
    #         gen_PLM_Path = 'llama',
    #         Sampling_cfg=Sampling_cfg_rams2wiki_llama)

  
