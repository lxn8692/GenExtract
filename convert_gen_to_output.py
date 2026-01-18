import os 
import json 
import argparse 
import re 
from copy import deepcopy
from tqdm import tqdm 
import pandas as pd
from utils import find_head, WhitespaceTokenizer, find_arg_span
import spacy 



def extract_args_from_template(ex, template, ontology_dict,):
    # extract argument text 
        
    template_words = template.strip().split()
    predicted_words = ex['predicted'].strip().split()    
    predicted_args = {}
    t_ptr = 0
    p_ptr = 0 
    evt_type = get_event_type(ex)[0]

    while t_ptr < len(template_words) and p_ptr < len(predicted_words):
        if re.match(r'<(arg\d+)>', template_words[t_ptr]):
            m = re.match(r'<(arg\d+)>', template_words[t_ptr])
            arg_num = m.group(1)
            arg_name = ontology_dict[evt_type.replace('n/a','unspecified')][arg_num]

            if predicted_words[p_ptr] == '<arg>':
                # 缺失参数，用 <arg> 占位
                p_ptr += 1 
                t_ptr += 1  
            else:
                arg_start = p_ptr 
                if t_ptr + 1 < len(template_words):
                    # 一直找到下一个锚点词（模板中当前 <argX> 后的词）
                    next_token = template_words[t_ptr + 1]
                    while p_ptr < len(predicted_words) and predicted_words[p_ptr] != next_token:
                        p_ptr += 1 
                else:
                    # 如果是最后一个参数，没有下一个锚点，就把剩下的词全拿走
                    p_ptr = len(predicted_words)

                arg_text = predicted_words[arg_start:p_ptr]
                predicted_args[arg_name] = arg_text 
                t_ptr += 1 
        else:
            t_ptr += 1 
            p_ptr += 1 

    return predicted_args

def get_event_type(ex):
        evt_type = []
        for evt in ex['evt_triggers']:
            for t in evt[2]:
                evt_type.append( t[0])
        return evt_type 

def check_coref(ex, arg_span, gold_spans):
    for clus in ex['corefs']:
        if arg_span in clus:
            matched_gold_spans = [span for span in gold_spans if span in clus]
            if len(matched_gold_spans) > 0:
                return matched_gold_spans[0]
    return arg_span 
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()


    parser.add_argument('--gen-file',type=str, default='experiments/wiki2wiki/output_llama/extractor/iter0/pred_out_single_is_eval_True_limit5000.jsonl')
    parser.add_argument('--test_file', type=str,default='/root/siton-data-Hzhang/ZGJ/2025/llm_zero_eae/data/wiki2wiki/test.jsonl')
    parser.add_argument('--output-file',type=str, default='test_output.jsonl')
    parser.add_argument('--ontology',default='/root/siton-data-Hzhang/ZGJ/2025/llm_zero_eae/data/wiki.csv',type=str)
    parser.add_argument('--head-only',action='store_true',default=False)
    parser.add_argument('--coref', action='store_true', default=False)
    args = parser.parse_args() 

    nlp = spacy.load('en_core_web_sm')
    nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)
    # read ontology 

    ontology_dict ={} 
    df =pd.read_csv(args.ontology, quotechar='"')
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
                        role = arg
                        ontology_dict[evt_type][f'arg{i+1}'] = role
                        ontology_dict[evt_type][role] = f'arg{i+1}'
    
    
    examples = {}
    with open(args.test_file, 'r') as f:
       
        for line in f:
            ex = json.loads(line.strip())
            ex['ref_evt_links'] = deepcopy(ex['gold_evt_links']) 
            ex['gold_evt_links'] = []
            examples[ex['doc_key']] =ex 
        
    
    with open(args.gen_file,'r') as f:
        for line in f:
            pred = json.loads(line.strip()) 
            examples[pred['doc_key']]['predicted'] = pred['predicted']
            examples[pred['doc_key']]['gold'] = pred['gold']

    
    writer = open(args.output_file, 'w') 
    for ex in tqdm(examples.values()):
        if 'predicted' not in ex:# this is used for testing 
            continue 
        # get template 
        evt_type = get_event_type(ex)[0]
        context_words = [w for sent in ex['sentences'] for w in sent ]
        template = ontology_dict[evt_type.replace('n/a','unspecified')]['template']
        # extract argument text 
    
        predicted_args = extract_args_from_template(ex,template, ontology_dict)
        # get trigger 
        # extract argument span
        trigger_start = ex['evt_triggers'][0][0]
        trigger_end = ex['evt_triggers'][0][1]
        doc = None 
        if args.head_only:
            doc = nlp(' '.join(context_words))

        for argname in predicted_args:
            arg_span = find_arg_span(predicted_args[argname], context_words, 
                trigger_start, trigger_end, head_only=args.head_only, doc=doc) 
            if arg_span:# if None means hullucination
                
                if args.head_only and args.coref:
                    # consider coreferential mentions as matching 
                    assert('corefs' in ex)
                    gold_spans = [a[1] for a in ex['ref_evt_links'] if a[2]==argname]
                    arg_span = check_coref(ex, list(arg_span), gold_spans)

                ex['gold_evt_links'].append([[trigger_start, trigger_end], list(arg_span), argname])

        writer.write(json.dumps(ex)+'\n')
    
    writer.close() 