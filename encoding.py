from pathlib import Path
from typing import Dict, List, Tuple

# from fire import Fire
from pydantic import BaseModel
from tqdm import tqdm
from transformers import AutoTokenizer
import random 
from transformer_base import run_summarization_rl

import re


class GenerateEncoder(BaseModel):
    def encode_x(self, r: str) -> str:
        return f"Relation : {r} ."

    def decode_x(self, text: str) -> str:
        event_type = text.split(", Roles: ")[0].split('Event type: ')[1]
        role_list = text.split(", Roles: ")[1].split(', ')
        return event_type, role_list

   
    def decode_y(self, text: str):
        
        try:
                role_list = text.split('following roles: ')[1].split(", please generate")[0].split(',')
                role_list = [role.strip().lower() for role in role_list] 
                Context = text.split("Context: ")[1].split("The trigger of the event in the context is : ")[0]
                Trigger = text.split("The trigger of the event in the context is :")[1].split(". Role-Argument Pairs")[0].strip()
                Role_Argument = text.split(". Role-Argument Pairs: ")[1]
        except (IndexError, AttributeError):
            return None
        role2arg = {}
        result = {}
        for chunk in Role_Argument.split(','):
                if ':' not in chunk:
                    continue
                role, val = chunk.strip().split(':', 1)
                role = role.strip()
                val = val.strip().strip('"').rstrip('.')  # 默认去掉结尾句点
                if role in role_list:
                    # 但如果 val 是缩写，如 U.N.、U.S.，不要去掉末尾句点
                    if re.fullmatch(r'[A-Z](?:\.[A-Z])+\.?', val):  # 匹配 U.N. 或 U.S.
                        val = val.rstrip('.') + '.'  # 保留末尾句点
                    if val.lower() == "none":
                        val = None
                    role2arg[role] = val
        if len(role2arg)==0 :
            return None
        result['Context']= Context
        result['Trigger']= Trigger
        result['Event type'] = event_type
        result["arguments"] = role2arg
        return result


class ExtractEncoder(BaseModel):
    def encode_x(self, text: str) -> str:
        return f"Context : {text}"
  
    def decode_x(self, x: str) -> str:
        return x.split("Context : ")[-1]

    def encode_entity_prompt(self, head: str, tail: str) -> str:
        return f"Head Entity : {head} , Tail Entity : {tail} , Relation :"

  

    def parse_line(self, line: str) -> Tuple[str, str]:
        return run_summarization.decode_from_line(line)



def select_encoder(name: str):
    mapping: Dict[str, Encoder] = dict(
        extract=ExtractEncoder(),
        generate=GenerateEncoder(),
    )
    encoder = mapping[name]
    return encoder

