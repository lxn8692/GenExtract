from typing import Dict, List, Optional, Tuple

import torch
# from fire import Fire
from torch import Tensor
from transformers import PreTrainedModel, PreTrainedTokenizerFast

from utils import DynamicModel


class TextGenerator(DynamicModel):
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerFast
    scores: Optional[List[Tensor]] = None
    max_length: int

    def tokenize(self, texts: List[str], **kwargs):
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            **kwargs,
        ).to(self.model.device)

    def run(
        self,
        texts: List[str],
        do_sample=True,
        top_k=50,
        temperature=1.0,
        num_return: int = 4,
        prompt: Optional[str] = None,
        prompt_ids: Optional[List[int]] = None,
        multi_prompt_ids: Optional[List[List[int]]] = None,
        decoder_input_ids: Optional[Tensor] = None,
        save_scores: bool = False,
        **kwargs,
    ) -> List[str]:
        # https://huggingface.co/transformers/v4.7.0/main_classes/model.html#generation
        tok = self.tokenizer
        eos, bos = tok.eos_token_id, tok.bos_token_id

        if prompt is not None:
            prompt_ids = self.tokenizer(prompt, add_special_tokens=False).input_ids
        if prompt_ids is not None:
            prompt_ids = [eos, bos] + prompt_ids
            decoder_input_ids = torch.tensor([prompt_ids])
        if multi_prompt_ids is not None:
            assert len(texts) == len(multi_prompt_ids)
            multi_prompt_ids = [[eos, bos] + lst for lst in multi_prompt_ids]
            decoder_input_ids = torch.tensor(multi_prompt_ids)
        if decoder_input_ids is not None:
            kwargs.update(decoder_input_ids=decoder_input_ids.to(self.model.device))
       
        model_inputs = {'input_ids':[],'attention_mask':[]}


        batch_texts = [
                (input_template, " ".join(input_context))
                for input_template, input_context in zip(texts[0], texts[1])
            ]
        model_inputs = self.tokenizer.batch_encode_plus(
            batch_texts,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation='only_second',
            padding='max_length',
            return_tensors="pt"  
        ).to(self.model.device)

        outputs = self.model.generate(
            **model_inputs,
            do_sample=do_sample,
            top_k=top_k,
            temperature=temperature,
            num_return_sequences=num_return,
            return_dict_in_generate=True,
            output_scores=save_scores,
            max_length=self.max_length,
            **kwargs,
        )

        # outputs = self.model.generate(
        #     **self.tokenize(texts),
        #     do_sample=do_sample,
        #     top_k=top_k,
        #     temperature=temperature,
        #     num_return_sequences=num_return,
        #     return_dict_in_generate=True,
        #     output_scores=save_scores,
        #     max_length=self.max_length,
        #     **kwargs,
        # )  
        self.scores = None
        if save_scores:
            self.scores = [_ for _ in torch.stack(outputs.scores, 1).cpu()]
        return self.decode(outputs.sequences)

    def decode(self, outputs) -> List[str]:
        tok = self.tokenizer
        texts = tok.batch_decode(
            outputs, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )

        # Manually remove <bos><eos><pad> in case we have custom special tokens
        special_tokens = [tok.eos_token, tok.bos_token, tok.pad_token]
        for i, t in enumerate(texts):
            for token in special_tokens:
                t = t.replace(token, "")
                texts[i] = t
        return texts



if __name__ == "__main__":
    Fire()
