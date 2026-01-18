from pathlib import Path
from typing import List, Optional, Tuple
import json
import torch
# from fire import Fire
from tqdm import tqdm
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer,
                          IntervalStrategy, Pipeline, TrainingArguments,
                          pipeline, set_seed)
import  re
from encoding import select_encoder
from generation import TextGenerator
from transformer_base import  run_summarization_rl,run_clm_rl
from utils import DynamicModel
from transformers import pipeline, AutoTokenizer
from transformers import TrainingArguments as HFTrainingArguments, IntervalStrategy
from dataclasses import dataclass, field


class BaseEAEModel(DynamicModel):
    model_dir: str
    data_dir: str
    model_name: str
    do_pretrain: bool
    encoder_name: str
    pipe_name: str
    batch_size: int = 8
    grad_accumulation: int = 2
    warmup_ratio: float = 0.2
    lr_pretrain: float = 3e-4
    lr_finetune: float = 3e-5
    epochs_pretrain: int = 3
    epochs_finetune: int = 5
    train_fp16: bool = True
    # random_seed: int

    def fit(self, path_train: str, path_dev: Optional[str] = None):
        raise NotImplementedError

    def run(self, *args, **kwargs):
        raise NotImplementedError

    def decode(self, *args, **kwargs):
        raise NotImplementedError

    def get_lr(self) -> float:
        return self.lr_pretrain if self.do_pretrain else self.lr_finetune

    def get_epochs(self) -> int:
        return self.epochs_pretrain if self.do_pretrain else self.epochs_finetune

    def make_pipe(self, **kwargs) -> Pipeline:
      
        pipe = pipeline(
            self.pipe_name,
            model=self.model_dir,
            tokenizer=self.model_name,
            device=0 if torch.cuda.is_available() else -1,
            **kwargs,
        )
        return pipe

    def get_encoder(self):
        return select_encoder(self.encoder_name)

    def get_train_args(self, do_eval: bool) -> TrainingArguments:

        return TrainingArguments(
            # seed=self.random_seed,
            do_train=True,
            do_eval=do_eval or None,  # False still becomes True after parsing
            overwrite_output_dir=True,
            per_device_train_batch_size=self.batch_size,
            gradient_accumulation_steps=self.grad_accumulation,
            warmup_ratio=self.warmup_ratio,
            output_dir=self.model_dir,
            save_strategy=IntervalStrategy.EPOCH,
            eval_strategy=IntervalStrategy.EPOCH
            if do_eval
            else IntervalStrategy.NO,
            learning_rate=5e-5,
            num_train_epochs=self.get_epochs(),
            load_best_model_at_end=True,
            fp16=False,
            max_grad_norm=1.0,
            fp16_full_eval=False,
            logging_strategy=IntervalStrategy.STEPS,
            logging_steps=100
        )


class EventGenerator(BaseEAEModel):
    model_name: str = "gpt2"
    block_size: int = 512
    encoder_name: str = "gpt_new_generate"
    pipe_name: str = "text-generation"
    lora_dir: Optional[str]  = None

    def fit(self, path_train: str, path_dev: Optional[str] = None):
  
        data_args = run_clm_rl.DataTrainingArguments(
            concat_texts=False,
            train_file=path_train,
            validation_file=path_dev,
            overwrite_cache=True,
            block_size=self.block_size,
        )
        train_args = self.get_train_args(do_eval=path_dev is not None)
        model_args = run_clm_rl.ModelArguments(model_name_or_path=self.model_name,lora_dir = self.lora_dir)
        run_clm_rl.main(
            model_args=model_args, training_args=train_args, data_args=data_args
        )

    def decode(self, *args, **kwargs):
        pass


class BartBaseModel(BaseEAEModel):
    model_name: str = "facebook/bart-base"
    max_source_length: int = 512
    max_target_length: int = 512
    encoder_name: str = "new_generate"
    pipe_name: str = "summarization"

    def fit(self, path_train: str, path_dev: Optional[str] = None):
        kwargs = {}

        data_args = run_summarization_rl.DataTrainingArguments(
            train_file=path_train,
            validation_file=path_dev,
            overwrite_cache=True,
            max_target_length=self.max_target_length,
            max_source_length=self.max_source_length,
            **kwargs,
        )
        train_args = self.get_train_args(do_eval=path_dev is not None)
        kwargs = {
            k: v for k, v in train_args.to_dict().items()  if not k.startswith("_") 
        }
        train_args = run_summarization_rl.Seq2SeqTrainingArguments(**kwargs)
        model_args = run_summarization_rl.ModelArguments(
            model_name_or_path=self.model_name
        )
        run_summarization_rl.main(
            model_args=model_args, training_args=train_args, data_args=data_args
        )

    def load_generator(self, device: torch.device) -> TextGenerator:
        gen = TextGenerator(
            model=AutoModelForSeq2SeqLM.from_pretrained(self.model_dir),
            tokenizer=AutoTokenizer.from_pretrained(self.model_dir),
            max_length=self.max_target_length,
        )
        gen.model = gen.model.to(device)
        return gen

  

    def decode(self, *args, **kwargs):
        pass


class EAE_Extractor(BartBaseModel):
    encoder_name: str = "new_extract"
    @staticmethod
    def gen_texts(texts: List[str], gen: TextGenerator, **kwargs):
        return gen.run(texts, do_sample=False, num_return=1, **kwargs)

    def run(
        self,
        texts: List[str],
        path_out: Path,
        batch_size: int = 512,
        device: torch.device = torch.device("cuda"),
    ):
        # set_seed(self.random_seed)
        encoder = self.get_encoder()
        prompts = [encoder.encode_x(t) for t in texts]
        gen = self.load_generator(device=device)
        preds = []

        for i in tqdm(range(0, len(texts), batch_size)):
            batch = prompts[i : i + batch_size]
            outputs = self.gen_texts(batch, gen)
            preds.extend(outputs)

        path_out.parent.mkdir(exist_ok=True, parents=True)
        with open(path_out, "w") as f:
            for x, y in zip(prompts, preds):
                f.write(run_summarization_rl.encode_to_line(x=x, y=y))



def select_model(name: str, **kwargs) -> BaseEAEModel:
    mapping = dict(
        generate=EventGenerator(**kwargs),
        extractor=EAE_Extractor(**kwargs), 
    )
    model = mapping[name]
    return model

