"""Reload saved weights and exactly replay three translations per selected model."""
import argparse
import json
import torch
from common import ROOT,device,seed_everything
from prepare_data import read_jsonl,save_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',choices=['basic','mt5','llama','gemma'],required=True)
    args=p.parse_args(); name=args.model; seed_everything(); dev=device()
    if name=='basic':
        import sentencepiece as spm
        from basic_transformer import TranslationTransformer,make_generator
        model=TranslationTransformer().to(dev)
        model.load_state_dict(torch.load(ROOT/'checkpoints'/'basic'/'model.pt',map_location=dev,weights_only=True))
        tokenizer=spm.SentencePieceProcessor(model_file=str(ROOT/'checkpoints'/'basic'/'tokenizer.model'))
        generate=make_generator(model,tokenizer,dev)
    else:
        from transformers import AutoTokenizer,AutoModelForCausalLM,AutoModelForSeq2SeqLM
        tokenizer=AutoTokenizer.from_pretrained(ROOT/'checkpoints'/f'{name}-base',use_fast=name!='mt5',local_files_only=True)
        cls=AutoModelForSeq2SeqLM if name=='mt5' else AutoModelForCausalLM
        model=cls.from_pretrained(ROOT/'checkpoints'/f'{name}-base',local_files_only=True,
            torch_dtype=torch.bfloat16 if dev.type=='cuda' else torch.float32,low_cpu_mem_usage=True).to(dev)
        if name=='mt5':
            from peft import PeftModel
            from finetune_mt5 import make_generator
            model=PeftModel.from_pretrained(model,ROOT/'checkpoints'/'mt5-adapter')
            model.config.use_cache=True
            generate=make_generator(model,tokenizer,dev)
        else:
            from prompt_decoder import make_generator
            if tokenizer.pad_token_id is None: tokenizer.pad_token=tokenizer.eos_token
            model.generation_config.do_sample=False; model.generation_config.temperature=1.
            model.generation_config.top_p=1.; model.generation_config.top_k=0
            generate=make_generator(model,tokenizer,read_jsonl(ROOT/'data'/'train.jsonl')[:3],dev)
    saved=json.loads((ROOT/'results'/f'translations_{name}.json').read_text(encoding='utf-8'))
    checks=[]
    for row in saved[:3]:
        replay=generate(row['source'])
        assert replay['text']==row['translation'],(name,row['id'],replay['text'],row['translation'])
        assert replay['tokens']==row['output_tokens']
        checks.append({'id':row['id'],'exact_text_match':True,'exact_token_count_match':True})
    save_json(ROOT/'results'/f'replay_{name}.json',checks)
    print(f'{name}: all three saved translations reproduced exactly.')


if __name__=='__main__': main()
