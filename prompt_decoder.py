"""Apply the same three-shot translation prompt to pretrained LLaMA and Gemma."""
import argparse
import json
import time
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM,GenerationConfig
from common import ROOT,MAX_NEW_TOKENS,seed_everything,device,sync,benchmark_translate,extra_evaluation
from prepare_data import read_jsonl,save_json


def prompt_text(source,demonstrations):
    instructions=('Translate the French sentence into English. Return only the English translation. '
                  'Preserve names, numbers, negation, and meaning. Do not explain or add information.\n\n')
    for row in demonstrations:
        instructions+=f"French: {row['source']}\nEnglish: {row['target']}\n\n"
    return instructions+f'French: {source}\nEnglish:'


def make_generator(model,tokenizer,demonstrations,dev):
    @torch.no_grad()
    def generate(text,temperature=0,sampling_seed=42):
        messages=[{'role':'user','content':prompt_text(text,demonstrations)}]
        ids=tokenizer.apply_chat_template(messages,add_generation_prompt=True,return_tensors='pt').to(dev)
        options={'max_new_tokens':MAX_NEW_TOKENS,'num_beams':1,'do_sample':temperature>0,
                 'pad_token_id':tokenizer.pad_token_id,'eos_token_id':model.generation_config.eos_token_id,
                 'repetition_penalty':1.0}
        if temperature>0:
            torch.manual_seed(sampling_seed); torch.cuda.manual_seed_all(sampling_seed)
            options.update(temperature=temperature,top_k=0,top_p=1.0)
        generated=model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),**options)[0,ids.shape[1]:]
        stops=model.generation_config.eos_token_id
        if isinstance(stops,int): stops=[stops]
        text=tokenizer.decode(generated,skip_special_tokens=True).strip()
        # No reference-based or semantic cleanup: verbosity is part of performance.
        return {'text':text,'tokens':len(generated),
                'hit_length_cap':len(generated)>=MAX_NEW_TOKENS and generated[-1].item() not in stops}
    return generate


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--model',choices=['llama','gemma'],required=True)
    args=p.parse_args(); name=args.model; seed_everything(); dev=device()
    folder=ROOT/'checkpoints'/f'{name}-base'
    tokenizer=AutoTokenizer.from_pretrained(folder,local_files_only=True)
    if tokenizer.pad_token_id is None: tokenizer.pad_token=tokenizer.eos_token
    model=AutoModelForCausalLM.from_pretrained(folder,local_files_only=True,
        torch_dtype=torch.bfloat16 if dev.type=='cuda' else torch.float32,low_cpu_mem_usage=True).to(dev).eval()
    # Stop IDs come from each model's supplied generation configuration.
    model.generation_config.do_sample=False; model.generation_config.temperature=1.0
    model.generation_config.top_p=1.0; model.generation_config.top_k=0
    demonstrations=read_jsonl(ROOT/'data'/'train.jsonl')[:3]
    save_json(ROOT/'results'/'prompt_demonstrations.json',demonstrations)
    save_json(ROOT/'results'/f'training_{name}.json',{'method':'three-shot prompting; no parameter updates',
        'training_seconds':None,'fit_wall_seconds':None,'parameters':sum(p.numel() for p in model.parameters()),
        'trainable_parameters':0,'precision':'bfloat16' if dev.type=='cuda' else 'float32','device':str(dev),
        'demonstration_ids':[r['id'] for r in demonstrations],
        'prompt_template':prompt_text('{SOURCE_FRENCH}',demonstrations)})
    generate=make_generator(model,tokenizer,demonstrations,dev)
    metrics=benchmark_translate(generate,read_jsonl(ROOT/'data'/'test.jsonl'),name)
    save_json(ROOT/'results'/f'metrics_{name}.json',metrics); print(metrics,flush=True)
    extra_evaluation(generate,name)


if __name__=='__main__': main()
