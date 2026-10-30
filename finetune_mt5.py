"""Fine-tune pretrained mT5-small using LoRA on French-to-English pairs."""
import argparse
import json
import math
import time
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer,AutoModelForSeq2SeqLM,DataCollatorForSeq2Seq
from peft import LoraConfig,TaskType,get_peft_model,PeftModel
from common import ROOT,SEED,MAX_NEW_TOKENS,seed_everything,device,sync,save_csv,benchmark_translate,extra_evaluation
from prepare_data import read_jsonl,save_json

PREFIX='translate French to English: '
CONFIG={'max_steps':800,'micro_batch_size':4,'gradient_accumulation':2,'learning_rate':.001,
        'warmup_steps':50,'evaluation_every':100,'source_max_tokens':64,'target_max_tokens':48,
        'lora_r':8,'lora_alpha':16,'lora_dropout':.05,'target_modules':['q','v','lm_head'],
        'seed':42,'optimizer':'AdamW','weight_decay':.01}


def write_adapter_card(folder,step,val):
    text=f'''---
base_model: google/mt5-small
library_name: peft
license: apache-2.0
tags:
- translation
- lora
---

# French-to-English mT5-small LoRA adapter

This is the validation-selected adapter for the ICT4403 Assignment 3 experiment.
The pretrained base is `google/mt5-small`, revision
`73fb5dbe4756edadc8fbe8c769b0a109493acf7a`; it is downloaded separately.

Adaptation adds rank-{CONFIG['lora_r']} LoRA matrices (alpha {CONFIG['lora_alpha']}, dropout {CONFIG['lora_dropout']}) to
query, value and language-model-head projections. Base weights stay frozen.
The model uses 6,000 attributed ManyThings/Tatoeba French-English training pairs
and 200 separate validation pairs. Selected update: {step}; validation NLL: {val:.6f}.

See [training metadata](../../results/training_mt5.json),
[evaluation](../../results/metrics_mt5.json), and the
[experiment README](../../README.md) for settings, dependencies, results and limitations.
Use `python finetune_mt5.py --evaluate-only` from the repository root after downloading the base.

This adapter has substantial translation errors and is an educational experiment.
Sensitive translations require bilingual human review. The base is Apache-2.0;
its license is included here. The changes consist of the trained LoRA adapters.
'''
    (folder/'README.md').write_text(text,encoding='utf-8',newline='\n')


def encode(rows,tokenizer):
    records=[]
    for row in rows:
        record=tokenizer(PREFIX+row['source'],truncation=False)
        record['labels']=tokenizer(text_target=row['target'],truncation=False)['input_ids']
        if len(record['input_ids'])>64 or len(record['labels'])>48:
            raise ValueError('An mT5 example exceeds the documented token limit')
        records.append(record)
    return records


@torch.no_grad()
def validate(model,loader,dev):
    model.eval(); total=n=0
    for batch in loader:
        batch={k:v.to(dev) for k,v in batch.items()}
        output=model(**batch,use_cache=False)
        count=batch['labels'].ne(-100).sum().item()
        if not torch.isfinite(output.loss): raise RuntimeError('Non-finite validation loss')
        total+=output.loss.item()*count; n+=count
    return total/n


def make_generator(model,tokenizer,dev):
    @torch.no_grad()
    def generate(text,temperature=0,sampling_seed=42):
        model.eval()
        batch=tokenizer(PREFIX+text,return_tensors='pt').to(dev)
        options={'max_new_tokens':MAX_NEW_TOKENS,'num_beams':1,'do_sample':temperature>0}
        if temperature>0:
            torch.manual_seed(sampling_seed); torch.cuda.manual_seed_all(sampling_seed)
            options.update(temperature=temperature,top_k=0,top_p=1.0)
        output=model.generate(**batch,**options)[0]
        generated=output[1:].tolist() # decoder start token excluded
        return {'text':tokenizer.decode(output,skip_special_tokens=True).strip(),'tokens':len(generated),
                'hit_length_cap':len(generated)>=MAX_NEW_TOKENS and generated[-1]!=tokenizer.eos_token_id}
    return generate


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluate-only',action='store_true'); parser.add_argument('--steps',type=int,default=800)
    args=parser.parse_args(); CONFIG['max_steps']=args.steps
    seed_everything(); dev=device(); folder=ROOT/'checkpoints'/'mt5-adapter'; folder.mkdir(parents=True,exist_ok=True)
    tokenizer=AutoTokenizer.from_pretrained(ROOT/'checkpoints'/'mt5-base',use_fast=False,local_files_only=True)
    base=AutoModelForSeq2SeqLM.from_pretrained(ROOT/'checkpoints'/'mt5-base',local_files_only=True,
        torch_dtype=torch.bfloat16 if dev.type=='cuda' else torch.float32,low_cpu_mem_usage=True).to(dev)
    base_parameters=sum(p.numel() for p in base.parameters())
    base.config._name_or_path='google/mt5-small'
    base.config.use_cache=False
    if args.evaluate_only:
        model=PeftModel.from_pretrained(base,folder)
    else:
        model=get_peft_model(base,LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM,r=CONFIG['lora_r'],
            lora_alpha=CONFIG['lora_alpha'],lora_dropout=CONFIG['lora_dropout'],target_modules=CONFIG['target_modules']))
        collator=DataCollatorForSeq2Seq(tokenizer=tokenizer,model=model,padding=True,label_pad_token_id=-100)
        training=DataLoader(encode(read_jsonl(ROOT/'data'/'train.jsonl'),tokenizer),batch_size=CONFIG['micro_batch_size'],
            shuffle=True,generator=torch.Generator().manual_seed(SEED),collate_fn=collator,num_workers=0)
        validation=DataLoader(encode(read_jsonl(ROOT/'data'/'validation.jsonl'),tokenizer),batch_size=4,collate_fn=collator,num_workers=0)
        opt=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=CONFIG['learning_rate'],weight_decay=.01)
        best=math.inf; best_step=0; history=[]; train_seconds=0; examples_seen=0
        iterator=iter(training); interval_loss=interval_count=0
        sync(); fit_start=time.perf_counter()
        for step in range(1,args.steps+1):
            model.train(); sync(); start=time.perf_counter(); opt.zero_grad(set_to_none=True)
            for _ in range(CONFIG['gradient_accumulation']):
                try: batch=next(iterator)
                except StopIteration: iterator=iter(training); batch=next(iterator)
                examples_seen+=len(batch['input_ids']); batch={k:v.to(dev) for k,v in batch.items()}
                output=model(**batch,use_cache=False)
                loss=output.loss
                if not torch.isfinite(loss): raise RuntimeError(f'Non-finite mT5 loss at step {step}')
                (loss/CONFIG['gradient_accumulation']).backward()
                count=batch['labels'].ne(-100).sum().item(); interval_loss+=loss.item()*count; interval_count+=count
            nn.utils.clip_grad_norm_(model.parameters(),1.0)
            factor=min(step/CONFIG['warmup_steps'],1.)*max(.1,1-step/args.steps)
            for group in opt.param_groups: group['lr']=CONFIG['learning_rate']*factor
            opt.step(); sync(); train_seconds+=time.perf_counter()-start
            if step%20==0: print(f'mT5 step {step}/{args.steps} loss={interval_loss/interval_count:.4f} train_s={train_seconds:.1f}',flush=True)
            if step%CONFIG['evaluation_every']==0 or step==args.steps:
                val=validate(model,validation,dev)
                if val<best:
                    best=val; best_step=step
                    model.peft_config['default'].base_model_name_or_path='google/mt5-small'
                    model.peft_config['default'].revision='73fb5dbe4756edadc8fbe8c769b0a109493acf7a'
                    model.save_pretrained(folder,save_embedding_layers=False)
                    tokenizer.save_pretrained(folder)
                    write_adapter_card(folder,step,val)
                history.append({'step':step,'train_nll':interval_loss/interval_count,'validation_nll':val,
                                'cumulative_train_seconds':train_seconds,'examples_seen':examples_seen})
                save_csv(ROOT/'results'/'history_mt5.csv',history)
                print(f'mT5 validation step {step}: {val:.4f}; best step={best_step}',flush=True)
                interval_loss=interval_count=0
        sync()
        save_json(ROOT/'results'/'training_mt5.json',{'method':'LoRA fine-tuning of pretrained mT5-small',
            'training_seconds':train_seconds,'fit_wall_seconds':time.perf_counter()-fit_start,'optimizer_steps':args.steps,
            'best_step':best_step,'best_validation_nll':best,'examples_seen':examples_seen,'base_parameters':base_parameters,
            'parameters':sum(p.numel() for p in model.parameters()),'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
            'precision':'bfloat16 frozen base with float32 LoRA adapters' if dev.type=='cuda' else 'float32',
            'device':str(dev),'config':CONFIG})
        # Load the saved validation-selected adapter before any test evaluation.
        from peft.utils.save_and_load import load_peft_weights,set_peft_model_state_dict
        state=load_peft_weights(folder,device=str(dev)); set_peft_model_state_dict(model,state)
    model.config.use_cache=True
    generate=make_generator(model,tokenizer,dev)
    metrics=benchmark_translate(generate,read_jsonl(ROOT/'data'/'test.jsonl'),'mt5')
    save_json(ROOT/'results'/'metrics_mt5.json',metrics); print(metrics,flush=True)
    extra_evaluation(generate,'mt5')


if __name__=='__main__': main()
