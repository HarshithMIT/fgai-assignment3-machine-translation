"""Shared experiment settings, reproducibility, scoring and timing helpers."""
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache'/'matplotlib'))
os.environ.setdefault('HF_HOME',str(ROOT/'.cache'/'huggingface'))
os.environ.setdefault('HF_HUB_DISABLE_XET','1')
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import csv
import json
import random
import time
import numpy as np
import torch
from sacrebleu.metrics import BLEU,CHRF
from prepare_data import save_json,read_jsonl

SEED=42
MAX_NEW_TOKENS=48


def seed_everything(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True,warn_only=True)


def device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def sync():
    if torch.cuda.is_available(): torch.cuda.synchronize()


def score(hypotheses,references):
    assert len(hypotheses)==len(references)>0
    bleu=BLEU(tokenize='13a',lowercase=False,effective_order=False)
    chrf=CHRF(char_order=6,word_order=0,beta=2)
    b=bleu.corpus_score(hypotheses,[references]); c=chrf.corpus_score(hypotheses,[references])
    return {'bleu':b.score,'chrf':c.score,'bleu_signature':str(bleu.get_signature()),
            'chrf_signature':str(chrf.get_signature()),'bleu_bp':b.bp,'bleu_ratio':b.sys_len/b.ref_len}


def save_csv(path,rows):
    with Path(path).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def benchmark_translate(generate, rows, name):
    """Warm-up excluded; batch-one tokenization + generation + decoding included."""
    for row in rows[:3]: generate(row['source'])
    records=[]; sync(); all_start=time.perf_counter()
    for i,row in enumerate(rows):
        sync(); start=time.perf_counter()
        result=generate(row['source']); sync()
        records.append({'id':row['id'],'source':row['source'],'reference':row['target'],
                        'translation':result['text'],'output_tokens':result['tokens'],
                        'hit_length_cap':result['hit_length_cap'],'seconds':time.perf_counter()-start})
        if (i+1)%25==0: print(f'{name}: translated {i+1}/{len(rows)}',flush=True)
    sync(); wall=time.perf_counter()-all_start
    save_json(ROOT/'results'/f'translations_{name}.json',records)
    metrics=score([r['translation'] for r in records],[r['reference'] for r in records])
    metrics.update({'inference_seconds':sum(r['seconds'] for r in records),'inference_wall_seconds':wall,
                    'mean_latency_seconds':float(np.mean([r['seconds'] for r in records])),
                    'median_latency_seconds':float(np.median([r['seconds'] for r in records])),
                    'test_sentences':len(rows),'length_cap_hits':sum(r['hit_length_cap'] for r in records)})
    return metrics


def extra_evaluation(generate,name):
    test=read_jsonl(ROOT/'data'/'test.jsonl')
    # Small matched secondary panel; fixed before seeing generated outputs.
    sampled=[]
    for temperature in [.3,.7,1.0]:
        for sample_seed in [101,202]:
            for row in test[:10]:
                output=generate(row['source'],temperature=temperature,sampling_seed=sample_seed)
                sampled.append({'id':row['id'],'temperature':temperature,'sampling_seed':sample_seed,
                                'source':row['source'],'reference':row['target'],'translation':output['text'],
                                'output_tokens':output['tokens'],'hit_length_cap':output['hit_length_cap']})
    save_json(ROOT/'results'/f'temperature_{name}.json',sampled)
    probes=json.loads((ROOT/'data'/'ethical_probes.json').read_text(encoding='utf-8'))
    for row in probes:
        result=generate(row['source']); row['translation']=result['text']; row['hit_length_cap']=result['hit_length_cap']
    save_json(ROOT/'results'/f'ethical_{name}.json',probes)


def record_environment():
    import platform,transformers,peft,psutil,sacrebleu
    save_json(ROOT/'results'/'environment.json',{'python':platform.python_version(),'torch':torch.__version__,
        'transformers':transformers.__version__,'peft':peft.__version__,'sacrebleu':sacrebleu.__version__,
        'platform':platform.platform(),'cpu':platform.processor(),'cpu_threads':4,
        'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'gpu_memory_bytes':torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0,
        'ram_bytes':psutil.virtual_memory().total,'cuda_version':torch.version.cuda})
