"""Recompute saved scores and create tables and publication-ready figures."""
import csv
import json
import numpy as np
from common import ROOT,score,save_csv
from prepare_data import read_jsonl,save_json,normalized_key
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MODELS=['basic','mt5','llama','gemma']
LABELS={'basic':'Basic Transformer','mt5':'mT5-small + LoRA','llama':'LLaMA 3.2 1B','gemma':'Gemma 3 270M'}
COLORS=['#526879','#287a8c','#ba7848','#866090']


def load(name): return json.loads((ROOT/'results'/name).read_text(encoding='utf-8'))


def main():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'savefig.dpi':200})
    tests=read_jsonl(ROOT/'data'/'test.jsonl')
    summary=[]; temps=[]; verification={}; combined=[]
    for name in MODELS:
        trans=load(f'translations_{name}.json'); metrics=load(f'metrics_{name}.json'); training=load(f'training_{name}.json')
        assert len(trans)==200
        assert [(r['id'],r['source'],r['reference']) for r in trans]==[(r['id'],r['source'],r['target']) for r in tests]
        recomputed=score([r['translation'] for r in trans],[r['reference'] for r in trans])
        for key in ('bleu','chrf'):
            assert abs(metrics[key]-recomputed[key])<1e-9
        assert abs(metrics['inference_seconds']-sum(r['seconds'] for r in trans))<1e-6
        assert all(r['output_tokens']<=48 and r['seconds']>0 for r in trans)
        row={'model':name,'label':LABELS[name],'bleu':metrics['bleu'],'chrf':metrics['chrf'],
             'parameters':training['parameters'],'base_parameters':training.get('base_parameters',training['parameters']),
             'adapted_parameters':training['trainable_parameters'],'training_seconds':training['training_seconds'],
             'fit_wall_seconds':training['fit_wall_seconds'],'inference_seconds':metrics['inference_seconds'],
             'mean_latency_ms':metrics['mean_latency_seconds']*1000,'length_cap_hits':metrics['length_cap_hits'],
             'normalized_source_copies':sum(normalized_key(r['translation'])==normalized_key(r['source']) for r in trans),
             'normalized_reference_matches':sum(normalized_key(r['translation'])==normalized_key(r['reference']) for r in trans),
             'mean_generated_tokens':float(np.mean([r['output_tokens'] for r in trans]))}
        summary.append(row)
        combined.extend({'model':name,**r} for r in trans)
        sampling=load(f'temperature_{name}.json'); probes=load(f'ethical_{name}.json')
        assert len(sampling)==60 and len(probes)==12
        assert {(r['id'],r['temperature'],r['sampling_seed']) for r in sampling}=={
            (r['id'],t,s) for r in tests[:10] for t in [.3,.7,1.] for s in [101,202]}
        for t in [0.,.3,.7,1.]:
            panel=[r for r in sampling if r['temperature']==t] if t else trans[:10]
            scores=[]
            for s in ([101,202] if t else [42]):
                items=[r for r in panel if r['sampling_seed']==s] if t else panel
                m=score([r['translation'] for r in items],[r['reference'] for r in items])
                scores.append(m)
            varied=sum(len({r['translation'] for r in panel if r['id']==test['id']})>1 for test in tests[:10])
            temps.append({'model':name,'temperature':t,'n_sources':10,'n_seeds':len(scores),
                          'mean_bleu':float(np.mean([s['bleu'] for s in scores])),
                          'bleu_sd':float(np.std([s['bleu'] for s in scores])),
                          'mean_chrf':float(np.mean([s['chrf'] for s in scores])),
                          'chrf_sd':float(np.std([s['chrf'] for s in scores])),
                          'sources_with_different_outputs':varied,
                          'length_cap_hits':sum(r['hit_length_cap'] for r in panel)})
        verification[name]={'test_alignment':True,'recomputed_bleu_chrf':True,'timing_sum':True,
                            'test_records':len(trans),'temperature_records':len(sampling),'ethical_records':len(probes)}
    save_csv(ROOT/'results'/'comparison.csv',summary); save_json(ROOT/'results'/'comparison.json',summary)
    save_csv(ROOT/'results'/'temperature_summary.csv',temps); save_json(ROOT/'results'/'temperature_summary.json',temps)
    save_csv(ROOT/'results'/'all_test_translations.csv',combined)
    save_json(ROOT/'results'/'verification.json',verification)
    x=np.arange(4); labels=[LABELS[m] for m in MODELS]
    fig,axes=plt.subplots(1,2,figsize=(10,3.7),layout='constrained')
    for ax,key,title in zip(axes,['bleu','chrf'],['BLEU (13a, case sensitive)','chrF (character 6-grams, beta=2)']):
        bars=ax.bar(x,[r[key] for r in summary],color=COLORS,width=.62)
        ax.bar_label(bars,fmt='%.2f',padding=3,fontsize=10)
        ax.set_xticks(x,labels,rotation=20,ha='right'); ax.set_ylim(0,max(r[key] for r in summary)*1.2+1)
        ax.set_title(title); ax.set_ylabel('Score / 100'); ax.grid(axis='y',alpha=.18); ax.set_axisbelow(True)
    fig.savefig(ROOT/'results'/'quality.png'); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,3.5),layout='constrained')
    for ax,key,title in zip(axes,['parameters','mean_latency_ms'],['Parameters (millions; includes adapters)','Mean translation latency (ms; batch=1)']):
        values=[r[key]/1e6 if key=='parameters' else r[key] for r in summary]
        bars=ax.bar(x,values,color=COLORS,width=.62); ax.bar_label(bars,fmt='%.1f',padding=3)
        ax.set_xticks(x,labels,rotation=20,ha='right'); ax.set_ylim(0,max(values)*1.22)
        ax.set_title(title); ax.grid(axis='y',alpha=.18); ax.set_axisbelow(True)
    fig.savefig(ROOT/'results'/'compute.png'); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,3.6),layout='constrained')
    for name,color in zip(MODELS,COLORS):
        rows=[r for r in temps if r['model']==name]
        for ax,key,title in zip(axes,['mean_bleu','mean_chrf'],['BLEU on the 10-source panel','chrF on the 10-source panel']):
            ax.plot([r['temperature'] for r in rows],[r[key] for r in rows],marker='o',label=LABELS[name],color=color)
            ax.set_xlabel('Temperature (0 = greedy)'); ax.set_title(title); ax.set_ylabel('Score / 100'); ax.grid(alpha=.2)
    axes[1].legend(fontsize=8); fig.savefig(ROOT/'results'/'temperature.png'); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,3.5),layout='constrained')
    for ax,name,xkey,title in zip(axes,['basic','mt5'],['epoch','step'],['Basic Transformer validation','mT5 LoRA validation']):
        with (ROOT/'results'/f'history_{name}.csv').open() as f: rows=list(csv.DictReader(f))
        ax.plot([int(r[xkey]) for r in rows],[float(r['validation_nll']) for r in rows],marker='o',color='#287a8c')
        ax.set_xlabel(xkey.capitalize()); ax.set_ylabel('Validation NLL / native token'); ax.set_title(title); ax.grid(alpha=.2)
    fig.savefig(ROOT/'results'/'learning_curves.png'); plt.close(fig)
    print(json.dumps(summary,indent=2))


if __name__=='__main__': main()
