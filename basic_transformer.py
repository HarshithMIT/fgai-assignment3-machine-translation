"""Train a genuine encoder-decoder Transformer from random initialization."""
import argparse
import math
import time
from contextlib import nullcontext
import sentencepiece as spm
import torch
from torch import nn
from torch.utils.data import DataLoader
from common import ROOT,SEED,MAX_NEW_TOKENS,seed_everything,device,sync,save_csv,benchmark_translate,extra_evaluation,record_environment
from prepare_data import read_jsonl,save_json

CONFIG={'vocab_size':2000,'d_model':128,'heads':4,'encoder_layers':2,'decoder_layers':2,
        'ff_dim':512,'dropout':.1,'batch_size':64,'epochs':20,'learning_rate':.001,
        'label_smoothing':.1,'seed':42,'source_max_tokens':64,'target_max_tokens':64,
        'max_new_tokens':48,'tokenizer':'SentencePiece BPE trained only on the training split'}


def portable_tokenizer_metadata(path):
    """Remove machine-specific training paths without changing learned pieces."""
    from sentencepiece import sentencepiece_model_pb2
    proto=sentencepiece_model_pb2.ModelProto()
    proto.ParseFromString(path.read_bytes())
    proto.trainer_spec.input[:]=['checkpoints/basic/tokenizer_training.txt']
    proto.trainer_spec.model_prefix='checkpoints/basic/tokenizer'
    path.write_bytes(proto.SerializeToString())


class PositionalEncoding(nn.Module):
    def __init__(self,d_model,dropout=.1,max_len=256):
        super().__init__(); self.dropout=nn.Dropout(dropout)
        positions=torch.arange(max_len).unsqueeze(1)
        divisor=torch.exp(torch.arange(0,d_model,2)*(-math.log(10000)/d_model))
        pe=torch.zeros(max_len,d_model)
        pe[:,0::2]=torch.sin(positions*divisor); pe[:,1::2]=torch.cos(positions*divisor)
        self.register_buffer('pe',pe.unsqueeze(0))
    def forward(self,x): return self.dropout(x+self.pe[:,:x.shape[1]])


class TranslationTransformer(nn.Module):
    def __init__(self,config=CONFIG):
        super().__init__(); self.config=config
        self.embedding=nn.Embedding(config['vocab_size'],config['d_model'],padding_idx=0)
        self.position=PositionalEncoding(config['d_model'],config['dropout'])
        self.transformer=nn.Transformer(d_model=config['d_model'],nhead=config['heads'],
            num_encoder_layers=config['encoder_layers'],num_decoder_layers=config['decoder_layers'],
            dim_feedforward=config['ff_dim'],dropout=config['dropout'],batch_first=True,norm_first=True)
        self.output=nn.Linear(config['d_model'],config['vocab_size'])
        self.output.weight=self.embedding.weight
        nn.init.normal_(self.embedding.weight,mean=0,std=config['d_model']**-.5)
        with torch.no_grad(): self.embedding.weight[0].zero_()
    def embed(self,x): return self.position(self.embedding(x)*math.sqrt(self.config['d_model']))
    def forward(self,src,tgt):
        causal=torch.triu(torch.ones(tgt.shape[1],tgt.shape[1],device=tgt.device,dtype=torch.bool),diagonal=1)
        hidden=self.transformer(self.embed(src),self.embed(tgt),tgt_mask=causal,
            src_key_padding_mask=src.eq(0),tgt_key_padding_mask=tgt.eq(0),memory_key_padding_mask=src.eq(0))
        return self.output(hidden)


def encoded_dataset(rows,sp):
    dataset=[]
    for r in rows:
        src=sp.encode(r['source'],out_type=int)+[3]
        target=[2]+sp.encode(r['target'],out_type=int)+[3]
        # Do not silently truncate evaluation or training examples.
        if len(src)>64 or len(target)>64: raise ValueError('An example exceeds the documented token limit')
        dataset.append((torch.tensor(src),torch.tensor(target)))
    return dataset


def collate(batch):
    src,tgt=zip(*batch)
    return nn.utils.rnn.pad_sequence(src,batch_first=True,padding_value=0),nn.utils.rnn.pad_sequence(tgt,batch_first=True,padding_value=0)


@torch.no_grad()
def validation_loss(model,loader,dev):
    model.eval(); total=n=0
    for src,tgt in loader:
        src,tgt=src.to(dev),tgt.to(dev)
        logits=model(src,tgt[:,:-1]); labels=tgt[:,1:]
        total+=nn.functional.cross_entropy(logits.reshape(-1,logits.shape[-1]),labels.reshape(-1),ignore_index=0,reduction='sum').item()
        n+=labels.ne(0).sum().item()
    return total/n


def make_generator(model,sp,dev):
    @torch.no_grad()
    def generate(text,temperature=0,sampling_seed=42):
        model.eval()
        src=torch.tensor([sp.encode(text,out_type=int)+[3]],device=dev)
        memory=model.transformer.encoder(model.embed(src),src_key_padding_mask=src.eq(0))
        target=torch.tensor([[2]],device=dev)
        rng=torch.Generator(device=dev).manual_seed(sampling_seed)
        stopped=False
        for _ in range(MAX_NEW_TOKENS):
            causal=torch.triu(torch.ones(target.shape[1],target.shape[1],device=dev,dtype=torch.bool),diagonal=1)
            decoded=model.transformer.decoder(model.embed(target),memory,tgt_mask=causal,memory_key_padding_mask=src.eq(0))
            logits=model.output(decoded[:,-1])[0].float()
            logits[[0,2]]=-torch.inf
            token=(torch.multinomial(torch.softmax(logits/temperature,-1),1,generator=rng).item()
                   if temperature>0 else logits.argmax().item())
            target=torch.cat([target,torch.tensor([[token]],device=dev)],dim=1)
            if token==3: stopped=True; break
        ids=target[0,1:].tolist()
        return {'text':sp.decode([i for i in ids if i not in (0,2,3)]).strip(),'tokens':len(ids),'hit_length_cap':not stopped}
    return generate


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--evaluate-only',action='store_true')
    args=parser.parse_args(); seed_everything(); dev=device(); record_environment()
    folder=ROOT/'checkpoints'/'basic'; folder.mkdir(parents=True,exist_ok=True)
    if not args.evaluate_only:
        rows=read_jsonl(ROOT/'data'/'train.jsonl')
        corpus=folder/'tokenizer_training.txt'
        corpus.write_text('\n'.join(x for r in rows for x in (r['source'],r['target'])),encoding='utf-8')
        spm.SentencePieceTrainer.train(input=str(corpus),model_prefix=str(folder/'tokenizer'),vocab_size=CONFIG['vocab_size'],
            model_type='bpe',character_coverage=1.0,pad_id=0,unk_id=1,bos_id=2,eos_id=3,
            shuffle_input_sentence=False,num_threads=1,minloglevel=2)
        portable_tokenizer_metadata(folder/'tokenizer.model')
        corpus.unlink()
    sp=spm.SentencePieceProcessor(model_file=str(folder/'tokenizer.model'))
    model=TranslationTransformer().to(dev)
    save_json(folder/'config.json',CONFIG)
    if not args.evaluate_only:
        train=DataLoader(encoded_dataset(rows,sp),batch_size=CONFIG['batch_size'],shuffle=True,
            generator=torch.Generator().manual_seed(SEED),collate_fn=collate,num_workers=0)
        val=DataLoader(encoded_dataset(read_jsonl(ROOT/'data'/'validation.jsonl'),sp),batch_size=64,collate_fn=collate)
        opt=torch.optim.AdamW(model.parameters(),lr=CONFIG['learning_rate'],betas=(.9,.98),weight_decay=.01)
        history=[]; best=math.inf; train_seconds=0; best_epoch=0
        sync(); fit_start=time.perf_counter()
        for epoch in range(1,CONFIG['epochs']+1):
            model.train(); total=n=0; sync(); start=time.perf_counter()
            for src,tgt in train:
                src,tgt=src.to(dev),tgt.to(dev); opt.zero_grad(set_to_none=True)
                logits=model(src,tgt[:,:-1]); labels=tgt[:,1:]
                loss=nn.functional.cross_entropy(logits.reshape(-1,CONFIG['vocab_size']),labels.reshape(-1),ignore_index=0,label_smoothing=CONFIG['label_smoothing'])
                if not torch.isfinite(loss): raise RuntimeError('Non-finite basic Transformer loss')
                loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
                count=labels.ne(0).sum().item(); total+=loss.item()*count; n+=count
            sync(); elapsed=time.perf_counter()-start; train_seconds+=elapsed
            vl=validation_loss(model,val,dev)
            if vl<best: best=vl; best_epoch=epoch; torch.save(model.state_dict(),folder/'model.pt')
            history.append({'epoch':epoch,'train_smoothed_loss':total/n,'validation_nll':vl,'train_seconds':elapsed})
            save_csv(ROOT/'results'/'history_basic.csv',history)
            print(f'Basic epoch {epoch}/20 train={total/n:.4f} val={vl:.4f} seconds={elapsed:.2f}',flush=True)
        sync()
        save_json(ROOT/'results'/'training_basic.json',{'method':'trained from random initialization','training_seconds':train_seconds,
            'fit_wall_seconds':time.perf_counter()-fit_start,'epochs':20,'best_epoch':best_epoch,'best_validation_nll':best,
            'parameters':sum(p.numel() for p in model.parameters()),'trainable_parameters':sum(p.numel() for p in model.parameters()),
            'precision':'float32','device':str(dev),'config':CONFIG})
    model.load_state_dict(torch.load(folder/'model.pt',map_location=dev,weights_only=True))
    generate=make_generator(model,sp,dev)
    metrics=benchmark_translate(generate,read_jsonl(ROOT/'data'/'test.jsonl'),'basic')
    save_json(ROOT/'results'/'metrics_basic.json',metrics); print(metrics,flush=True)
    extra_evaluation(generate,'basic')


if __name__=='__main__': main()
