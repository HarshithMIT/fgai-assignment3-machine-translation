"""Download exact public model revisions; do not redistribute base weights."""
import os
import argparse
os.environ.setdefault('HF_HUB_DISABLE_XET','1')
os.environ.setdefault('HF_HUB_DISABLE_SYMLINKS_WARNING','1')
os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT','120')
from huggingface_hub import snapshot_download
from prepare_data import ROOT,save_json

MODELS={
    'mt5':{'repo_id':'google/mt5-small','revision':'73fb5dbe4756edadc8fbe8c769b0a109493acf7a','license':'Apache-2.0'},
    'llama':{'repo_id':'unsloth/Llama-3.2-1B-Instruct','revision':'5a8abab4a5d6f164389b1079fb721cfab8d7126c','license':'Llama 3.2 Community License'},
    'gemma':{'repo_id':'unsloth/gemma-3-270m-it','revision':'23cf460f6bb16954176b3ddcc8d4f250501458a9','license':'Gemma Terms of Use'},
}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models',nargs='+',choices=list(MODELS),default=list(MODELS))
    args=parser.parse_args()
    for key in args.models:
        config=MODELS[key]
        print('Downloading',key,config['repo_id'],flush=True)
        snapshot_download(repo_id=config['repo_id'],revision=config['revision'],local_dir=ROOT/'checkpoints'/f'{key}-base',
                          allow_patterns=['*.json','*.safetensors','pytorch_model.bin','*.model','README.md','LICENSE*','USE_POLICY*'],
                          ignore_patterns=['onnx/*'],max_workers=2)
        print('Ready:',key,flush=True)
    save_json(ROOT/'results'/'model_sources.json',MODELS)
