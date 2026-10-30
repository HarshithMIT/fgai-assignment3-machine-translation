"""Build README, readable sample archives and the report from measured results."""
import json
from pathlib import Path
from xml.sax.saxutils import escape
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,PageBreak,Table,TableStyle,Image

ROOT=Path(__file__).resolve().parent
REPO='https://github.com/HarshithMIT/fgai-assignment3-machine-translation'
DATA='https://www.manythings.org/anki/'
MODELS=['basic','mt5','llama','gemma']
LABELS={'basic':'Basic Transformer','mt5':'mT5-small + LoRA','llama':'LLaMA 3.2 1B','gemma':'Gemma 3 270M'}
REFERENCES=[
    ('ManyThings parallel corpus and download links',DATA),
    ('Tatoeba terms and attribution','https://tatoeba.org/en/terms_of_use'),
    ('Transformer architecture (Vaswani et al., 2017)','https://arxiv.org/abs/1706.03762'),
    ('mT5 paper (Xue et al., 2021)','https://aclanthology.org/2021.naacl-main.41/'),
    ('Official mT5-small model card','https://huggingface.co/google/mt5-small'),
    ('Official LLaMA 3.2 1B Instruct model card','https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct'),
    ('Public LLaMA weights used','https://huggingface.co/unsloth/Llama-3.2-1B-Instruct'),
    ('Official Gemma 3 270M instruction model card','https://huggingface.co/google/gemma-3-270m-it'),
    ('Public Gemma weights used','https://huggingface.co/unsloth/gemma-3-270m-it'),
    ('LoRA paper (Hu et al., 2021)','https://arxiv.org/abs/2106.09685'),
    ('SacreBLEU implementation and metric signatures','https://github.com/mjpost/sacrebleu'),
]

DATASET=(
    'The task is French-to-English translation using the public ManyThings/Tatoeba French-English parallel corpus. '
    'The downloaded archive is the 2026-02-13 release and contains 240,521 aligned sentence pairs. '
    'Sentences are normalized to Unicode NFC and whitespace is collapsed, while case and punctuation are preserved. '
    'Both sides must contain 3-12 whitespace-delimited words and at most 120 characters. This leaves 222,367 eligible pairs. '
    'After a seed-42 shuffle, duplicates are removed globally using separate source and target keys with Unicode NFKC normalization, '
    'case folding and punctuation removal. The remaining 147,965 pairs have unique source and target keys. '
    'The first 6,000, next 200 and following 200 pairs form the training, validation and test splits. '
    'Unused eligible data are excluded to fit the local compute budget.')
DATA_LIMITS=(
    'The splits eliminate normalized exact overlap on both language sides, but not all paraphrases or template similarity. '
    'Short, general-purpose sentences dominate; these data do not represent long documents, specialist domains or every French variety. '
    'There is one English reference per test sentence, so valid alternative translations can receive low automatic scores. '
    'The public pretrained models may have encountered Tatoeba during pretraining; that overlap cannot be ruled out.')
BASIC=(
    'The basic Transformer is implemented with PyTorch encoder and decoder layers and trained from random initialization. '
    'It has two encoder layers, two decoder layers, dimension 128, four attention heads, feed-forward dimension 512, '
    'sinusoidal positions and dropout 0.1. Source/target embeddings and the output projection share weights. '
    'A 2,000-token joint SentencePiece BPE vocabulary is learned from training sentences only. '
    'The encoder reads the source bidirectionally; causal decoder self-attention prevents access to future target tokens, '
    'and cross-attention exposes the source representation. Padding is masked. Training uses shifted teacher forcing, '
    'AdamW at 0.001, label smoothing 0.1, gradient clipping 1.0, batches of 64 and 20 epochs. '
    'The checkpoint with the lowest unsmoothed validation token NLL is selected.')
MT5=(
    'mT5-small starts from multilingual pretrained encoder-decoder weights. Low-rank adaptation (LoRA) fine-tunes '
    'query/value attention projections and the language-model head, with rank 8, alpha 16 and dropout 0.05. '
    'The 300,176,768 base parameters are frozen and 2,349,056 adapter parameters are updated. '
    'The native tokenizer and the prefix "translate French to English: " are used. '
    'AdamW uses learning rate 0.001, 50 warm-up steps, subsequent decay, micro-batches of four and two accumulation steps. '
    'The fixed budget is 800 optimizer updates (6,400 sentence presentations, about 1.07 passes). '
    'Validation occurs every 100 updates; the lowest validation NLL selects the adapter. '
    'The frozen base uses bfloat16 on the GPU and adapters use float32. No quantization is used.')
DECODERS=(
    'LLaMA 3.2 1B Instruct and Gemma 3 270M IT are pretrained instruction-following decoder-only models, '
    'loaded from publicly accessible Unsloth mirrors at pinned revisions. Each uses its native tokenizer and chat template. '
    'Both receive the same instruction and the first three training pairs as demonstrations. '
    'The prompt requests only an English translation and preservation of names, numbers, negation and meaning. '
    'No test reference appears in the prompt, and no parameters are updated. Both use bfloat16 without quantization. '
    'A decoder-only model attends causally to the French prompt, demonstrations and generated target tokens within one sequence; '
    'it has no separate source encoder or encoder-decoder cross-attention module.')
PROTOCOL=(
    'All four models translate the same 200 test sources in the same order. The main run uses greedy decoding, '
    'one beam, batch size one and at most 48 newly generated native tokens, with each model\'s normal end token(s). '
    'Only special-token removal and surrounding-whitespace stripping are applied; explanations and repetitions are retained. '
    'An equal native-token cap is only approximately comparable because tokenizers differ. '
    'BLEU is corpus-level SacreBLEU with 13a tokenization, mixed case and exponential smoothing. '
    'chrF uses character order 6, word order 0 and beta 2. Both range from 0 to 100; higher is better. '
    'These reference-overlap scores do not directly measure factual fidelity, bias or safety.')
TIMING=(
    'Inference time includes tokenization, generation and decoding, measured sequentially with GPU synchronization; '
    'three warm-up translations and model loading are excluded. The table reports the sum of per-sentence times. '
    'Training time counts optimization loops; fit wall time also includes validation and checkpoint writes, but excludes '
    'model downloads, initial loading, data preparation and tokenization. Prompted models have N/A training time, '
    'not zero total cost: their pretraining compute is unknown and excluded for all pretrained models. '
    'Parameter counts are unique PyTorch parameter elements, with adapter parameters included for mT5.')
REPRO=(
    'Python 3.12.4; PyTorch 2.6.0+cu124; Transformers 4.57.1; PEFT 0.17.1; SacreBLEU 2.5.1; '
    'NVIDIA GeForce RTX 3050 Laptop GPU with 4 GiB VRAM; Windows; four PyTorch CPU threads. '
    'Random seeds are fixed. PyTorch warns that the memory-efficient attention backward kernel may be nondeterministic, '
    'so an independent retraining run may vary slightly. Saved-checkpoint inference and saved-output score recomputation '
    'are checked independently. Timings are one local run and may vary with device load and thermal conditions.')
TEMP_METHOD=(
    'A secondary panel uses the first 10 fixed test sources at temperatures 0.3, 0.7 and 1.0 with seeds 101 and 202: '
    '60 sampled translations per model. The greedy baseline is recomputed on those same 10 sources. '
    'Sampling divides logits by temperature and samples from the full allowed vocabulary (top-k disabled, top-p 1.0). '
    'The 48-token cap is unchanged. The reported sampled score is the mean of two corpus scores, one per seed, '
    'rather than a score from duplicated references pooled together. Two seeds and 10 sources are exploratory, '
    'not sufficient for a reliable estimate of population-level differences.')
RUN='''python -m venv .venv
.venv\\Scripts\\activate
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python download_models.py
python basic_transformer.py
python finetune_mt5.py
python prompt_decoder.py --model llama
python prompt_decoder.py --model gemma
python analyze_results.py
python verify_replay.py --model basic
python verify_replay.py --model mt5
python verify_replay.py --model llama
python verify_replay.py --model gemma'''


def load(name): return json.loads((ROOT/'results'/name).read_text(encoding='utf-8'))
def md_table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(str(x).replace('|','\\|').replace('\n',' ') for x in row)+' |' for row in rows])


def main():
    rows=load('comparison.json'); temps=load('temperature_summary.json'); notes=load('analysis_notes.json')
    train={m:load(f'training_{m}.json') for m in MODELS}
    translations={m:load(f'translations_{m}.json') for m in MODELS}
    ethics={m:load(f'ethical_{m}.json') for m in MODELS}
    table_rows=[[LABELS[r['model']],f"{r['bleu']:.2f}",f"{r['chrf']:.2f}",f"{r['parameters']/1e6:.3f}",
                 f"{r['training_seconds']:.2f}" if r['training_seconds'] is not None else 'N/A',f"{r['inference_seconds']:.2f}"] for r in rows]
    headers=['Model','BLEU','chrF','Params (M)','Train (s)','Test (s)']
    temp_rows=[]
    for m in MODELS:
        selected=[r for r in temps if r['model']==m]
        temp_rows.append([LABELS[m]]+[f"{r['mean_bleu']:.2f} / {r['mean_chrf']:.2f}" for r in selected])
    temp_headers=['Model','Greedy','T=0.3','T=0.7','T=1.0']
    archive=['# Complete ethical-probe translations','',
             'Twelve authored diagnostic prompts, separate from the 200-sentence benchmark. These examples do not constitute a demographic fairness audit.','']
    for i,probe in enumerate(ethics['basic']):
        archive.extend([f"## {probe['id']}: {probe['category']}",'',f"French: {probe['source']}",'',f"Expected meaning: {probe['reference']}",''])
        for m in MODELS: archive.append(f"- **{LABELS[m]}:** {ethics[m][i]['translation']}")
        archive.extend(['',notes['ethical_notes'][probe['id']],''])
    (ROOT/'results'/'ethical_review.md').write_text('\n'.join(archive),encoding='utf-8')
    sample=['# Fixed test examples','', 'The first six test rows, selected by split order rather than by output quality. All 800 benchmark outputs are in all_test_translations.csv.','']
    for i in range(6):
        r=translations['basic'][i]; sample.extend([f"## {r['id']}",'',f"French: {r['source']}",'',f"Reference: {r['reference']}",''])
        sample.extend(f"- **{LABELS[m]}:** {translations[m][i]['translation']}" for m in MODELS); sample.append('')
    (ROOT/'results'/'sample_translations.md').write_text('\n'.join(sample),encoding='utf-8')
    readme=f'''# French-to-English machine translation

ICT4403 Foundations of Generative AI - Assignment 3 | Harshith Velkumar

## Objective

Implement, adapt and compare a basic Transformer, mT5, LLaMA and Gemma on one parallel corpus, measuring translation quality and compute while examining responsible-AI concerns.

{notes['conclusion']}

[Full experimental report](Assignment_Report.pdf) · [All test translations](results/all_test_translations.csv) · [Ethical review](results/ethical_review.md)

## Dataset and preprocessing

{DATASET}

Source: [ManyThings/Tatoeba]({DATA}); [French-English archive](https://www.manythings.org/anki/fra-eng.zip). Each selected JSONL row retains the original Tatoeba attribution and sentence IDs. [SOURCE_LICENSE.txt](data/SOURCE_LICENSE.txt) retains the archive's notices. The row labels specify CC-BY 2.0 (France); respect the underlying attribution terms. Do not treat authorship credit as model-generated content.

{DATA_LIMITS}

Exact selected data are checked into `data/`. Archive and split SHA-256 hashes are in [dataset_stats.json](results/dataset_stats.json). Running `prepare_data.py` with a future live archive can produce a different split: use the supplied JSONL files for this experiment. To regenerate from the exact original archive, run `python prepare_data.py --archive /path/to/fra-eng.zip` and verify the recorded archive hash first.

## Models and adaptation

### Basic Transformer

{BASIC}

### mT5

{MT5}

### LLaMA and Gemma

{DECODERS}

The prompt and demonstration IDs are saved in `results/training_llama.json`, `results/training_gemma.json` and `results/prompt_demonstrations.json`. Exact download repositories and revisions are in [model_sources.json](results/model_sources.json). Base weights are not redistributed; the repository contains the basic checkpoint and the mT5 adapter. The model cards and their licenses/terms remain applicable.

## Evaluation protocol

{PROTOCOL}

{TIMING}

{REPRO}

Gemma's first timing run crossed a long session interruption. Its entire evaluation was repeated with identical settings; the completed rerun supplies the reported timing. The translations and accuracy scores matched the initial run. See `results/execution_notes.json`.

## Results

{md_table(headers,table_rows)}

The last column is total inference time for 200 sources. Trainable parameters: basic {train['basic']['trainable_parameters']:,}; mT5 {train['mt5']['trainable_parameters']:,}; prompted LLaMA and Gemma 0 updated parameters. mT5 base alone has {train['mt5']['base_parameters']:,} parameters. Fit wall times: basic {train['basic']['fit_wall_seconds']:.2f} s; mT5 {train['mt5']['fit_wall_seconds']:.2f} s. Selected checkpoints: basic epoch {train['basic']['best_epoch']}; mT5 update {train['mt5']['best_step']}. Length-cap hits in the main test: {', '.join(LABELS[r['model']]+' '+str(r['length_cap_hits'])+'/200' for r in rows)}.

![Translation quality](results/quality.png)

![Compute comparison](results/compute.png)

## Model and architecture comparison

{notes['architecture']}

![Validation histories](results/learning_curves.png)

Validation NLL is useful for selecting checkpoints within a model. Native vocabularies and tokenization differ, so cross-model token NLL values are not directly comparable. No test metric was used to select the saved trained checkpoint.

## Sampling-temperature analysis

{TEMP_METHOD}

Cells below are **BLEU / chrF**. Standard deviations and output-variation counts are in [temperature_summary.csv](results/temperature_summary.csv); the complete 240 sampled outputs are retained in `results/temperature_*.json`.

{md_table(temp_headers,temp_rows)}

![Temperature experiment](results/temperature.png)

{notes['temperature']}

## Qualitative translation review

{notes['qualitative']}

See [fixed first-six examples](results/sample_translations.md). Outputs are not edited to resemble references.

## Responsible and ethical AI

Twelve additional authored probes cover explicit and unspecified gender, a drug warning, a named allergy, appointment times, an idiom, politeness, a cultural food, religion, distress and consent. These probes are separate from the benchmark and do not affect BLEU/chrF ranking. [All 48 outputs and case-specific commentary](results/ethical_review.md) are available.

{notes['ethics']}

{notes['mitigations']}

## Limitations and conclusion

{notes['limitations']}

{notes['conclusion']}

## Reproduce the experiments

The commands below were tested on Windows with Python 3.12 and the NVIDIA CUDA 12.4 wheel. Run from the repository root. On Linux/macOS activate the environment with `source .venv/bin/activate`; choose a PyTorch wheel appropriate for that hardware. CPU fallback is supported, but performance and timing will differ. Downloading the base models needs several GB of disk space and network access. Run model jobs sequentially on a 4 GiB GPU.

```powershell
{RUN}
```

For saved-checkpoint evaluation without retraining: `python basic_transformer.py --evaluate-only` and `python finetune_mt5.py --evaluate-only`. These regenerate output/timing files. To check submitted results without changing their timings, run `python analyze_results.py` and the replay commands instead. The six unit tests check disjoint data splits and attribution, causal masking, padding invariance, teacher-forcing alignment, scoring sanity and training-only prompt demonstrations.

To regenerate this report: install `requirements-report.txt`, then run `python build_report.py`. The authoring environment used the supplied document runtime; Arial fonts are preferred, with a ReportLab font fallback. Training logs, model settings, environment versions and numerical verification records are stored under `results/`.

## Repository guide

| Path | Purpose |
| --- | --- |
| `prepare_data.py`, `data/` | Reproducible split selection, attributed parallel data, extra probes |
| `download_models.py` | Public model downloads at immutable revisions |
| `basic_transformer.py` | Train and evaluate the Transformer from random weights |
| `finetune_mt5.py` | LoRA fine-tuning and evaluation of mT5-small |
| `prompt_decoder.py` | Matched few-shot LLaMA/Gemma prompts and evaluation |
| `common.py` | Seeds, metrics, timing and temperature panel |
| `analyze_results.py`, `verify_replay.py`, `tests/` | Numerical, reload and implementation checks |
| `checkpoints/basic/`, `checkpoints/mt5-adapter/` | Validation-selected reusable checkpoints |
| `results/` | Complete outputs, settings, timings, plots, histories and analysis |
| `build_report.py`, `Assignment_Report.pdf` | Report generator and readable report |

## References

'''+ '\n'.join(f'- [{title}]({url})' for title,url in REFERENCES)+'\n'
    (ROOT/'README.md').write_text(readme,encoding='utf-8')

    # PDF layout: fixed section boundaries and readable vector text.
    regular=Path('C:/Windows/Fonts/arial.ttf'); bold=Path('C:/Windows/Fonts/arialbd.ttf')
    font='Helvetica'; font_bold='Helvetica-Bold'
    if regular.exists() and bold.exists():
        pdfmetrics.registerFont(TTFont('ReportArial',str(regular))); pdfmetrics.registerFont(TTFont('ReportArialBold',str(bold)))
        pdfmetrics.registerFontFamily('ReportArial',normal='ReportArial',bold='ReportArialBold',italic='ReportArial',boldItalic='ReportArialBold')
        font='ReportArial'; font_bold='ReportArialBold'
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle('ReportTitle',fontName=font_bold,fontSize=23,leading=28,spaceAfter=13,textColor=colors.black))
    styles.add(ParagraphStyle('ReportHeading',fontName=font_bold,fontSize=16,leading=20,spaceBefore=10,spaceAfter=10,keepWithNext=True,textColor=colors.black))
    styles.add(ParagraphStyle('ReportBody',fontName=font,fontSize=10.5,leading=14.5,spaceAfter=9))
    styles.add(ParagraphStyle('ReportSmall',fontName=font,fontSize=9,leading=12,spaceAfter=7))
    styles.add(ParagraphStyle('ReportCell',fontName=font,fontSize=8.7,leading=11.5))
    styles.add(ParagraphStyle('ReportHeader',fontName=font_bold,fontSize=8.7,leading=11.5,textColor=colors.white))
    story=[]
    def p(text,style='ReportBody'): story.append(Paragraph(escape(text).replace('\n','<br/>'),styles[style]))
    def h(text): p(text,'ReportHeading')
    def page(): story.append(PageBreak())
    def fig(name,width=510):
        from PIL import Image as PILImage
        with PILImage.open(ROOT/'results'/name) as im: height=width*im.height/im.width
        story.append(Image(str(ROOT/'results'/name),width=width,height=height)); story.append(Spacer(1,7))
    def table(header,body,widths):
        cells=[[Paragraph(escape(str(x)),styles['ReportHeader']) for x in header]]+[[Paragraph(escape(str(x)),styles['ReportCell']) for x in row] for row in body]
        t=Table(cells,colWidths=widths,repeatRows=1,hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#284956')),
                              ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f2f5f6')]),
                              ('GRID',(0,0),(-1,-1),.45,colors.HexColor('#D9D9D9')),
                              ('VALIGN',(0,0),(-1,-1),'MIDDLE'),('LEFTPADDING',(0,0),(-1,-1),7),
                              ('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7)]))
        story.append(t); story.append(Spacer(1,10))
    p('French-to-English\nMachine Translation','ReportTitle')
    p('ICT4403 | Foundations of Generative AI | Assignment 3\nHarshith Velkumar','ReportSmall')
    p('Four-model comparison: basic Transformer, mT5, LLaMA and Gemma.')
    p(notes['conclusion'])
    h('Dataset and preprocessing'); p(DATASET); p(DATA_LIMITS)
    table(['Split','Pairs','Mean French words','Mean English words'],[
        ['Training','6,000','6.47','5.97'],['Validation','200','6.41','5.89'],['Test','200','6.38','5.99']],[165,80,132,133])
    p('Source: '+DATA+' | French-English archive, released 13 February 2026. Each selected row preserves its original Tatoeba authorship and sentence IDs; the original archive notice and hashes accompany the data.','ReportSmall')
    p('Repository: '+REPO,'ReportSmall')
    page(); h('Model implementation and adaptation')
    p(BASIC); p(MT5); p(DECODERS)
    h('Controlled evaluation'); p(PROTOCOL)
    p('Main-test decoding cap reached: '+', '.join(LABELS[r['model']]+f" {r['length_cap_hits']}/200" for r in rows)+'.','ReportSmall')
    page(); h('Measured translation quality and compute')
    table(headers,table_rows,[145,55,55,75,85,95])
    p('Parameters are millions, including mT5 adapters. Train time is optimization-loop time; test time covers all 200 sources. Prompted models receive no parameter updates, so training time is N/A.','ReportSmall')
    fig('quality.png'); fig('compute.png')
    p(TIMING,'ReportSmall')
    page(); h('Architecture comparison and training behavior')
    p(notes['architecture'])
    fig('learning_curves.png')
    p(f"The selected basic checkpoint is epoch {train['basic']['best_epoch']}/20 (validation NLL {train['basic']['best_validation_nll']:.4f}); mT5 selects update {train['mt5']['best_step']}/800 (NLL {train['mt5']['best_validation_nll']:.4f}). Fit wall time is {train['basic']['fit_wall_seconds']:.2f} s and {train['mt5']['fit_wall_seconds']:.2f} s, respectively. mT5 updates only {100*train['mt5']['trainable_parameters']/train['mt5']['parameters']:.2f}% of its total parameters.")
    p('Token NLL cannot be compared directly across different vocabularies. Validation selects checkpoints within each trained model; automatic test scores are then calculated once for that selected checkpoint. No checkpoint was selected by test BLEU or chrF.')
    p(REPRO)
    page(); h('Sampling-temperature analysis')
    p(TEMP_METHOD)
    table(temp_headers,temp_rows,[142,92,92,92,92])
    p('Each cell is BLEU / chrF. Scores use only the 10-source temperature panel. Sampled cells average the two seed-specific corpus scores; greedy has one deterministic output per source.','ReportSmall')
    fig('temperature.png'); p(notes['temperature'])
    p('Complete sampled texts, per-seed settings, length-cap flags, seed variation counts and score standard deviations are included in the repository. Temperature is an exploratory control here, not a setting tuned to optimize the main test results.','ReportSmall')
    page(); h('Qualitative translation examples')
    p('The following are the first two test rows in the fixed split order; they were not chosen for unusually good outputs. A first-six panel and all 800 benchmark translations are available in the repository.','ReportSmall')
    for i in range(2):
        r=translations['basic'][i]
        p('French: '+r['source']); p('Reference: '+r['reference'],'ReportSmall')
        for m in MODELS: p(LABELS[m]+': '+translations[m][i]['translation'],'ReportSmall')
        story.append(Spacer(1,6))
    p(notes['qualitative'])
    h('Responsible-AI review design')
    p('Twelve additional authored probes test explicit and unspecified gender, medical wording, numbers, cultural meaning, politeness, identity, distress and consent. The 48 outputs are separate from benchmark scoring. This is a targeted error review, not a representative fairness or safety certification.')
    p(notes['ethics'])
    page(); h('Sensitive meaning, culture and bias')
    for probe_id in ['number','allergy','idiom']:
        index=next(i for i,r in enumerate(ethics['basic']) if r['id']==probe_id)
        r=ethics['basic'][index]
        p('French: '+r['source']); p('Expected meaning: '+r['reference'],'ReportSmall')
        for m in MODELS: p(LABELS[m]+': '+ethics[m][index]['translation'],'ReportSmall')
        p(notes['ethical_notes'][probe_id],'ReportSmall'); story.append(Spacer(1,5))
    p(notes['mitigations'])
    page(); h('Limitations and conclusion')
    p(notes['limitations']); p(notes['conclusion'])
    h('Reproduction and verification')
    p('Install the pinned dependencies, download the exact public model revisions, and run the four model scripts followed by analyze_results.py. The README provides complete commands. Six integrity tests cover data separation, causal masking, padding invariance, teacher-forcing alignment, scoring sanity and prompt separation. Saved output scores and all test IDs are recomputed and checked. Three benchmark translations per model are replayed after reloading its saved weights.','ReportSmall')
    p('The repository includes selected data with attribution, source links and hashes, basic-model weights, mT5 adapters, hyperparameters, prompt demonstrations, training histories, complete outputs, timing records, environment versions, plots and verification results. Large pretrained weights are downloaded separately under their original terms.','ReportSmall')
    h('References')
    for title,url in REFERENCES:
        story.append(Paragraph(escape(title)+': <link href="'+escape(url)+'" color="#174f79">'+escape(url)+'</link>',styles['ReportSmall']))
    def footer(canvas,document):
        canvas.setFont(font,8); canvas.setFillColor(colors.HexColor('#555555'))
        canvas.drawString(42,26,'ICT4403 Assignment 3 | Harshith Velkumar'); canvas.drawRightString(A4[0]-42,26,str(document.page))
    SimpleDocTemplate(str(ROOT/'Assignment_Report.pdf'),pagesize=A4,leftMargin=42,rightMargin=42,topMargin=35,bottomMargin=43,
                      title='French-to-English Machine Translation: Four-Model Comparison',author='Harshith Velkumar').build(story,onFirstPage=footer,onLaterPages=footer)
    print('Created README, sample archives and Assignment_Report.pdf')


if __name__=='__main__': main()
