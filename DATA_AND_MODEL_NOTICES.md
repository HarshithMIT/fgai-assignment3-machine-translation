# Data and model attribution

The parallel sentences come from [ManyThings](https://www.manythings.org/anki/) and
[Tatoeba](https://tatoeba.org/). Every selected JSONL record preserves its original
attribution, sentence IDs and contributor names. The archive's original notice is
retained in [data/SOURCE_LICENSE.txt](data/SOURCE_LICENSE.txt). The attribution
column specifies CC-BY 2.0 (France); the source notice links the applicable terms.
Selection, normalization and deduplication are documented in `prepare_data.py`.

The mT5 base is [google/mt5-small](https://huggingface.co/google/mt5-small),
published under Apache-2.0. The included adapter is a task-specific LoRA adaptation;
the full base weights are downloaded separately. A copy of Apache-2.0 is included
in `checkpoints/mt5-adapter/LICENSE`.

LLaMA weights are obtained from the public
[Unsloth mirror](https://huggingface.co/unsloth/Llama-3.2-1B-Instruct) of
[Meta LLaMA 3.2 1B Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct),
subject to the LLaMA 3.2 Community License and associated use policy.
Gemma weights are obtained from the public
[Unsloth mirror](https://huggingface.co/unsloth/gemma-3-270m-it) of
[Google Gemma 3 270M IT](https://huggingface.co/google/gemma-3-270m-it),
subject to the Gemma terms. These base weights are not redistributed in this repository.
Exact source revisions are in `results/model_sources.json`.

The basic Transformer starts from random weights. Its included checkpoint and
training-only tokenizer are outputs of this experiment. The additional ethical
probes were authored for the experiment and are separate from the public corpus.
