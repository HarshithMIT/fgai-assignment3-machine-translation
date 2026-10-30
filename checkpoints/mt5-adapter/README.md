---
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

Adaptation adds rank-8 LoRA matrices (alpha 16, dropout 0.05) to
query, value and language-model-head projections. Base weights stay frozen.
The model uses 6,000 attributed ManyThings/Tatoeba French-English training pairs
and 200 separate validation pairs. Selected update: 800; validation NLL: 1.985527.

See [training metadata](../../results/training_mt5.json),
[evaluation](../../results/metrics_mt5.json), and the
[experiment README](../../README.md) for settings, dependencies, results and limitations.
Use `python finetune_mt5.py --evaluate-only` from the repository root after downloading the base.

This adapter has substantial translation errors and is an educational experiment.
Sensitive translations require bilingual human review. The base is Apache-2.0;
its license is included here. The changes consist of the trained LoRA adapters.
