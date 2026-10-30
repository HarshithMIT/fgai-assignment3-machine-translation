"""Checks that protect the validity of the translation experiment."""
import sys
import unittest
import hashlib
import json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from prepare_data import ROOT,read_jsonl,normalized_key
from basic_transformer import TranslationTransformer,CONFIG,collate
from common import score
from prompt_decoder import prompt_text


class ExperimentIntegrity(unittest.TestCase):
    def test_split_sizes_and_no_source_or_target_overlap(self):
        splits=[read_jsonl(ROOT/'data'/f'{s}.jsonl') for s in ('train','validation','test')]
        self.assertEqual([len(s) for s in splits],[6000,200,200])
        stats=json.loads((ROOT/'results'/'dataset_stats.json').read_text(encoding='utf-8'))
        for name in ('train','validation','test'):
            self.assertEqual(hashlib.sha256((ROOT/'data'/f'{name}.jsonl').read_bytes()).hexdigest(),stats['splits'][name]['sha256'])
        for field in ('source','target'):
            keys=[[normalized_key(r[field]) for r in rows] for rows in splits]
            for split in keys: self.assertEqual(len(split),len(set(split)))
            for i in range(3):
                for j in range(i): self.assertFalse(set(keys[i]) & set(keys[j]))
        for rows in splits:
            for row in rows: self.assertIn('CC-BY 2.0',row['attribution'])

    def test_future_decoder_tokens_cannot_change_earlier_logits(self):
        torch.manual_seed(42)
        config=dict(CONFIG,vocab_size=24,d_model=16,heads=2,ff_dim=32,dropout=0)
        model=TranslationTransformer(config).eval()
        src=torch.tensor([[4,5,3,0]])
        before=torch.tensor([[2,7,8,9]])
        after=torch.tensor([[2,7,14,18]])
        with torch.no_grad():
            a,b=model(src,before),model(src,after)
        torch.testing.assert_close(a[:,:2],b[:,:2],rtol=1e-5,atol=1e-5)

    def test_padding_does_not_change_real_source_outputs(self):
        torch.manual_seed(42)
        model=TranslationTransformer(dict(CONFIG,vocab_size=24,d_model=16,heads=2,ff_dim=32,dropout=0)).eval()
        target=torch.tensor([[2,7,8]])
        with torch.no_grad():
            a=model(torch.tensor([[4,5,3]]),target)
            b=model(torch.tensor([[4,5,3,0,0]]),target)
        torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-5)

    def test_teacher_forcing_shift_and_padding(self):
        _,target=collate([(torch.tensor([4,3]),torch.tensor([2,8,9,3])),
                          (torch.tensor([5,6,3]),torch.tensor([2,7,3]))])
        self.assertEqual(target[:,:-1].tolist(),[[2,8,9],[2,7,3]])
        self.assertEqual(target[:,1:].tolist(),[[8,9,3],[7,3,0]])

    def test_metric_perfect_translation_and_bad_translation(self):
        refs=['This is a complete English sentence.','The doctor is working at the hospital.']
        perfect=score(refs,refs)
        self.assertAlmostEqual(perfect['bleu'],100,places=7)
        self.assertAlmostEqual(perfect['chrf'],100,places=7)
        self.assertLess(score(['unrelated','unrelated'],refs)['chrf'],20)

    def test_prompt_demonstrations_use_only_training_rows(self):
        train=read_jsonl(ROOT/'data'/'train.jsonl')[:3]
        test=read_jsonl(ROOT/'data'/'test.jsonl')
        self.assertFalse({r['id'] for r in train}&{r['id'] for r in test})
        text=prompt_text('UNE_SOURCE_UNIQUE',train)
        self.assertTrue(text.endswith('French: UNE_SOURCE_UNIQUE\nEnglish:'))
        self.assertEqual(text.count('French: '),4)


if __name__=='__main__': unittest.main(verbosity=2)
