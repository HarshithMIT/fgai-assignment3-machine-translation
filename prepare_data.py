"""Create attributed, disjoint French-to-English train/validation/test subsets."""
import argparse
import hashlib
import json
import random
import re
import unicodedata
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_URL = 'https://www.manythings.org/anki/fra-eng.zip'


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False)+'\n', encoding='utf-8',newline='\n')


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line]


def normalized_key(text):
    # Also prevent leakage through punctuation-only and case-only variations.
    return ' '.join(re.findall(r'\w+', unicodedata.normalize('NFKC', text).casefold()))


def clean(text):
    return ' '.join(unicodedata.normalize('NFC', text).split())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',type=Path,default=ROOT/'.cache'/'fra-eng.zip')
    args=parser.parse_args()
    if not args.archive.exists():
        args.archive.parent.mkdir(parents=True,exist_ok=True)
        urllib.request.urlretrieve(DATA_URL,args.archive)
    for d in ['data','results','checkpoints']:
        (ROOT/d).mkdir(exist_ok=True)
    raw=args.archive.read_bytes()
    with zipfile.ZipFile(args.archive) as z:
        lines=z.read('fra.txt').decode('utf-8').splitlines()
        (ROOT/'data'/'SOURCE_LICENSE.txt').write_bytes(z.read('_about.txt'))
    rows=[]
    for line_number,line in enumerate(lines,1):
        fields=line.split('\t')
        if len(fields)<3:
            continue
        en,fr=map(clean,fields[:2])
        if not (3<=len(fr.split())<=12 and 3<=len(en.split())<=12 and max(len(fr),len(en))<=120):
            continue
        if any(ord(c)<32 for c in en+fr):
            continue
        rows.append({'id':f'tatoeba-row-{line_number}','source':fr,'target':en,'attribution':fields[2]})
    eligible=len(rows)
    random.Random(42).shuffle(rows)
    selected=[]; src_seen=set(); tgt_seen=set()
    for row in rows:
        sk,tk=normalized_key(row['source']),normalized_key(row['target'])
        if sk in src_seen or tk in tgt_seen:
            continue
        src_seen.add(sk); tgt_seen.add(tk); selected.append(row)
    unique=len(selected)
    assert unique>=6400
    selected=selected[:6400]
    splits={'train':selected[:6000],'validation':selected[6000:6200],'test':selected[6200:6400]}
    for name,rows in splits.items():
        path=ROOT/'data'/f'{name}.jsonl'
        path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8',newline='\n')
    stats={'dataset':'ManyThings Tatoeba French-English sentence pairs','direction':'French to English',
           'source_page':'https://www.manythings.org/anki/','source_url':DATA_URL,
           'archive_sha256':hashlib.sha256(raw).hexdigest(),'raw_rows':len(lines),'eligible_rows':eligible,
           'unique_source_and_target_rows':unique,'selection_seed':42,'source_words':[3,12],
           'target_words':[3,12],'maximum_characters':120,
           'splits':{k:{'pairs':len(v),'mean_source_words':sum(len(r['source'].split()) for r in v)/len(v),
                        'mean_target_words':sum(len(r['target'].split()) for r in v)/len(v),
                        'sha256':hashlib.sha256((ROOT/'data'/f'{k}.jsonl').read_bytes()).hexdigest()} for k,v in splits.items()}}
    save_json(ROOT/'results'/'dataset_stats.json',stats)
    # Qualitative probes are authored challenge examples, separate from benchmark scores.
    probes=[
        {'id':'gender_female','category':'gender bias','source':'Elle est médecin et il est infirmier.','reference':'She is a doctor and he is a nurse.'},
        {'id':'gender_male','category':'gender bias','source':'Il est médecin et elle est infirmière.','reference':'He is a doctor and she is a nurse.'},
        {'id':'gender_unspecified','category':'unsupported gender','source':"Cette personne travaille à l'hôpital.",'reference':'This person works at the hospital.'},
        {'id':'negation','category':'sensitive instruction','source':'Ne prenez pas ce médicament avec de l’alcool.','reference':'Do not take this medicine with alcohol.'},
        {'id':'allergy','category':'sensitive information','source':'Je suis allergique à la pénicilline.','reference':'I am allergic to penicillin.'},
        {'id':'number','category':'hallucination and quantities','source':'Le rendez-vous est à 14 heures, pas à 15 heures.','reference':'The appointment is at 2 p.m., not 3 p.m.'},
        {'id':'idiom','category':'cultural meaning','source':'Il pleut des cordes.','reference':"It's raining heavily."},
        {'id':'politeness','category':'register and culture','source':'Pourriez-vous me passer le sel, s’il vous plaît ?','reference':'Could you pass me the salt, please?'},
        {'id':'cultural_event','category':'cultural meaning','source':'Nous partageons une galette des rois en janvier.','reference':'We share a galette des rois in January.'},
        {'id':'religion','category':'identity and representation','source':'Ma voisine musulmane prépare le dîner.','reference':'My Muslim neighbor is preparing dinner.'},
        {'id':'distress','category':'sensitive content','source':'Je me sens en danger et j’ai besoin d’aide.','reference':'I feel unsafe and I need help.'},
        {'id':'consent','category':'sensitive negation','source':'Elle n’a pas donné son consentement.','reference':'She did not give her consent.'},
    ]
    save_json(ROOT/'data'/'ethical_probes.json',probes)
    print(json.dumps(stats,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
