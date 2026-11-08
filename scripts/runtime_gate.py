#!/usr/bin/env python3
"""Bounded API, deterministic-answer and GDN neighboring-length upgrade gates."""
import argparse
import json
import math
from pathlib import Path
import urllib.request
import urllib.error


def call(path, body):
    req = urllib.request.Request('http://127.0.0.1:19622' + path,
        data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    assert not a.out.exists()
    report = {'checks': []}
    def check(name, passed, evidence):
        report['checks'].append(dict(name=name, passed=bool(passed), evidence=evidence))
        a.out.write_text(json.dumps(report, indent=2) + '\n')
        assert passed, name
    cases = [('exact', 'Return exactly OK and nothing else.', 'OK'),
             ('copy', 'Copy this phrase exactly, without quotes: satin cobalt orbit', 'satin cobalt orbit'),
             ('arithmetic', 'What is 17 + 43? Reply only with the integer.', '60'),
             ('fact', 'What is the capital of France? Reply only with the city.', 'Paris'),
             ('logic', 'All cedars are trees. All trees are plants. Are all cedars plants? Reply only YES or NO.', 'YES'),
             ('python', 'What does len(set([1, 2, 2, 3])) evaluate to in Python? Reply only with the integer.', '3')]
    common = dict(model='qwen-3.8-27b', temperature=0, seed=20260609, reasoning_effort='none', max_tokens=128)
    for name, prompt, expected in cases:
        result = call('/v1/chat/completions', dict(common, messages=[dict(role='user', content=prompt)]))
        c = result['choices'][0]
        check(name, c['finish_reason'] == 'stop' and c['message']['content'].strip() == expected, c)
    result = call('/v1/chat/completions', dict(common, messages=[dict(role='user', content='Return a JSON object with integer field answer equal to 42.')], response_format={'type':'json_object'}))
    c = result['choices'][0]
    check('json', c['finish_reason']=='stop' and json.loads(c['message']['content']) == {'answer':42}, c)
    outputs=[]
    for _ in range(4):
        r=call('/v1/chat/completions',dict(common,messages=[dict(role='user',content=cases[1][1])]))
        outputs.append(r['choices'][0]['message']['content'])
    check('repeat-stability',len(set(outputs))==1,outputs)
    for effort in ('off','minimal','high','max'):
        try:
            call('/v1/chat/completions',dict(common,reasoning_effort=effort,messages=[dict(role='user',content='hello')]))
            valid=False
        except urllib.error.HTTPError as e:
            valid=e.code in (400,422)
        check('reject-'+effort,valid,effort)
    for n in (*range(1,129),133,197,261):
        r=call('/v1/completions',dict(model='qwen-3.8-27b',prompt=[1000]*n,max_tokens=1,temperature=0,logprobs=1,ignore_eos=True,cache_salt='upgrade-gate-'+str(n)))
        c=r['choices'][0]; values=c['logprobs']['token_logprobs']
        check('finite-length-'+str(n),r['usage']['prompt_tokens']==n and r['usage']['completion_tokens']==1 and all(v is not None and math.isfinite(v) for v in values),{'usage':r['usage'],'logprobs':values})
    report['passed']=all(c['passed'] for c in report['checks'])
    a.out.write_text(json.dumps(report,indent=2)+'\n')
    print('Passed',len(report['checks']),'checks')

if __name__ == '__main__':
    main()
