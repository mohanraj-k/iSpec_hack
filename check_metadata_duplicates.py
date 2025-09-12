import json
from collections import Counter

def norm(v):
    # return str(v).strip().lower() if v is not None else ''
    return str(v) if v is not None else ''

keys = [
     'sponsor_name', 'study_id','dq_name', 'dq_description',
    'query_text', 'query_target', 'primary_dataset'
]


with open('data/precomputed_metadata.json') as f:
    data = json.load(f)

sigs = [tuple(norm(rec.get(k)) for k in keys) for rec in data]
# print(sigs[:3])

counts = Counter(sigs)
print(list(counts.items())[:1])
dups = [sig for sig, c in counts.items() if c > 1]

print(f'Duplicate signature count: {len(dups)}')
print(dups)
# print('--- Example duplicate signatures (up to 5) ---')
# for d in dups[:5]:
#     print(d)

# if len(dups) > 0:
#     print('\nSample records for first duplicate:')
#     print(dups)
    # for rec in data:
    #     sig = tuple(norm(rec.get(k)) for k in keys)
    #     if sig == dups[0]:
    #         print(json.dumps(rec, indent=2))
