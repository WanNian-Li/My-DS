'''
合并train_list_DS29.json 和 train_list_small4.json，去除 val_list4.json 中的场景，得到 train_list_merged.json

用法：
    直接运行即可
'''

import json
with open('datalists/train_list_DS29.json') as f:
    a = json.load(f)
with open('datalists/train_list_small4.json') as f:
    b = json.load(f)
with open('datalists/val_list4.json') as f:
    val = json.load(f)

merged = list((set(a) | set(b)) - set(val))
with open('datalists/train_list_merged.json', 'w') as f:
    json.dump(merged, f, indent=2)
print(f'合并后共 {len(merged)} 个场景')