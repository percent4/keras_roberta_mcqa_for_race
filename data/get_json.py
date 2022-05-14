# -*- coding: utf-8 -*-
# @Time : 2021/4/10 10:17
# @Author : Jclian91
# @File : get_json.py
# @Place : Yangpu, Shanghai
import os
import json

content = []
data_type = 'middle'
data_set = 'test'
file_dir = f"RACE/{data_set}/{data_type}"
for file in os.listdir(file_dir):
    if file.endswith(".txt"):
        file_path = os.path.join(file_dir, file)
        with open(file_path, "r", encoding="utf-8") as f:
            content.append(json.loads(f.read()))

with open(f"RACE/RACE_{data_type}/{data_set}.json", "w", encoding="utf-8") as g:
    g.write(json.dumps(content, ensure_ascii=False, indent=4))
