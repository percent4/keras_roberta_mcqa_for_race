# -*- coding: utf-8 -*-
# @Time : 2022/5/14 11:33
# @Author : Jclian91
# @File : load_data.py
# @Place : Minghang, Shanghai
import re
import json
from collections import namedtuple

from utils.params import train_file_path_list, dev_file_path_list

Sample = namedtuple("sample", ["id", "article", "question", "answers", "correct_answer"])

samples = []
for train_file_path in train_file_path_list:
    with open(train_file_path, "r", encoding="utf-8") as f:
         samples.extend(json.loads(f.read()))

train_samples = []
for sample in samples:
    article = re.sub("(（.+?）)", "", sample["article"].replace("\u3000", "").replace("\n", "").replace("\r", ""))
    for question, option, answer in zip(sample["questions"], sample["options"], sample["answers"]):
        if question:
            question = re.sub("(（.+?）)", "", question.replace("\u3000", "")
                              .replace("\n", "").replace("\r", ""))
            new_sample = Sample(sample["id"], article, question, option, ord(answer)-ord("A"))
            train_samples.append(new_sample)

samples = []
for dev_file_path in dev_file_path_list:
    with open(dev_file_path, "r", encoding="utf-8") as f:
        samples.extend(json.loads(f.read()))

dev_samples = []
for sample in samples:
    article = re.sub("(（.+?）)", "", sample["article"].replace("\u3000", "").replace("\n", "").replace("\r", ""))
    for question, option, answer in zip(sample["questions"], sample["options"], sample["answers"]):
        if question:
            question = re.sub("(（.+?）)", "", question.replace("\u3000", "")
                              .replace("\n", "").replace("\r", ""))
            new_sample = Sample(sample["id"], article, question, option, ord(answer)-ord("A"))
            dev_samples.append(new_sample)


print("训练集数量: ", len(train_samples))
print("测试集数量: ", len(dev_samples))
print("总样本数: ", len(train_samples) + len(dev_samples))

# view first 5 train data and test data
print("*"*100)
for _ in train_samples[:5]:
    print(_)
print("*"*100)
for _ in dev_samples[:5]:
    print(_)
