# -*- coding: utf-8 -*-
# @Time : 2022/5/14 11:35
# @Author : Jclian91
# @File : model_predict.py
# @Place : Minghang, Shanghai
import json
import numpy as np

from model import SimpleMultiChoiceMRC
from model_train import tokenizer
from utils.params import (dataset,
                          CHECKPOINT_FILE_PATH,
                          CONFIG_FILE_PATH,
                          NUM_CHOICES,
                          MAX_SEQ_LENGTH
                          )

# 加载训练好的模型
model = SimpleMultiChoiceMRC(CONFIG_FILE_PATH, CHECKPOINT_FILE_PATH, MAX_SEQ_LENGTH, NUM_CHOICES).create_model()
model.load_weights("mcqa_race_model_0.6777777891192172.h5")

with open("./data/{}/predict.json".format(dataset), "r", encoding="utf-8") as f:
    content = json.loads(f.read())

article = content["article"]
question = content["question"]
options = content["options"]

X1 = np.empty(shape=(1, NUM_CHOICES, MAX_SEQ_LENGTH))
X2 = np.empty(shape=(1, NUM_CHOICES, MAX_SEQ_LENGTH))

for choice_num, answer in enumerate(options):
    p_q = question + article
    x1, x2 = tokenizer.encode(first=answer, second=p_q, max_len=MAX_SEQ_LENGTH)
    X1[0, choice_num, :] = x1
    X2[0, choice_num, :] = x2

predict_result = model.predict([X1, X2])
print(predict_result)