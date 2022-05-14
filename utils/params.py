# -*- coding: utf-8 -*-
# @Time : 2022/5/14 11:32
# @Author : Jclian91
# @File : params.py
# @Place : Minghang, Shanghai
import os
dataset = ["RACE_high", "RACE_middle"]
train_file_path_list = [f"./data/RACE/{_}/train.json" for _ in dataset]
dev_file_path_list = [f"./data/RACE/{_}/dev.json" for _ in dataset]
test_file_path_list = [f"./data/RACE/{_}/test.json" for _ in dataset]

# 模型配置
roberta_path = 'roberta-base'
tf_roberta_path = 'tf_roberta_base'
tf_ckpt_name = 'tf_roberta_base.ckpt'
vocab_path = 'keras_roberta'

CONFIG_FILE_PATH = os.path.join(tf_roberta_path, 'bert_config.json')
CHECKPOINT_FILE_PATH = os.path.join(tf_roberta_path, tf_ckpt_name)
GPT_BPE_VOCAB = os.path.join(vocab_path, 'encoder.json')
GPT_BPE_MERGE = os.path.join(vocab_path, 'vocab.bpe')
ROBERTA_DICT = os.path.join(roberta_path, 'dict.txt')

# 模型参数配置
NUM_CHOICES = 4
EPOCH = 20
BATCH_SIZE = 3
MAX_SEQ_LENGTH = 380
WARMUP_RATION = 0.1
