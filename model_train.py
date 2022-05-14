# -*- coding: utf-8 -*-
# @Time : 2022/5/14 11:33
# @Author : Jclian91
# @File : model_train.py
# @Place : Minghang, Shanghai
import numpy as np

from keras.losses import categorical_crossentropy
from keras.callbacks import ModelCheckpoint
from keras_bert import AdamWarmup, calc_train_steps
from tensorflow.python import math_ops, array_ops

from utils.load_data import train_samples, dev_samples
from utils.robeberta_tokernizer import tokenizer_encode
from model import SimpleMultiChoiceMRC
from utils.params import *
from keras_roberta.tokenizer import RobertaTokenizer

tokenizer = RobertaTokenizer(GPT_BPE_VOCAB, GPT_BPE_MERGE, ROBERTA_DICT)


# data generator for model
class DataGenerator:

    def __init__(self, data, batch_size=BATCH_SIZE):
        self.data = data
        self.batch_size = batch_size
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1

    def __len__(self):
        return self.steps

    def __iter__(self):
        while True:
            idxs = list(range(len(self.data)))
            X1 = np.empty(shape=(self.batch_size, NUM_CHOICES, MAX_SEQ_LENGTH))
            X2 = np.empty(shape=(self.batch_size, NUM_CHOICES, MAX_SEQ_LENGTH))
            Y = np.zeros(shape=(self.batch_size, NUM_CHOICES))
            i = 0
            for j in idxs:
                sample = self.data[j]
                Y[i % self.batch_size, sample.correct_answer] = 1
                for choice_num, answer in enumerate(sample.answers):
                    x1, x2 = tokenizer_encode(tokenizer, sample.article, sample.question+answer, MAX_SEQ_LENGTH)
                    X1[i % self.batch_size, choice_num, :] = x1
                    X2[i % self.batch_size, choice_num, :] = x2

                if ((i+1) % self.batch_size == 0) or j == idxs[-1]:
                    yield [X1, X2], Y
                    X1 = np.empty(shape=(self.batch_size, NUM_CHOICES, MAX_SEQ_LENGTH))
                    X2 = np.empty(shape=(self.batch_size, NUM_CHOICES, MAX_SEQ_LENGTH))
                    Y = np.zeros(shape=(self.batch_size, NUM_CHOICES))

                i += 1


# 标签平滑机制
def categorical_cross_entropy_with_label_smoothing(y_true, y_pred, label_smoothing=0.1):
    num_classes = math_ops.cast(array_ops.shape(y_true)[1], y_pred.dtype)
    y_true = y_true * (1.0 - label_smoothing) + (label_smoothing / num_classes)
    return categorical_crossentropy(y_true, y_pred)


if __name__ == '__main__':
    # 模型训练
    train_D = DataGenerator(train_samples)
    dev_D = DataGenerator(dev_samples)
    model = SimpleMultiChoiceMRC(CONFIG_FILE_PATH, CHECKPOINT_FILE_PATH, MAX_SEQ_LENGTH, NUM_CHOICES).create_model()
    # add warmup
    total_steps, warmup_steps = calc_train_steps(
        num_example=len(train_samples),
        batch_size=BATCH_SIZE,
        epochs=EPOCH,
        warmup_proportion=WARMUP_RATION,
    )
    optimizer = AdamWarmup(total_steps, warmup_steps, lr=2e-5, min_lr=1e-8)
    filepath = "models/mcqa_race_model-{epoch:02d}-{val_acc:.4f}.h5"
    checkpoint = ModelCheckpoint(filepath,
                                 monitor='val_acc',
                                 verbose=1,
                                 save_best_only=True,
                                 save_weights_only=True,
                                 mode='max')
    model.compile(
        loss=categorical_cross_entropy_with_label_smoothing,
        optimizer=optimizer,
        metrics=['accuracy']
    )

    print("begin model training...")
    model.fit_generator(
        train_D.__iter__(),
        steps_per_epoch=len(train_D),
        epochs=EPOCH,
        validation_data=dev_D.__iter__(),
        validation_steps=len(dev_D),
        callbacks=[checkpoint]
    )

    print("finish model training!")

    result = model.evaluate_generator(dev_D.__iter__(), steps=len(dev_D))
    print("model evaluate result: ", result)
