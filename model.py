# -*- coding: utf-8 -*-
# @Time : 2022/5/14 11:34
# @Author : Jclian91
# @File : model.py
# @Place : Minghang, Shanghai
import tensorflow as tf
from keras.models import Model
from keras.utils import plot_model
from keras.layers import Input, Lambda, Dense, Activation, MaxPooling1D

from keras_roberta.roberta import build_bert_model


# model structure of SimpleMultiChoiceMRC
class SimpleMultiChoiceMRC(object):
    def __init__(self, config_path, checkpoint_path, max_len, num_choices):
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.max_len = max_len
        self.num_choices = num_choices

    def create_model(self):
        # Roberta model
        roberta_model = build_bert_model(self.config_path, self.checkpoint_path, roberta=True)  # 建立模型，加载权重
        for layer in roberta_model.layers:
            layer.trainable = True

        # get bert encoder vector
        x1_in = Input(shape=(self.num_choices, self.max_len, ), name="token_ids")
        x2_in = Input(shape=(self.num_choices, self.max_len, ), name="segment_ids")
        reshape_x1_in = Lambda(lambda x: tf.reshape(x, [-1, self.max_len]), name="reshape1")(x1_in)
        reshape_x2_in = Lambda(lambda x: tf.reshape(x, [-1, self.max_len]), name="reshape2")(x2_in)
        bert_layer = roberta_model([reshape_x1_in, reshape_x2_in])
        cls_layer = Lambda(lambda x: x[:, 0], name="cls_layer")(bert_layer)
        logits = Dense(1, name="classifier", activation=None)(cls_layer)
        reshape_layer = Lambda(lambda x: tf.reshape(x, [-1, self.num_choices]), name="reshape3")(logits)
        output = Activation(activation="softmax")(reshape_layer)

        model = Model([x1_in, x2_in], output)
        model.summary()
        # plot_model(model, to_file="model.png")

        return model


if __name__ == '__main__':
    model_config = "./chinese_L-12_H-768_A-12/bert_config.json"
    model_checkpoint = "./chinese_L-12_H-768_A-12/bert_model.ckpt"
    model = SimpleMultiChoiceMRC(model_config, model_checkpoint, 400, 4).create_model()