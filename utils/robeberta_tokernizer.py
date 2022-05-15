# -*- coding: utf-8 -*-
# @Time : 2022/5/14 11:41
# @Author : Jclian91
# @File : robeberta_tokernizer.py
# @Place : Minghang, Shanghai
from utils.params import *
from keras_roberta.tokenizer import RobertaTokenizer


# roberta tokenizer function for text pair
def tokenizer_encode(tokenizer, text1, text2, max_seq_length):
    sep = [tokenizer.sep_token]
    cls = [tokenizer.cls_token]
    # 1. 先用'bpe_tokenize'将文本转换成bpe tokens
    tokens1 = tokenizer.bpe_tokenize(text1)
    tokens2 = tokenizer.bpe_tokenize(text2)
    while len(tokens1) + len(tokens2) > max_seq_length - 4:
        tokens1.pop()
    token_ids1 = cls + tokens1 + sep
    token_ids2 = sep + tokens2 + sep
    # 2. 最后转换成id
    token_ids1 = tokenizer.convert_tokens_to_ids(token_ids1)
    token_ids2 = tokenizer.convert_tokens_to_ids(token_ids2)
    token_ids = token_ids1 + token_ids2
    segment_ids = [0] * len(token_ids1) + [1] * len(token_ids2)

    pad_length = max_seq_length - len(token_ids)
    if pad_length >= 0:
        token_ids += [0] * pad_length
        segment_ids += [0] * pad_length

    return token_ids, segment_ids


if __name__ == '__main__':
    tokenizer = RobertaTokenizer(GPT_BPE_VOCAB, GPT_BPE_MERGE, ROBERTA_DICT)
    text1 = "Pit-a-pat. Pit-a-pat. It's raining. \"I want to go outside and play, Mum,\" Robbie says, " \
            "\"When can the rain stop?\" His mum doesn't know what to say. She hopes the rain can stop, too. " \
            "\"You can watch TV with me,\" she says. \"No, I just want to go outside.\" \"1Put on your raincoat.\" " \
            "\"Does it stop raining?\" \"No, but you can go outside and play in the rain. Do you like that?\" \"Yes, " \
            "mum.\" He runs to his bedroom and puts on his red raincoat. \"Here you go. Go outside and play.\" " \
            "Mum opens the door and says. Robbie runs into the rain. Water goes 2here and there. Robbie's mum " \
            "watches her son. He is having so much fun. \"Mum, come and play with me!\" Robbie calls. The door " \
            "opens and his mum walks out. She is in her yellow raincoat. Mother and son are out in the rain for a " \
            "long time. They play all kinds of games in the rain."
    text2 = "Which is the best title for the passage? Robbie's raincoat"
    print(len(text1.split()))
    a_ids, b_ids = tokenizer_encode(tokenizer, text1, text2, 300)
    print(a_ids, len(a_ids))
    print(b_ids, len(b_ids))
