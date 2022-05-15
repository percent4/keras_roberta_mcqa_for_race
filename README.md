# Keras Roberta for Multiple Choice Question Answering(MCQA)

### Requirements

see requirements.txt

### How to Convert Torch Roberta to TensorFlow Roberta:

0. Download `Fairseq` Roberta Pretrained Model: [https://github.com/pytorch/fairseq/blob/main/examples/roberta/README.md](https://github.com/pytorch/fairseq/blob/main/examples/roberta/README.md)

1. Convert roberta weights from PyTorch to Tensorflow

```
python convert_roberta_to_tf.py 
    --model_name your_model_name
    --cache_dir /path/to/pytorch/roberta 
    --tf_cache_dir /path/to/converted/roberta
```

2. Extract features as `tf_roberta_demo.py`

```
python tf_roberta_demo.py 
    --roberta_path /path/to/pytorch/roberta
    --tf_roberta_path /path/to/converted/roberta
    --tf_ckpt_name your_mode_name
```

### train Multiply Choice Question Model

1. get data: `RACE` dataset: [https://huggingface.co/datasets/race](https://huggingface.co/datasets/race), use `tar` command to extract, then run `get_json.py` in data directory.

2. train the model: `python model_train.py`

3. model result:



### test model result

- 初中阅读理解题目

1. 
```
{
  "article": "Edward rose early on the New-year morning. He looked in every room and wished a Happy New Year to his family. Then he ran into the street to repeat that to those he might meet.\n\n　　When he came back, his father gave him two bright, new silver dollars.\n\n　　His face lighted up as he took them. He had wished for a long time to buy some pretty books that he had seen at the bookstore.\n\n　　He left the house with a light heart, expecting to buy the books. As he ran down the street, he saw a poor family.\n\n　　“I wish you a Happy New Year.” said Edward, as he was passing on. The man shook his head.\n\n　　“You are not from this country.” said Edward. The man again shook his head, for he could not understand or speak his language. But he pointed to his mouth and to the children shaking with cold, as if (好像) to say, “These little ones have had nothing to eat for a long time.”\n\n　　Edward quickly understood that these poor people were in trouble. He took out his dollars and gave one to the man, and the other to his wife.\n\n　　They were excited and said something in their language, which doubtless meant, “We thank you so much that we will remember you all the time.”\n\n　　When Edward came home, his father asked what books he had bought. He hung his head a moment, but quickly looked up.\n\n　　“I have bought no books”, said he. “I gave my money to some poor people, who seemed to be very hungry then.” He went on, “I think I can wait for my books till next New Year.”\n\n　　“My dear boy,” said his father, “here are some books for you, more as a prize for your goodness of heart than as a New-year gift”\n\n　　“I saw you give the money cheerfully to the poor German family. It was nice for a little boy to do so. Be always ready to help others and every year of your life will be to you a Happy New Year.”",
  "question": "We know that Edward_________ from the passage?",
  "options": [
    "A. got a prize for his kind heart",
    "B. had to buy his books next year",
    "C. bought the books at the bookstore",
    "D. got more money from his father"
  ]
}
```

预测结果为：

```
[[]]
正确答案: 
```

### 参考网址
1. [keras_roberta](https://github.com/midori1/keras_roberta)
2. [keras_bert_multiple_choice_MRC](https://github.com/percent4/keras_bert_multiple_choice_MRC)
3. [RACE: Large-scale Reading Comprehension Dataset From Examinations](https://arxiv.org/pdf/1704.04683v5.pdf)