# 进行onehot编码的演示
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"]="0"

import jieba
from tensorflow.keras.preprocessing.text import Tokenizer
# 或者使用from keras.src.legacy.preprocessing.text import Tokenizer
# 效果完全一样
import joblib


# 准备语料
text = "你说的对，但是《原神》是由米哈游自主研发的一款全新开放世界冒险游戏"
def cut_fit(text):
    # 分词
    vocab = jieba.lcut(text)

    # 创建映射器并训练模型
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(vocab)

    # 得到训练数据
    word_dict = tokenizer.word_index

    # 手动进行编码
    # 对于每一个词，创建一个长度为词数的向量，初始化为0
    veclen = len(vocab)
    for word in vocab:
        word_vec = [0] * veclen
        word_vec[word_dict.get(word) - 1] = 1
        # word_vec[word_dict[word]-1] = 1
        # print(f"对于’{word}’, 它的词向量是{word_vec}")
        # 同一个词它的词向量是一样的，如这句话中的“的”

    joblib.dump(tokenizer, 'tokenizer.pkl')
def findword(word):
    tokenizer = joblib.load('tokenizer.pkl')
    # 得到onehot向量

    word_dict = tokenizer.word_index
    word_vec = [0]*len(word_dict)
    word_vec[word_dict[word] - 1] = 1
    print(f"对于‘{word}’, 它的向量是{word_vec}")

def cut_fit_standard(text):
    vocab = jieba.lcut(text)
    # 获得每个词的词向量
    for word in vocab:
        wordvec = [0] * len(vocab)
        wordvec[vocab.index(word)] = 1
        print(f"对于’{word}’, 它的词向量是{wordvec}")



if __name__ == '__main__':
    # word = "原神"
    # cut_fit(text)
    # findword(word)
    cut_fit_standard(text)