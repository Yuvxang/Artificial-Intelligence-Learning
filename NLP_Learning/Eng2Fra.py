from random import random

import torch
import torch.nn as nn
import re
import matplotlib.pyplot as plt
from click.termui import hidden_prompt_func
from torch.utils.data import TensorDataset, DataLoader
from torch.utils import data
from tqdm import tqdm

## 在实际工作中，工作量大概在10天左右
## 大部分时间是在开会修bug

# 设备选择, 我们可以选择在cuda或者cpu上运行你的代码
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 起始标志
SOS_token = 0
# 结束标志
EOS_token = 1
# 最大句子长度不能超过10个 (包含标点)
MAX_LENGTH = 10
# 数据文件路径
data_path = "dataset/eng-fra-v2.txt"


def normalize_str(str):
    str = str.lower().strip()
    str = re.sub(r"([.!?])", r" \1", str)
    str = re.sub(r"[^a-zA-Z.!?]+", " ", str)
    return str

def get_data():
    with open(data_path) as fr:
        lines = fr.read().strip().split("\n")

    pairs = [[normalize_str(asen) for asen in line.split("\t")] for line in lines]
    eng_word2index = {"SOS": SOS_token, "EOS": EOS_token}
    fra_word2index = {"SOS": SOS_token, "EOS": EOS_token}


    for pair in pairs:
        for word in pair[0].split(" "):
            if word not in eng_word2index: # 去重
                eng_word2index[word] = len(eng_word2index)

    for pair in pairs:
        for word in pair[1].split(" "):
            if word not in fra_word2index:
                fra_word2index[word] = len(fra_word2index)

    eng_index2word = {v:k for k, v in eng_word2index.items()}
    fra_index2word = {v:k for k, v in fra_word2index.items()}
    # 字典也有解析式

    return eng_word2index, eng_index2word, len(eng_word2index), fra_word2index, fra_index2word, len(fra_word2index), pairs

result = get_data()
eng_word2index = result[0]
eng_index2word = result[1]
vocab_size = len(eng_index2word)
fra_word2index = result[3]
pairs = result[-1]

# 构建dataset数据源

class MyPairsDataset(data.Dataset):
    def __init__(self, my_pairs, eng_word2index, fra_word2index):
        super().__init__()
        # 给出样本
        self.my_pairs = my_pairs
        self.eng_word2index = eng_word2index
        self.fra_word2index = fra_word2index

        # 给出样本数
        self.samplelen = len(my_pairs)

    def __len__(self):
        return self.samplelen

    # 拿到某一个条目
    def __getitem__(self, idx):
        # 传一个索引，取出样本

        # index异常修正
        index = min(max(idx, 0), len(self.my_pairs) - 1)

        # 按索引获取，x, y
        x = self.my_pairs[index][0] # 英
        y = self.my_pairs[index][1] # 法

        # 文本张量化
        x = [self.eng_word2index[word] for word in x.split(' ')]        # 单词转索引
        x.append(EOS_token)# 句子的末尾
        # 为什么在这里加EOS_Token
        # 在Seq2seq+注意力机制中，不管是Decoder/Encoder 必须明确要有句子末尾标识。
        # 开始标识不是必须的，同时我们在后续模型训练的时候再加上开始标识token
        tensor_x = torch.tensor(x, dtype=torch.long, device=device)

        y = [self.fra_word2index[word] for word in y.split(' ')]
        y.append(EOS_token)
        tensor_y = torch.tensor(y, dtype=torch.long, device=device)

        return tensor_x, tensor_y

def get_dataloader(pairs, eng_word2index, fra_word2index, batch_size):
    engfra_dataset = MyPairsDataset(pairs, eng_word2index, fra_word2index)
    dataloader = DataLoader(dataset=engfra_dataset, batch_size=batch_size, shuffle=True)
    return dataloader

# 构建GRU
class GRUEncoder(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        result = self.embedding(input)
        result = self.gru(result, hidden) # gru的输入 rnn的输入 h0 x
        return result

    def inithidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)



class Encoder(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        # 1.初始化父类
        super().__init__()
        # 2.设置属性值
        self.vocab_size:int = vocab_size
        self.hidden_size:int = hidden_size

        # 3.定义网络结构
        # 词嵌入层
        """
            num_embeddings 词汇表中词的个数 - vocab_size
            embedding_dim 词嵌入后向量的维度 - hidden_size - 隐藏层256 512 1024
        
        在ebd下
            input_size 输入的向量维度
            hidden_size 隐藏状态向量维度
            num_layers 隐藏层层数
            batch_first batch_size放在张量第一位
            只会影响输入和输出的张量形状，不会改变隐藏状态的张量形状
            
            本次输入(batch_size, seq_len, input_size)
            h0(num_layers, batch_size, hidden_size)        
        """
        self.ebd = nn.Embedding(num_embeddings=vocab_size, embedding_dim=hidden_size, device=device)
        self.gru = nn.GRU(input_size=hidden_size, hidden_size=hidden_size, batch_first=True, device=device)
        # 词嵌入层的结果作为输入给到GRU，所以a列 = b行 斜对角相等。
    """
    input 本次输入到GRU中的数据[seq_len, batch_size]
    hidden 上一时间步的隐藏状态[num_layers, batch_size, hidden_size]
    output[seq_len, batch_size, hidden_size]
   
    解释一下输出形状
    打个比方，x=[1，8]，因为设置了batch_first = True，所以这里的1就是batch_size
    只有一个样本，8也就是8个词的seq_len。
    经过embed，对于每一个词，将所有的词变为256维，然后在词表里面查找，
    于是就可以得到8个256维，就成了[1, 8, 256]
    
    经过gru，使用h0以及embed(x)
    embed为[1, 8, 256]
    h0为[1, 1, 256]
    对于embed有8个词需要处理，使用8个时间步，将所有的时间步的结果堆起来就是
    [1, 8, 256]
    即使hidden_size和input_size不同，hidden_size决定了(就是)h的第三维
    如h0为[1, 1, 128]
    结果是[1, 8, 128]
    """
    def forward(self, input, hidden):
        return self.gru(self.ebd(input), hidden)
        # 返回的是一个包，(output, hidden)
    # 最初，input是二维的只有batch_size和seq_len，经过embedding变为三维

    def init_hidden(self):
        # 用作h0 1就是num_layers, 1是batch_size句子长度不等=1，然后就是hidden_size
        return torch.zeros(1, 1, self.hidden_size, device=device)


def use_Encoder(english_word2index):
    hidden_size = 256
    encoder = Encoder(len(english_word2index), hidden_size)
    train_dataloader = get_dataloader(pairs, english_word2index, fra_word2index, 1)
    for x, y in train_dataloader:
        h0 = encoder.init_hidden()
        output, hidden = encoder(x, h0)
        print(output.shape)
        print(h0.shape)
        break

## 不带注意力的decoder
# 为了得到翻译结果 新加入两个线性层 out 和 softmax 一个是线性求和，一个是激活函数，出预测结果
# 另外解码器本身也有词嵌入层以及gru层，

class Decoder(nn.Module):
    def __init__(self, vocab_size, hidden_size):   # 这边的vocab_size指的是法语的输出单词个数
        super().__init__()
        self.vocab_size = vocab_size                       # 法语单词的个数
        self.hidden_size:int = hidden_size

        self.embed = nn.Embedding(vocab_size, hidden_size) # 词嵌入层
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers=1, batch_first=True)        # 神经网络
        self.out = nn.Linear(hidden_size, vocab_size)      # 输出线性变换
        self.softmax = nn.LogSoftmax(dim=1)                # 激活函数
        # 旧版本 LogSoftmax激活函数  + NLLLoss损失函数 过会模型训练的时候必须使用NLLLoss()
        # 新版本CrossEntropyLoss，自带softmax
    # 如果inputsize和hiddensize一样，可以直接只使用一个

    # 翻译的时候解码器怎么会有输入呢？。。。
    # 解码是一个token一个token解码的，所以input是(1, 1)
    def forward(self, input, hidden):
        embed = torch.relu(self.embed(input))   # 激活函数 防止过拟合/dropout随机失活也可以
        # 加relu使得矩阵稀疏，防止过拟合，因为relu在小于0的时候输出为0
        output, hidden = self.gru(embed, hidden) # embed[1, 1, 256] hidden[1, 1, 256]，输出还是一样、
        tmp_output = output[0] # 送入gru需要2-3，送入out需要3-2 得到[1, 4345]
        '''
            output张量形状[batch_size, seq_len, hidden_size]
            output[0] 针对翻译/文本生成业务场景，N vs M 连续输出
            output[:, -1, :] 针对文本分类的业务场景 N vs 1   一次输出
            只有一层中括号就是取其中一个小矩阵
             0就是第一个时间步/当前时间步的隐藏状态
            -1就是最后一个时间步的隐藏状态
            
            同时，forward中的input输入来自于dataloader
            这是因为现在还在模型训练阶段，
            为了避免错误积累，使用teacher forcing策略，需要使用目标句子。
        '''
        # output = torch.functional.log_softmax(result, dim=-1)
        output = self.softmax(self.out(tmp_output))
        return output, hidden

    def inithidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)
        # 只要是想要在GPU上运行，自己定义的张量都得扔到GPU device上

def useDecoder():
    hidden_size = 256
    myDataloader = get_dataloader(eng_word2index,fra_word2index, fra_word2index, 1)
    myEncoder = Encoder(len(eng_word2index), hidden_size).to(device)
    myDecoder = Decoder(len(fra_word2index), hidden_size).to(device)
    # 需要encoder的最后一个时间步的输出作为h0，或者直接使用全0
    h0 = myEncoder.init_hidden()
    for i, (x, y) in enumerate(myDataloader):
        encoder_output, hidden = myEncoder(x, h0)
        # 翻译必须一个词一个词翻译，串行进行
        for idx in range(y.shape[1]): # y是已经张量化过的法语句子，它的[1]就是词个数
            print(y[0][idx])
            temp = y[0][idx].view(1, -1) # view和reshape作用一样，标量变成张量
            decoder_output, hidden = myDecoder(temp, hidden)
            break

    # AttnDecoder 带有注意力的解码器
    """
        有什么区别？
        1.dropout随机失活参数
        2.用于QK相似性计算以及形状变换的线性层
        实际上就是在原先的解码器上加上了6步
    """
class AttnDecoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.dropout = nn.Dropout(p=dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers=1, batch_first=True)
        self.attn = nn.Linear(hidden_size + hidden_size,MAX_LENGTH) # 计算QK相似性，maxlength是为了计算权重的时候不需要根据编码器第二位变换而定义多个线性层
        self.attn_combine = nn.Linear(hidden_size + hidden_size, hidden_size) # 将qc拼接后形状变换
        self.out = nn.Linear(hidden_size, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input, K, V):   # input是上一时间步的法语词词索引
        Q = self.dropout(self.embed(input))

        # 计算专属信息包
        # QK拼接，计算相似度，相似度转概率，概率*V得专属信息包，专属信息包和Q拼接，拼接后形状变换
        QK = torch.concat([Q, K], dim=-1)
        score = self.attn(QK)
        scoreweight = torch.softmax(score, dim=1)
        C = torch.bmm(scoreweight, V)
        QC = torch.concat([Q, C], dim=-1)
        hidden_input = self.attn_combine(QC)
        output, hidden = self.gru(hidden_input, K)

        tmp_output = output[0]
        final_output = self.softmax(self.out(tmp_output))

        return final_output, hidden, scoreweight
        # 2:00:00
    """
    train本身是一个函数
    单条样本的训练过程 是一个函数
    """

def train_iters(x, y, encoder, decoder, encoder_opt, decoder_opt, loss):
    # 编码器使用
    encoder_hidden = encoder.inithidden()
    encoder_output, encoder_hidden = encoder(x, encoder_hidden)

    # 句子长度规范 batch_size, 词上限=10, 每个词词向量256
    value = torch.zeros(1, MAX_LENGTH, 256, device=device)
    copy_len = min(MAX_LENGTH, x.shape[1]) # 英语句子中词个数， x是二维[batch, seq_len]
    # 实际上是截断的工作
    value[:, :copy_len, :] = encoder_output[:, :copy_len, :]
    # 不等于的补为0，是补齐

    decoder_hidden = decoder.inithidden()
    input_y = torch.tensor([[SOS_token]], device=device) # 翻译结果的张量，先放入开始
    loss_value = torch.tensor(0.0, device=device) # 初始损失值
    y_len = y.shape[1]

    # 教师机制，当发现预测结果不正确的时候，告诉模型真实的目标值
    # 让模型不至于越来越错。训练平稳，固定设置为True 模型完全过拟合
    # 注意！ 只能使用在模型训练过程，不能用在预测过程。
    teacher_forcing_flag = True if random() < 0.5 else False
    if teacher_forcing_flag:
        for i in range(y_len):
            output, decoder_hidden, attn_weights = decoder(input_y, decoder_hidden, value)
            y_true = y[0][i].reshape(1)
            loss_value += loss(output, y_true)

            input_y = y[0][i].reshape(1, -1)
    else:
        for i in range(y_len):
            output, decoder_hidden, attn_weights = decoder(input_y, decoder_hidden, value)
            y_true = y[0][i].reshape(1)
            loss_value += loss(output, y_true)

            input_y = output[0][i].reshape(1, -1)



def train(english_word_n):
    dataloader = get_dataloader()

    size = 256
    # 模型初始化
    encoder = Encoder(vocab_size=english_word_n, hidden_size=size)
    my_encoder = Encoder(vocab_size=size, hidden_size=size).to(device=device)
    my_decoder = AttnDecoder(vocab_size=size, hidden_size=size).to(device=device)

    # 优化器初始化 梯度下降优化器
    encoder_adam = torch.optim.Adam(params=my_encoder.parameters(), lr=1e-4)
    decoder_adam = torch.optim.Adam(params=my_decoder.parameters(), lr=1e-4)

    # 损失函数 NLLLoss 新版是 CrossEntropyLoss
    loss = nn.NLLLoss()

    epochs = 1 # 训练轮次
    plot_avg_loss_list = []
    for epoch in range(epochs):
        # 外层控制轮次
        plot_total_loss = 0.0   # 每轮次总损失值
        print_total_loss = 0.0  #

        for i, (x, y) in enumerate (tqdm(dataloader), start=1): # x英语y法语，特征对标签
            # 内层控制批次

            loss_value = train_iters(x, y, my_encoder, my_decoder, encoder_adam, decoder_adam, loss)

            print_total_loss += loss_value
            plot_total_loss += loss_value

            if i % 100 == 0:
                avg_loss = plot_total_loss / 100
                plot_avg_loss_list.append(avg_loss)
                plot_total_loss = 0.0
            # 每100个轮次单独算一个损失

            if i % 1000 == 0:
                avg_loss = print_total_loss / 1000
                print(f"第{epoch+1}轮次, 已经训练的样本条数{i}, 平均损失为{avg_loss}")



    torch.save(my_encoder.state_dict(), '../model/my_encoder.pkl')
    torch.save(my_decoder.state_dict(), '../model/my_attendecoder.pkl')
if __name__ == '__main__':
    # return eng_word2index, eng_index2word, len(eng_word2index), fra_word2index, fra_index2word, len(
    #     fra_word2index), pairs
    useDecoder()

# 总之 翻译问题就是使用英语语料，对于目标语言的字典的多分类问题。