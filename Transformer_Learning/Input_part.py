# Transformer Input
# 输入部分，处理两边的输入/词嵌入以及位置编码
import torch.nn as nn
import torch
import math

class Embedding(nn.Module):
    def __init__(self, vocab_size, output_dim):
        super().__init__()

        self.vocab_size = vocab_size
        self.output_dim = output_dim

        self.embed = nn.Embedding(vocab_size, output_dim)


    def forward(self, input):
        return self.embed(input) * math.sqrt(self.output_dim)

class PositionalEncoding(nn.Module):
    # 有关于数学公式，很不好理解，看情况了解。
    # 基本上是固定代码。
    def __init__(self, d_model, dropout=0.1, max_len=60):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model) # 词上限maxlen

        # 固定代码
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = 1/(10000**(torch.arange(0,d_model,2).float()/d_model))
        position_value = position * div_term
        pe[:, 0::2] = torch.sin(position_value)
        pe[:, 1::2] = torch.cos(position_value)

        pe = pe.unsqueeze(0) # pe形状变3维
        self.register_buffer('pe', pe)
        # pe注册到缓存中, 当pos和i变化的时候计算出对应的位置编码值。

    def forward(self, embed):
        result = embed + self.pe[:, :embed.size(1)]
        # 为什么对位置编码进行切片，位置编码是对于最大词数说的，如果词数不够只需要前面几个就可以了。
        return self.dropout(result)

