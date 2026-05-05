import torch.nn as nn
import torch
import attention

# 注意力/多头注意力对象
# 残差连接/层归一化
# 子层连接
class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.k = nn.Parameter(torch.ones(d_model))
        self.b = nn.Parameter(torch.zeros(d_model))

    def forward(self, data):
        mean = data.mean(-1, keepdim=True)
        std = data.std(-1, keepdim=True)
        eps = 1e-6
        return self.k * (data - mean) /std + eps  + self.b


class SubLayerConnection(nn.Module):
    def __init__(self, d_model, dropout_p=0.1):
        super().__init__()

    def forward(self, data, sublayer_obj):
        pass


class EncoderLayer(nn.Module):
    def __init__(self, d_model, multi_head_self_attn, feed_forward, dropout_p = 0.1):
        super().__init__()
        self.multi_head_self_attn = multi_head_self_attn # 多头自注意力
        self.feed_forward = feed_forward                 # 前馈
        self.dropout = nn.Dropout(p=dropout_p)
        self.d_model = d_model

        # 每一个层有两个子层，每一个子层分别包含层归一化/残差连接和注意力/前馈网络层
        self.multiheadLayer = SubLayerConnection(d_model, dropout_p)
        self.feed_forward = SubLayerConnection(d_model, dropout_p)
    def forward(self, data):
        multi_output = self.multiheadLayer(data, lambda x: self.multi_head_self_attn(q = x, k = x, v = x))
        # lambda x 不能提前运算，将层作为可调用对象传入。
        feed_output = self.feed_forward(multi_output, lambda x: self.feed_forward(x))
        return feed_output

# 编码器可以叠好几层 每一个层有 注意力子层/前馈网络子层
# 注意力 - 多头自注意力 + 层归一化 + 残差联接
# 前馈网络 - 前馈网络 + 层归一化 + 残差连接

class Encoder(nn.Module):
    def __init__(self, Encoder_Layer, N):
        super().__init__()
        # 串行处理
        self.encoderList = attention.clones(Encoder_Layer, N)
        # 层归一化,不是必须的，缓解经过N层处理后的数据，让数据变得更加平稳
        # 六层处理完之后层归一化
        self.layer_norm = LayerNorm(Encoder_Layer.d_model)

    def forward(self, data):
        result = data
        # 编码器层串行执行
        for layer in self.encoder_layer_list:
            data = layer(data)
            # 处理的数据名称必须与data保持一致，保证串行执行
        # 最终处理好的数据，再次经过层归一化
        return self.layer_norm(data)


# 1:05:46