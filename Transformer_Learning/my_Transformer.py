import torch.nn as nn
import torch

class MyTransformer(nn.Module):
    def __init__(self, encoder_ebd_pos, encoder, decoder_ebd_pos, decoder, output):
        super().__init__()
        self.encoder_ebd_pos = encoder_ebd_pos
        self.encoder = encoder
        self.decoder_ebd_pos = decoder_ebd_pos
        self.decoder = decoder
        self.output = output


    def forward(self, encoderInput, decoderInput, mask):
        encoder_input = self.encoder_ebd_pos(encoderInput)
        decoder_input = self.decoder_ebd_pos(decoderInput)
        encoder_output = self.encoder(encoderInput)
        decoder_output = self.decoder(decoderInput, mask, encoder_output)
        return self.output(decoder_output)

# 后续需要使用的时候，对于每一个需要使用的组件进行初始化以及组装，然后放入My_transformer类就可以了。
# nn.sequential 按顺序的 en.ebd - en_pos 将这两个对象传入encoder_ebd_pos



