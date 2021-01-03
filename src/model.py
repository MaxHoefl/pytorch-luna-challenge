import torch
import torch.nn as nn
import math


class LunaModel(nn.Module):
    def __init__(self, in_channels=1, conv_channels=8):
       super().__init__()
       self.tail_batchnorm = nn.BatchNorm3d(1)

       self.block1 = LunaBlock(in_channels, conv_channels)
       self.block2 = LunaBlock(conv_channels, conv_channels * 2)
       self.block3 = LunaBlock(conv_channels * 2, conv_channels * 4)
       self.block4 = LunaBlock(conv_channels * 4, conv_channels * 8)
       self.head_linear = nn.Linear(1152, 2)
       self.head_softmax = nn.Softmax(dim=1)
       self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if type(m) in {nn.Linear, nn.Conv3d}:
                nn.init.kaiming_normal_(
                    m.weight.data, a=0, mode='fan_out', nonlinearity='relu'
                )
                if m.bias is not None:
                    fan_in, fan_out = \
                        nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                    bound = 1 / math.sqrt(fan_out)
                    nn.init.normal_(m.bias, -bound, bound)
        

class LunaBlock(nn.Module):
    def __init__(self, in_channels, conv_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(
            in_channels, conv_channels, kernel_size=3, padding=1, bias=True
        )
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(
            in_channels, conv_channels, kernel_size=3, padding=1, bias=True
        )
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(2, 2)

    def forward(self, input_batch):
        block_out = self.conv1(input_batch)
        block_out = self.relu1(block_out)
        block_out = self.conv2(block_out)
        block_out = self.relu2(block_out)
        conv_flat = block_out.view(
            block_out.size(0),
            -1
        )
        logits = self.head_linear(conv_flat)
        return logits, self.head_softmax(logits)

