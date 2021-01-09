import torch
import torch.nn as nn
import math
import utils


class LunaModel(nn.Module):
    def __init__(self, in_channels, conv_channels, depth, height, width):
       super().__init__()
       self.in_channels = in_channels
       self.conv_channels = conv_channels
       self.depth = depth
       self.height = height
       self.width = width
       self.tail_batchnorm = nn.BatchNorm3d(1)
       self.luna_blocks = [
           LunaBlock(in_channels, conv_channels),
           LunaBlock(conv_channels, conv_channels * 2),
           LunaBlock(conv_channels * 2, conv_channels * 4),
           LunaBlock(conv_channels * 4, conv_channels * 8)
       ]
       self.head_linear = nn.Linear(self.calc_flat_size(), 2)
       self.head_softmax = nn.Softmax(dim=1)
       self._init_weights()
    
    def calc_flat_size(self):
        # each luna block halves the depth, height and width
        @utils.repeat_unary_func(len(self.luna_blocks))
        def halve_dimension(dim):
            return dim // 2
        d = halve_dimension(self.depth)
        h = halve_dimension(self.height)
        w = halve_dimension(self.width)
        c = self.luna_blocks[-1].conv_channels 
        return c * d * h * w

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

    def forward(self, input_batch):
        out = self.tail_batchnorm(input_batch)
        for luna_block in self.luna_blocks:
            out = luna_block(out)
        flat_out = out.view(
            out.size(0),
            -1
        )
        logits = self.head_linear(flat_out)
        return logits, self.head_softmax(logits)
        

class LunaBlock(nn.Module):
    def __init__(self, in_channels, conv_channels):
        super().__init__()
        self.in_channels = in_channels
        self.conv_channels = conv_channels
        self.conv1 = nn.Conv3d(
            in_channels, conv_channels, kernel_size=3, padding=1, bias=True
        )
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(
            conv_channels, conv_channels, kernel_size=3, padding=1, bias=True
        )
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(2, 2)
        

    def forward(self, input_batch):
        block_out = self.conv1(input_batch)
        block_out = self.relu1(block_out)
        block_out = self.conv2(block_out)
        block_out = self.relu2(block_out)
        return self.maxpool(block_out) 

