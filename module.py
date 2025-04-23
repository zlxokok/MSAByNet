import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable, Function


class MSblock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(MSblock, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=3, dilation=1, padding=1),
                                   nn.BatchNorm2d(out_ch),
                                   nn.ReLU(inplace=True)
                                   )
        self.conv2 = nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=3, dilation=3, padding=3),
                                   nn.BatchNorm2d(out_ch),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(out_ch, out_ch, kernel_size=1, dilation=1),
                                   nn.BatchNorm2d(out_ch),
                                   nn.ReLU(inplace=True),
                                   )
        self.conv3 = nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=3, dilation=1, padding=1),
                                   nn.BatchNorm2d(out_ch),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(out_ch, out_ch, kernel_size=3, dilation=3, padding=3),
                                   nn.BatchNorm2d(out_ch),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(out_ch, out_ch, kernel_size=1, dilation=1),
                                   nn.BatchNorm2d(out_ch),
                                   nn.ReLU(inplace=True),
                                   )
        self.conv4 = nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=3, dilation=1, padding=1),
                                   nn.BatchNorm2d(out_ch),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(out_ch, out_ch, kernel_size=3, dilation=3, padding=3),
                                   nn.BatchNorm2d(out_ch),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(out_ch, out_ch, kernel_size=3, dilation=5, padding=5),
                                   nn.BatchNorm2d(out_ch),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(out_ch, out_ch, kernel_size=1, dilation=1),
                                   nn.BatchNorm2d(out_ch),
                                   nn.ReLU(inplace=True),
                                   )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        return x1 + x2 + x3 + x4
        
        
        
        
class channel_attention(nn.Module):
    def __init__(self, k_size=3):
        super(channel_attention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size -1 ) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)
        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)        
        
class spatial_attention(nn.Module):
    def __init__(self, k_size=3):
        super().__init__()

        padding = k_size // 2

        self.conv = DepthWiseConv(in_channel=2, out_channel=1, k_size=k_size,pad = padding)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        x_maxpool, _ = torch.max(inputs, dim=1, keepdim=True)

        x_avgpool = torch.mean(inputs, dim=1, keepdim=True)
        x = torch.cat([x_maxpool, x_avgpool], dim=1)

        x = self.conv(x)

        x = self.sigmoid(x)

        outputs = inputs * x

        return outputs      
        
        
class DepthWiseConv(nn.Module):
    def __init__(self, in_channel, out_channel, k_size,pad):
        super(DepthWiseConv, self).__init__()

        self.depth_conv = nn.Conv2d(in_channels=in_channel,
                                    out_channels=in_channel,
                                    kernel_size=k_size,
                                    stride=1,
                                    padding=pad,
                                    groups=in_channel)
                                    
        self.point_conv = nn.Conv2d(in_channels=in_channel,
                                    out_channels=out_channel,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1)


    def forward(self, input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out
        
        
        
        
        
        
        
        
        
        
        
        
        
        
class casaModule(nn.Module):
    def __init__(self, k_size):
        super().__init__()
        self.ca = channel_attention(k_size=k_size)
        self.spa = spatial_attention(k_size=k_size)

    def forward(self, x):
        out = self.ca(x)
        out = self.spa(out)
        return out        
