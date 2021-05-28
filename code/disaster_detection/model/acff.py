import torch
import torch.nn as nn

debug = False
'''Atrous Convolution Feature Fusion (ACFF) Block'''


class ACFF(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        ''' 
        Dilated Convolution

        i = input
        o = output
        p = padding
        k = kernel_size
        s = stride
        d = dilation
        
        o = [i + 2*p - k - (k-1)*(d-1)]/s + 1
        '''

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=0, dilation=1,
                               groups=in_channels, bias=True)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, dilation=2,
                               groups=in_channels, bias=True)
        self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=2, dilation=3,
                               groups=in_channels, bias=True)
        self.fused_conv = nn.Conv2d(in_channels * 3, out_channels, kernel_size=1, stride=1, padding=0, dilation=1,
                                    bias=True)
        self.leaky_relu = nn.LeakyReLU(0.01)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):

        if debug:
            print('Shape of input in ACFF Forward= ', x.shape)
            print('Output of layer1(x): ', self.conv1(x).shape)
            print('Output of layer2(x): ', self.conv2(x).shape)
            print('Output of layer3(x): ', self.conv3(x).shape)

        # Fusion
        out = torch.cat((self.conv1(x), self.conv2(x), self.conv3(x)), 1)

        if debug:
            print('Shape after concat in ACFF forward: ', out.shape)

        out = self.fused_conv(out)
        out = self.leaky_relu(out)
        out = self.batch_norm(out)
        out = self.dropout(out)

        if debug:
            print('Final shape of ACFF out: ', out.shape, '\n')

        return out
