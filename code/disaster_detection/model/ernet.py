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


class ErNET(nn.Module):
    def __init__(self):
        super(ErNET, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=0, bias=False)
        self.acff1 = ACFF(16, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.acff2 = ACFF(64, 96)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.acff3 = ACFF(96, 128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.acff4 = ACFF(128, 128)
        self.acff5 = ACFF(128, 128)
        self.acff6 = ACFF(128, 256)
        self.conv2 = nn.Conv2d(in_channels=256, out_channels=5, kernel_size=1, stride=1, padding=0, bias=False)
        self.globalpool = nn.AvgPool2d(kernel_size=5, stride=1, padding=0)
        self.fc = nn.Linear(3 * 3 * 5, 5)
        self.soft = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.acff1(out)
        out = self.pool1(out)
        out = self.acff2(out)
        out = self.pool2(out)
        out = self.acff3(out)
        out = self.pool3(out)
        out = self.acff4(out)
        out = self.acff5(out)
        out = self.acff6(out)
        out = self.conv2(out)
        out = self.globalpool(out)

        if debug:
            print('Shape of globalpool output: ', out.shape)

        out = out.view(-1, 5 * 3 * 3)
        out = self.fc(out)
        out = self.soft(out)

        if debug:
            print('Final shape of ErNET Output: ', out.shape)

        return out
