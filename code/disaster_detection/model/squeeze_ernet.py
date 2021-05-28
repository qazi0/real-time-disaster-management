import torch.nn as nn
from model.acff import ACFF

debug = False


class Squeeze_ErNET(nn.Module):
    def __init__(self):
        super(Squeeze_ErNET, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv_red1 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=1, stride=1, padding=0, bias=False)
        self.acff1 = ACFF(8, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_red2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0, bias=False)
        self.acff2 = ACFF(32, 96)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_red3 = nn.Conv2d(in_channels=96, out_channels=48, kernel_size=1, stride=1, padding=0, bias=False)
        self.acff3 = ACFF(48, 128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_red4 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False)
        self.acff4 = ACFF(64, 256)
        self.conv2 = nn.Conv2d(in_channels=256, out_channels=5, kernel_size=1, stride=1, padding=0, bias=False)
        self.globalpool = nn.AvgPool2d(kernel_size=5, stride=1, padding=1)
        self.fc = nn.Linear(2 * 2 * 5, 5)
        self.soft = nn.Softmax(dim=1)

    def forward(self, x):
        #(x.shape)
        out = self.conv1(x)
        out = self.conv_red1(out)
        out = self.acff1(out)
        out = self.pool1(out)
        out = self.conv_red2(out)
        out = self.acff2(out)
        out = self.pool2(out)
        out = self.conv_red3(out)
        out = self.acff3(out)
        out = self.pool3(out)
        out = self.conv_red4(out)
        out = self.acff4(out)
        out = self.conv2(out)
        out = self.globalpool(out)

        if debug:
            print('Shape of globalpool output: ', out.shape)

        out = out.view(-1, 2 * 2 * 5)
        out = self.fc(out)
        out = self.soft(out)

        if debug:
            print('Final shape of Squeeze ErNET Output: ', out.shape)

        return out
