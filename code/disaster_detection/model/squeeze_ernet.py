import torch.nn as nn
from model.acff import ACFF

debug = False


class Squeeze_ErNET(nn.Module):
    def __init__(self):
        super(Squeeze_ErNET, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=0, bias=False)
        self.acff1 = ACFF(16, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.acff2 = ACFF(64, 96)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.acff3 = ACFF(96, 128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.acff4 = ACFF(128, 256)
        self.conv2 = nn.Conv2d(in_channels=256, out_channels=5, kernel_size=1, stride=1, padding=0, bias=False)
        self.globalpool = nn.AvgPool2d(kernel_size=5, stride=1, padding=1)
        self.fc = nn.Linear(2 * 2 * 5, 5)
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