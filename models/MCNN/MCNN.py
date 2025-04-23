import torch.nn as nn
import torch.nn.functional as F
import torch
from torchsummary import summary


class MCNN(nn.Module):

    def __init__(self, ClassDim= 4, Points = 640, ChNum=25):
        super(MCNN, self).__init__()

        self.T = Points

        self.F1 = 16
        self.D = 2
        self.kern = 64
        self.p = 0.25
        self.F2 = self.F1 * self.D
        self.fcin = Points // 32 * self.F2

        self.seq = nn.Sequential(
            nn.Conv2d(1, self.F1, (1, self.kern), padding="same"), nn.BatchNorm2d(self.F1, False),
            nn.Conv2d(self.F1, self.F1 * self.D, (ChNum, 1), padding="valid", groups=self.F1),
            nn.BatchNorm2d(self.F1 * self.D, False), nn.ELU(), nn.AvgPool2d((1, 4)), nn.Dropout(self.p),
            nn.Conv2d(self.F1 * self.D, self.F1 * self.D, (1, 16), padding="same", groups=self.F1 * self.D),
            nn.Conv2d(self.F1 * self.D, self.F2, (1, 1)), nn.BatchNorm2d(self.F2, False), nn.ELU(), nn.AvgPool2d(
                (1, 8)), nn.Dropout(self.p), nn.Flatten(), nn.Linear(self.fcin, ClassDim))

    def forward(self, x, *other_input):
        x = x.unsqueeze(1)
        x = x.permute(0, 1, 3, 2)
        # print(x.shape)
        # x = trail * 1 * chNum * points
        # other_input = None
        return self.seq(x)


if __name__ == '__main__':
    input = torch.randn(32, 1, 25, 502)
    model = MCNN(ClassDim=4, Points=502, ChNum=22)
    out = model(input)
    print('===============================================================')
    print('out', out)
    print('model', model)
    summary(model=model, input_size=(1, 22, 502), batch_size=1, device="cpu")
