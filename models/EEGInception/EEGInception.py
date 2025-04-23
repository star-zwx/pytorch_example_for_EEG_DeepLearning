# Revision: Chengxuan Qin <Chengxuan.Qin@outlook.com> Description: Santamaría-Vázquez developed "EEGInception" in
# tensorflow, as shown in https://github.com/esantamariavazquez/EEG-Inception. I simply implement in pytorch the
# input and output shapes to maintain uniformity with other networks. The class is called "Inception_EEG".


import torch.nn as nn
import torch
from thop import profile


class depthwise_separable_conv(nn.Module):
    def __init__(self, nin, nout, kernel_size):
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Conv2d(nin, nout, kernel_size=(1, kernel_size), padding=0, groups=nin)
        self.pointwise = nn.Conv2d(nout, nout, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class InceptionEEGNet_Block1(nn.Module):
    def __init__(self, kernel_size, num_channel=64):
        super(InceptionEEGNet_Block1, self).__init__()
        self.F = 8
        self.D = 2
        self.branch1 = nn.Sequential(
            nn.Conv2d(1, self.F, kernel_size=(1, kernel_size)),
            nn.ZeroPad2d((int(kernel_size / 2) - 1, int(kernel_size / 2), 0, 0)),
            nn.BatchNorm2d(self.F, False),
            nn.ELU(),
            nn.Dropout(p=0.25),
            nn.Conv2d(self.F, self.F * self.D, kernel_size=(num_channel, 1), groups=self.F),
            nn.BatchNorm2d(self.F * self.D, False),
            nn.ELU(),
            nn.Dropout(p=0.25)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(1, self.F, kernel_size=(1, int(kernel_size / 2))),
            nn.ZeroPad2d((int(kernel_size / 4) - 1, int(kernel_size / 4), 0, 0)),
            nn.BatchNorm2d(self.F, False),
            nn.ELU(),
            nn.Dropout(p=0.25),
            nn.Conv2d(self.F, self.F * self.D, kernel_size=(num_channel, 1), groups=self.F),
            nn.BatchNorm2d(self.F * self.D, False),
            nn.ELU(),
            nn.Dropout(p=0.25)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(1, self.F, kernel_size=(1, int(kernel_size / 4))),
            nn.ZeroPad2d((int(kernel_size / 8) - 1, int(kernel_size / 8), 0, 0)),
            nn.BatchNorm2d(self.F, False),
            nn.ELU(),
            nn.Dropout(p=0.25),
            nn.Conv2d(self.F, self.F * self.D, kernel_size=(num_channel, 1), groups=self.F),
            nn.BatchNorm2d(self.F * self.D, False),
            nn.ELU(),
            nn.Dropout(p=0.25)
        )
        self.branch_pool = nn.AvgPool2d(kernel_size=(1, 4))

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        N1 = torch.cat((branch1, branch2, branch3), dim=1)
        A1 = self.branch_pool(N1)
        return A1


class InceptionEEGNet_Block2(nn.Module):
    def __init__(self, kernel_size, num_channel=64):
        super(InceptionEEGNet_Block2, self).__init__()
        self.F = 8
        self.D = 2
        self.branch1 = nn.Sequential(
            nn.Conv2d(48, self.F, kernel_size=(1, int(kernel_size / 4))),
            nn.ZeroPad2d((int(kernel_size / 8) - 1, int(kernel_size / 8), 0, 0)),
            nn.BatchNorm2d(self.F, False),
            nn.ELU(),
            nn.Dropout(p=0.25)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(48, self.F, kernel_size=(1, int(kernel_size / 8))),
            nn.ZeroPad2d((int(int(kernel_size / 8) / 2) - 1, int(int(kernel_size / 8) / 2), 0, 0)),
            nn.BatchNorm2d(self.F, False),
            nn.ELU(),
            nn.Dropout(p=0.25)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(48, self.F, kernel_size=(1, int(kernel_size / 16))),
            nn.ZeroPad2d((int(int(kernel_size / 16) / 2), int(int(kernel_size / 16) / 2), 0, 0)),
            nn.BatchNorm2d(self.F, False),
            nn.ELU(),
            nn.Dropout(p=0.25)
        )
        self.branch_pool = nn.AvgPool2d(kernel_size=(1, 2))

    def forward(self, x):
        branch1 = self.branch1(x)
        # print(branch1.size())
        branch2 = self.branch2(x)
        # print(branch2.size())
        branch3 = self.branch3(x)
        # print(branch3.size())
        N2 = torch.cat((branch1, branch2, branch3), dim=1)
        A2 = self.branch_pool(N2)
        return A2


class Inception_EEG(nn.Module):

    def __init__(self, num_channel=25, num_class=4, len_window=480):
        super(Inception_EEG, self).__init__()
        # feature extractor
        self.__hidden_len__ = None
        self.kernel_size = 80
        self.F1 = 8
        self.D = 2
        self.F2 = 16
        self.num_channel = num_channel
        self.n_classes = num_class
        self.len_window = len_window
        self.feature = nn.Sequential()
        # (N,1,64,256)
        self.feature.add_module('f_block1', InceptionEEGNet_Block1(kernel_size=80, num_channel=self.num_channel))
        # (N,48,1,256/4)
        self.feature.add_module('f_block2', InceptionEEGNet_Block2(kernel_size=80, num_channel=self.num_channel))
        # (N,24,1,256/4/2)
        self.feature.add_module('f_conv3', nn.Conv2d(24, 12, kernel_size=(1, int(self.kernel_size / 8))))
        self.feature.add_module('f_padding3',
                                nn.ZeroPad2d((int(self.kernel_size / 16) - 1, int(self.kernel_size / 16), 0, 0)))
        self.feature.add_module('f_batchnorm3', nn.BatchNorm2d(12, False))
        self.feature.add_module('f_ELU3', nn.ELU())

        self.feature.add_module('f_dropout3', nn.Dropout(p=0.25))
        self.feature.add_module('f_pooling3',
                                nn.AvgPool2d(kernel_size=(1, 2)))
        # (N,12,1,256/4/2/2)
        self.feature.add_module('f_conv4', nn.Conv2d(12, 6, kernel_size=(1, int(self.kernel_size / 16))))
        self.feature.add_module('f_padding4',
                                nn.ZeroPad2d((int(self.kernel_size / 32), int(self.kernel_size / 32), 0, 0)))
        self.feature.add_module('f_batchnorm4', nn.BatchNorm2d(6, False))
        self.feature.add_module('f_ELU4', nn.ELU())
        self.feature.add_module('f_dropout4', nn.Dropout(p=0.25))
        self.feature.add_module('f_pooling4',
                                nn.AvgPool2d(kernel_size=(1, 2)))
        # (N,6,1,256/4/2/2/2)= (N,6,1,256/4/2/2/2)=48

        # class_classifier
        # 48- > 24
        # 24 - > 8
        # 8 - > 2
        __hidden_feature__ = self.feature(torch.rand(16, 1, self.num_channel, self.len_window))
        # print("__hidden_feature__:",__hidden_feature__.shape)
        self.__hidden_len__ = __hidden_feature__.shape[1] * __hidden_feature__.shape[2] * __hidden_feature__.shape[3]
        # print("__hidden_len__:",self.__hidden_len__)
        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(120, 64))  # 这个位置的隐藏输出维度有问题啊
        self.class_classifier.add_module('f_dropout1', nn.Dropout(p=0.25))
        self.class_classifier.add_module('c_bn1', nn.BatchNorm1d(64))
        self.class_classifier.add_module('c_fc2', nn.Linear(64, 32))
        self.class_classifier.add_module('f_dropout2', nn.Dropout(p=0.25))
        self.class_classifier.add_module('c_bn2', nn.BatchNorm1d(32))
        self.class_classifier.add_module('c_fc3', nn.Linear(32, self.n_classes))

    def forward(self, input_data):
        input_data.unsqueeze_(1)

        input_data = input_data.type(torch.cuda.FloatTensor)

        feature = self.feature(input_data)
        self.__hidden_len__ = feature.shape[1] * feature.shape[2] * feature.shape[3]
        feature = feature.view(-1, self.__hidden_len__)
        class_output = self.class_classifier(feature)
        return class_output


if __name__ == "__main__":
    net = Inception_EEG(num_channel=25, num_class=2, len_window=640).cuda()
    print(net)
    test = torch.rand(32, 25, 640)
    flops, params = profile(net, inputs=(test,), )
    print("Parameters:{0}, Flops:{1}".format(params, flops))
    output = net(test)
    print(output.size())
