import torch
import torch.nn as nn


class ModelFusion(nn.Module):
    """
    shared classifier for doing audio/visual speech recognition
    """
    def __init__(self, config):
        super(ModelFusion, self).__init__()
        self.fc_1 = nn.Linear(12 * 256, 512)
        self.fc_2 = nn.Linear(512, 500)
        self.relu = nn.ReLU(True)
        self.sig = nn.Sigmoid()
        self.config = config
        self.dis = nn.Linear(512, 1)
        if not config.resume:
            self.fc_1.weight.data.normal_(0, 0.0001)
            self.fc_1.bias.data.zero_()

    def forward(self, x):
        x = x.view(-1, 12 * 256)
        net = self.fc_1(x)
        net0 = self.relu(net)
        net = self.fc_2(net0)
        #net0 = self.dropout(net)
        #dis_feature = self.sig(self.dis(net0))
        return net


class discriminator_audio(nn.Module):
    """
    discriminator for distinguishing whether Fv (and Fa) comes from video or audio
    """
    def __init__(self):
        super(discriminator_audio, self).__init__()
        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, 1)
        self.relu = nn.ReLU(True)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = x.view(-1, 256)
        net = self.fc1(x)
        net = self.fc2(self.relu(net))
        dis1 = self.sig(net)
        return dis1


class Face_ID_fc(nn.Module):
    """
    Identity classifier C_p
    """
    def __init__(self, config=config):
        super(Face_ID_fc, self).__init__()
        self.fc = nn.Linear(256, config.id_label_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x0 = x.view(-1, 256)
        net = self.dropout(x0)
        net = self.fc(net)
        return net