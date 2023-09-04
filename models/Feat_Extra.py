import torch.nn as nn

def init_weights(m):
    if ((type(m) == nn.Linear) and (not m.bias is None)):
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

class Feat_Extra(nn.Module):
    def __init__(self, in_channel=1, out_channel=128,**args):
        super(Feat_Extra, self).__init__()

        self.dropout_ratio = args.get('dropout_ratio', 0.3)
        channel_list = [in_channel, 512, out_channel]

        # fully connected layer
        self.nn = []
        for idx, num in enumerate(channel_list[:-1]):
            self.nn.append(nn.Linear(channel_list[idx], channel_list[idx + 1]))
            self.nn.append(nn.BatchNorm1d(channel_list[idx + 1]))
            if self.dropout_ratio >= 0:
                self.nn.append(nn.Dropout(self.dropout_ratio))
            self.nn.append(nn.ReLU())

        self.proj_head = nn.Sequential(*self.nn)
        self.reset_parameters()

    def reset_parameters(self, ):
        self.proj_head.apply(init_weights)

    def forward(self, x):
        x = self.proj_head(x)
        return x