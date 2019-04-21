import math
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, args):
        '''model_type: funnel or block'''
        super(MLP, self).__init__()
        hidden_dim = args.hidden_dim
        num_layers = args.num_layers
        model_type = args.model_type
        dropout = args.dropout

        hidden_dims = [hidden_dim for _ in range(num_layers)]
        if model_type == 'funnel':
            log_input_dim = int(math.log(input_dim, 2))
            log_output_dim = int(math.log(output_dim, 2))
            delta = (log_input_dim-log_output_dim)/(num_layers+1)
            log_hidden_dims = [log_input_dim-delta*(i+1) for i in range(num_layers)]
            hidden_dims = [int(math.pow(2, l)) for l in log_hidden_dims]
        elif model_type == 'triangle':
            delta = (input_dim-output_dim)/(num_layers+1)
            hidden_dims = [int(input_dim-delta*(i+1)) for i in range(num_layers)]
        dims = [input_dim]+hidden_dims
        self.fc_layers = nn.ModuleList([
                nn.Sequential(nn.Linear(dims[i], dims[i+1]), nn.Dropout(dropout), nn.ReLU()) \
            for i in range(num_layers)])
        self.output = nn.Linear(dims[-1], output_dim)

    def forward(self, x):
        for i, l in enumerate(self.fc_layers):
            x = self.fc_layers[i](x)
        x = self.output(x)
        return x


class Simple_LSTM(nn.Module):
    def __init__(self, input_dim, output_dim, args):
        super(Simple_LSTM, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = args.hidden_dim
        self.num_layers = args.num_layers
        self.bidirectional = args.bidirectional
        self.num_directions = 2 if self.bidirectional else 1

        self.seq_model = nn.LSTM(self.input_dim, self.hidden_dim, 
            self.num_layers, bidirectional=self.bidirectional, batch_first=True)

        self.output = nn.Linear(self.hidden_dim*self.num_directions, self.output_dim)

    def init_hidden(self, batch_size):
        return(Variable(torch.randn(self.num_layers*self.num_directions, batch_size, self.hidden_dim)),
            Variable(torch.randn(self.num_layers*self.num_directions, batch_size, self.hidden_dim)))

    def forward(self, x):
        '''x has shape (batch_size, seq_len, num_feats)'''
        o, (h, c) = self.seq_model(x, self.init_hidden(x.shape[0]))
        h = h.view(self.num_layers, self.num_directions, x.shape[0], self.hidden_dim)
        if self.bidirectional:
            h = torch.cat((h[-1, 0, :, :], h[-1, 1, :, :]), 1)
                # (batch_size, 2*hidden_dim)
        else:
            h = h[-1, 0, :, :]
        x = self.output(h)
        return x

class SoundClassifier(nn.Module):
    '''
    CNN based on SoundNet http://soundnet.csail.mit.edu/ 
    '''
    def __init__(self, input_dim, output_dim, args):
        super(SoundClassifier, self).__init__()
        self.globalpool = F.max_pool2d # avg_pool2d

        self.layer1 = nn.Sequential(nn.Conv2d(1, 16, (64, 1), stride=(2, 1), padding=(32, 0)), nn.BatchNorm2d(16), nn.ReLU())
        self.layer2 = nn.MaxPool2d((8, 1), stride=(8, 1))
        self.layer3 = nn.Sequential(nn.Conv2d(16, 32, (32, 1), stride=(2, 1), padding=(16, 0)), nn.BatchNorm2d(32), nn.ReLU())
        self.layer4 = nn.MaxPool2d((8, 1), stride=(8, 1))
        self.layer5 = nn.Sequential(nn.Conv2d(32, 64, (16, 1), stride=(2, 1), padding=(8, 0)), nn.BatchNorm2d(64), nn.ReLU())
        self.layer6 = nn.Sequential(nn.Conv2d(64, 128, (8, 1), stride=(2, 1), padding=(4, 0)), nn.BatchNorm2d(128), nn.ReLU())
        self.layer7 = nn.Sequential(nn.Conv2d(128, 256, (4, 1), stride=(2, 1), padding=(2, 0)), nn.BatchNorm2d(256), nn.ReLU())
        self.layer8 = nn.MaxPool2d((4, 1), stride=(4, 1))
        self.layer9 = nn.Sequential(nn.Conv2d(256, 512, (4, 1), stride=(2, 1), padding=(2, 0)), nn.BatchNorm2d(512), nn.ReLU())
        self.layer10 = nn.Sequential(nn.Conv2d(512, 1024, (4, 1), stride=(2, 1), padding=(2, 0)), nn.BatchNorm2d(1024), nn.ReLU())
        self.layer11 = nn.Sequential(nn.Linear(1024, 32), nn.ReLU())
        self.layer12 = nn.Linear(32, output_dim)

    def forward(self, x):
        x_p = torch.unsqueeze(x, 1)
        x_p = torch.unsqueeze(x_p, 3)
        out = self.layer1(x_p)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)
        out = self.layer10(out)
        out = self.globalpool(out, kernel_size=out.size()[2:])
        out = out.view(out.size(0),-1)
        out = self.layer11(out)
        out = self.layer12(out)
        return out