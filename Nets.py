import math

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence

import utils


class CNN_MLP(nn.Module):
    def __init__(self, args):
        super(CNN_MLP, self).__init__()
        kernel_size = args.cnn_kernel_size
        self.CNN = nn.Sequential(nn.Conv1d(1, 64, kernel_size=kernel_size, stride=1),
                                 nn.BatchNorm1d(64),
                                 nn.GELU(),
                                 nn.MaxPool1d(
                                     kernel_size=kernel_size, stride=1),
                                 nn.Conv1d(
                                     64, 32, kernel_size=kernel_size, stride=1),
                                 nn.BatchNorm1d(32),
                                 nn.GELU(),
                                 nn.MaxPool1d(
                                     kernel_size=kernel_size, stride=1),
                                 nn.Conv1d(
                                     32, 16, kernel_size=kernel_size, stride=1),
                                 nn.BatchNorm1d(16),
                                 nn.GELU(),
                                 nn.MaxPool1d(kernel_size=kernel_size, stride=1))

        dim = args.dim
        self.classifier = nn.Sequential(nn.Linear(dim, dim // 2),
                                        nn.GELU(),
                                        nn.Dropout(0.5),
                                        nn.Linear(dim // 2, dim // 4),
                                        nn.GELU(),
                                        nn.Dropout(0.5),
                                        nn.Linear(dim // 4, 2))

    def forward(self, features):
        features = torch.unsqueeze(features, 1)
        features = self.CNN(features)
        features = features.view(features.size(0), -1)
        pred = self.classifier(features)
        pred = F.softmax(pred, dim=1)
        return pred

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                # nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class BiLSTM_Kernel(nn.Module):
    def __init__(self, args):
        super(BiLSTM_Kernel, self).__init__()
        self.args = args
        self.lstm_forward = nn.GRU(
            input_size=args.embed_num,
            hidden_size=args.hidden_dim,
            bidirectional=False,
            num_layers=1)
        self.lstm_backward = nn.GRU(
            input_size=args.embed_num,
            hidden_size=args.hidden_dim,
            bidirectional=False,
            num_layers=args.num_layers)
        self.dropout = nn.Dropout(p=self.args.dropout)

    def forward(self, code):
        code_embedded, code_rev_embedded = code
        lstm_forward_out, _ = self.lstm_forward(code_embedded)
        lstm_backward_out, _ = self.lstm_backward(code_rev_embedded)
        bilstm_out = torch.cat((lstm_forward_out, lstm_backward_out), dim=2)
        return bilstm_out

    def initialize(self):
        nn.init.orthogonal_(self.lstm_forward.weight_ih_l0)
        nn.init.orthogonal_(self.lstm_forward.weight_hh_l0)
        nn.init.orthogonal_(self.lstm_backward.weight_ih_l0)
        nn.init.orthogonal_(self.lstm_backward.weight_hh_l0)

        nn.init.zeros_(self.lstm_forward.bias_ih_l0)
        nn.init.ones_(
            self.lstm_forward.bias_ih_l0[self.args.hidden_dim:self.args.hidden_dim * 2])
        nn.init.zeros_(self.lstm_forward.bias_hh_l0)
        nn.init.ones_(
            self.lstm_forward.bias_hh_l0[self.args.hidden_dim:self.args.hidden_dim * 2])

        nn.init.zeros_(self.lstm_backward.bias_ih_l0)
        nn.init.ones_(
            self.lstm_backward.bias_ih_l0[self.args.hidden_dim:self.args.hidden_dim * 2])
        nn.init.zeros_(self.lstm_backward.bias_hh_l0)
        nn.init.ones_(
            self.lstm_backward.bias_hh_l0[self.args.hidden_dim:self.args.hidden_dim * 2])


class CNN_2d_Kernel(nn.Module):
    def __init__(self, args):
        super(CNN_2d_Kernel, self).__init__()
        self.args = args
        self.conv = nn.Conv2d(in_channels=1,
                              out_channels=self.args.cnn_out_channels,
                              kernel_size=(self.args.embed_num,
                                           self.args.kernel_size),
                              stride=self.args.cnn_stride)
        self.dropout = nn.Dropout(p=self.args.dropout)

    def forward(self, embedded):
        embedded = torch.transpose(embedded, 0, 1)
        embedded = embedded.unsqueeze(0)
        embedded = embedded.unsqueeze(0)
        embedded = self.conv(embedded)
        embedded = torch.transpose(embedded, 1, 2)
        embedded = torch.transpose(embedded, 2, 3)
        embedded = embedded.squeeze(0)
        embedded = embedded.squeeze(0)
        return embedded


class CNN_BiLSTM_Kernel(nn.Module):
    def __init__(self, args):
        super(CNN_BiLSTM_Kernel, self).__init__()
        self.args = args
        self.bilstm = BiLSTM_Kernel(args)
        self.cnn = CNN_2d_Kernel(args)
        self.cnn_rev = CNN_2d_Kernel(args)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, embedded):
        features = [self.cnn(self.dropout(i)) for i in embedded]
        features_rev = [torch.flip(i, dims=[0]) for i in features]

        features = pad_sequence(features)
        features_rev = pad_sequence(features_rev)
        bilstm_out = self.bilstm((features, features_rev))
        lengths = [int((i.shape[0] - self.args.kernel_size) /
                       self.args.cnn_stride) for i in embedded]
        output = torch.sum(bilstm_out, dim=0)
        output = output.transpose(0, 1)
        output = output / torch.tensor(lengths).to(self.args.device)
        output = output.transpose(0, 1)

        return output

    def initialize(self):
        self.bilstm.initialize()


class CNN_BiLSTM(nn.Module):
    def __init__(self, args):
        super(CNN_BiLSTM, self).__init__()
        self.args = args
        self.cnn_bilstm = CNN_BiLSTM_Kernel(args)

        self.embedding = nn.Embedding.from_pretrained(
            utils.embed_from_pretrained(args),
            freeze=self.args.freeze_embed
        )

    def forward(self, seqs):
        embedded = [self.embedding(i.to(self.args.device)) for i in seqs]
        output = self.cnn_bilstm(embedded)
        return output

    def initialize(self):
        self.cnn_bilstm.initialize()


class Linear_Classifier(nn.Module):
    def __init__(self, args):
        super(Linear_Classifier, self).__init__()
        self.args = args
        self.fc1 = nn.Linear(2 * args.hidden_dim,
                             args.hidden_dim)
        self.fc3 = nn.Linear(args.hidden_dim,
                             args.hidden_dim // 2)
        self.fc4 = nn.Linear(args.hidden_dim // 2,
                             2)
        self.dropout = nn.Dropout(p=self.args.dropout_LC)

    def forward(self, features):
        output = F.gelu(self.fc1(features))
        output = self.dropout(output)
        output = F.gelu(self.fc3(output))
        output = self.fc4(output)
        return output


class CNN_GRU(nn.Module):
    def __init__(self, args):
        super(CNN_GRU, self).__init__()
        self.args = args
        self.cnn_bilstm = CNN_BiLSTM(args)
        self.classifier = Linear_Classifier(args)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, seqs):
        features = self.cnn_bilstm(seqs)
        features = self.dropout(features)
        pred = self.classifier(features)
        pred = F.softmax(pred, dim=1)
        return pred

    def initialize(self):
        self.cnn_bilstm.initialize()


class en_model(nn.Module):
    def __init__(self, args):
        super(en_model, self).__init__()
        self.args = args
        self.CNN = CNN_MLP(args)
        self.GRU = CNN_GRU(args)
        self.w1 = nn.Parameter(data=torch.tensor(1.0), requires_grad=True)
        self.w2 = nn.Parameter(data=torch.tensor(1.0), requires_grad=True)
        self.w3 = nn.Parameter(data=torch.tensor(1.0), requires_grad=True)
        self.w4 = nn.Parameter(data=torch.tensor(1.0), requires_grad=True)
        self.classifier = nn.Linear(4, 2)

    def forward(self, inputcode, feature):
        features1 = self.CNN(feature)
        features2 = self.GRU(inputcode)
        #
        pred1 = self.w1 * features1[:, 0] + self.w2 * features2[:, 0]
        pred2 = self.w3 * features1[:, 1] + self.w4 * features2[:, 1]
        pred1 = torch.unsqueeze(pred1, dim=1)
        pred2 = torch.unsqueeze(pred2, dim=1)
        pred = torch.cat((pred1, pred2), dim=1)
        pred = F.softmax(pred, dim=1)
        return pred

    def initialize(self):
        self.CNN.initialize()
        self.GRU.initialize()


class LDAMLoss(nn.Module):

    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30, device='cuda:0'):
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.FloatTensor(m_list).to(device)
        # m_list = torch.FloatTensor(m_list)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight
        self.device = device

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)
        index_float = index.type(torch.FloatTensor).to(self.device)
        batch_m = torch.matmul(self.m_list[None, :].to(self.device), index_float.transpose(0, 1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m
        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.s * output, target, weight=self.weight)
