import torch
from torch import nn
import torch.nn.functional as F

import warnings
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Predictor modules
# ---------------------------------------------------------------------------

class ORNet(nn.Module):
    """Ordinal regression output layer."""
    def __init__(self, in_features, nclass, logit=True):
        super(ORNet, self).__init__()
        self.nclass = nclass
        self.in_features = in_features
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        biasm = torch.randn(self.nclass - 1)
        self.or_bias = torch.nn.Parameter(biasm.sort(descending=True)[0])
        self.fc1_bias = nn.Linear(self.in_features, 1, bias=False)
        self.logit = logit

    def forward(self, x):
        x = self.fc1_bias(x)
        out = []
        for i, wp_i in enumerate(range(self.nclass - 1)):
            temp = self.or_bias[i] + x
            if not self.logit:
                temp = torch.sigmoid(temp)
            out.append(temp)
        y = torch.cat(out, 1)
        return y

    def y2logits(self, y):
        probs = torch.sigmoid(y)
        current_device = y.device
        x = torch.cat((
            torch.ones((probs.shape[0], 1), device=current_device),
            probs,
            torch.zeros((probs.shape[0], 1), device=current_device)
        ), dim=1)
        logits = x[:, :-1] - x[:, 1:]
        return logits


class BottleneckLayer(nn.Module):
    def __init__(self, hidden_dim):
        super(BottleneckLayer, self).__init__()
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv1d(in_channels=hidden_dim, out_channels=128, kernel_size=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=32, kernel_size=3, padding="same")

    def forward(self, x):
        batch_size = x.size(0)
        if batch_size == 1 and self.training:
            bn1_training = self.bn1.training
            bn2_training = self.bn2.training
            self.bn1.eval()
            self.bn2.eval()
            x = self.conv1(self.relu1(self.bn1(x)))
            x = self.conv2(self.relu2(self.bn2(x)))
            self.bn1.training = bn1_training
            self.bn2.training = bn2_training
        else:
            x = self.conv1(self.relu1(self.bn1(x)))
            x = self.conv2(self.relu2(self.bn2(x)))
        return x


class DenseBlock(nn.Module):
    def __init__(self, nb_layers=6, hidden_dim=128):
        super(DenseBlock, self).__init__()
        self.nb_layers = nb_layers
        self.hidden_dim = hidden_dim
        self.bn = nn.ModuleList([BottleneckLayer(hidden_dim + 32 * i) for i in range(nb_layers)])

    def forward(self, x):
        layers_concat = []
        layers_concat.append(x)
        for i in range(self.nb_layers):
            x = torch.cat(layers_concat, dim=1)
            x = self.bn[i](x)
            layers_concat.append(x)
        x = torch.cat(layers_concat, dim=1)
        return x


class TransitionLayer(nn.Module):
    def __init__(self, input_dim):
        super(TransitionLayer, self).__init__()
        self.bn1 = nn.BatchNorm1d(input_dim)
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=int(input_dim * 0.5), kernel_size=1)
        self.pool1 = nn.AvgPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        batch_size = x.size(0)
        if batch_size == 1 and self.training:
            bn1_training = self.bn1.training
            self.bn1.eval()
            x = self.relu1(self.bn1(x))
            x = self.pool1(self.conv1(x))
            self.bn1.training = bn1_training
        else:
            x = self.relu1(self.bn1(x))
            x = self.pool1(self.conv1(x))
        return x


class BidirectionalLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input):
        """
        input : visual feature [batch_size x T x input_size]
        output : contextual feature [batch_size x T x output_size]
        """
        input = input.permute(0, 2, 1)
        self.rnn.flatten_parameters()
        recurrent, _ = self.rnn(input)
        output = self.linear(recurrent)
        output = output.permute(0, 2, 1)
        return output
