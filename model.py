import torch
import torch.nn as nn
import torch.nn.functional as F


# LSTM模型(旧)
# class LSTM(nn.Module):
#     def __init__(self, input_size=(train_X.shape[1]), hidden_layer_size=100, output_size=1):
#         super().__init__()
#         self.hidden_layer_size = hidden_layer_size
#
#         self.lstm = nn.LSTM(input_size, hidden_layer_size)
#
#         self.linear = nn.Linear(hidden_layer_size, output_size)
#
#         self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size),
#                             torch.zeros(1, 1, self.hidden_layer_size))
#
#     def forward(self, input_seq):
#         lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1), self.hidden_cell)
#         predictions = self.linear(lstm_out.view(len(input_seq), -1))
#         return predictions[-1]


# LSTM模型(confirmed)
class LSTMNet(nn.Module):

    def __init__(self, input_size):
        super(LSTMNet, self).__init__()
        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
        )
        self.out = nn.Sequential(
            nn.Linear(64, 1)
        )

    def forward(self, x):
        r_out, (h_n, h_c) = self.rnn(x, None)  # None 表示 hidden state 会用全0的 state
        out = self.out(r_out[:, -1, :])
        # print(out.shape)
        return out


# Bi-LSTM模型(confirmed)
class BiLSTMNet(nn.Module):

    def __init__(self, input_size):
        super(BiLSTMNet, self).__init__()
        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=50,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        self.out = nn.Sequential(
            nn.Linear(100, 1)
        )

    def forward(self, x):
        r_out, (h_n, h_c) = self.rnn(x, None)  # None 表示 hidden state 会用全0的 state
        out = self.out(r_out[:, -1, :])
        # print(out.shape)
        return out


# new model
# class RNN_BI(nn.Module):
#
#     def __init__(self, input_dim, hidden_dim=50, batch_size=1, output_dim=1, num_layers=2, rnn_type='LSTM'):
#         super(RNN_BI, self).__init__()
#         self.input_dim = input_dim
#         self.hidden_dim = hidden_dim
#         self.batch_size = batch_size
#         self.num_layers = num_layers
#
#         #Define the initial linear hidden layer
#         self.init_linear = nn.Linear(self.input_dim, self.input_dim)
#
#         # Define the LSTM layer
#         self.lstm = eval('nn.' + rnn_type)(self.input_dim, self.hidden_dim, self.num_layers, batch_first=True, bidirectional=True)
#
#         # Define the output layer
#         self.linear = nn.Linear(self.hidden_dim * 2, output_dim)
#
#     def init_hidden(self):
#         # This is what we'll initialise our hidden state as
#         return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
#                 torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))
#
#     def forward(self, input):
#         #Forward pass through initial hidden layer
#         linear_input = self.init_linear(input)
#
#         # Forward pass through LSTM layer
#         # shape of lstm_out: [batch_size, input_size ,hidden_dim]
#         # shape of self.hidden: (a, b), where a and b both
#         # have shape (batch_size, num_layers, hidden_dim).
#         lstm_out, self.hidden = self.lstm(linear_input)
#
#         # Can pass on the entirety of lstm_out to the next layer if it is a seq2seq prediction
#         y_pred = self.linear(lstm_out)
#         return y_pred

# BP神经网络
class Net_BP(torch.nn.Module):
    def __init__(self, n_features, n_hidden=50, n_output=1):
        # n_features输入层神经元数量，也就是特征数量
        # n_hidden隐层神经元数量
        # n_output输出层神经元数量
        super(Net_BP, self).__init__()
        self.hidden = torch.nn.Linear(n_features, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x


# RNN神经网络
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size=50, output_size=1, num_layers=1):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers)
        self.reg = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x, _ = self.rnn(x) # 未在不同序列中传递hidden_state
        return self.reg(x)

