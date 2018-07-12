import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size, hidden_size)
        self.i2o = nn.Linear(input_size, output_size)
        self.o2o = nn.Linear(hidden_size + output_size, output_size)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, category, input, hidden):
        input_combined = torch.cat((category, input, hidden), 1)
        hidden = self.i2h(input_combined)
        output = self.i2o(input_combined)
        output_combined = torch.cat((hidden, output), 1)
        output = self.o2o(output_combined)
        output = self.dropout(output)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

    criterion = nn.NLLLoss()

    learning_rate = 0.0005

    def train(category_tensor, input_line_tensor, target_line_tensor):
        target_line_tensor.unsqueeze_(-1)
        hidden = rnn.initHidden()

        rnn.zero_grad()

        loss = 0

        for i in range(input_line_tensor.size(0)):
            output, hidden = rnn(category_tensor, input_line_tensor[i], hidden)
            l = criterion(output, target_line_tensor[i])
            loss += l

        loss.backward()

        for p in rnn.parameters():
            p.data.add_(-learning_rate, p.grad.data)

        return output, loss.item() / input_line_tensor.size(0)