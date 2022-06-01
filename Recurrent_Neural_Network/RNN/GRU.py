"""
GRU(gated recurrent unit)网络
"""
import torch.nn as nn
import torch

class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GRUCell, self).__init__()
        self.hidden_size = hidden_size
        self.gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        z_gate = self.sigmoid(self.gate(combined))
        r_gate = self.sigmoid(self.gate(combined))
        combined01 = torch.cat((input, torch.mul(hidden, r_gate)), 1)
        h1_state = self.tanh(self.gate(combined01))

        h_state = torch.add(torch.mul((1 - z_gate), hidden), torch.mul(h1_state, z_gate))
        output = self.output(h_state)
        output = self.softmax(output)
        return output, h_state

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

grucell = GRUCell(input_size=10,hidden_size=20,output_size=10)
input=torch.randn(32,10)
h_0=torch.randn(32,20)

output,hn=grucell(input,h_0)
print(output.size(),hn.size())
