import torch

class LSTM(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers, bidirectional=False, num_epochs=100):
        super(LSTM, self).__init__()
        self.output = None
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_epochs = num_epochs
        self.lstm = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                                  bidirectional=bidirectional, batch_first=True)
        self.linear = torch.nn.Linear(hidden_size, output_size)
        self.bilinear = torch.nn.Linear(hidden_size * 2, output_size)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        # TODO: Implement LSTM forward pass
        b = 1
        if self.lstm.bidirectional:
            b = 2
        h0 = torch.randn(b * self.lstm.num_layers, x.shape[0], self.lstm.hidden_size)
        c0 = torch.randn(b * self.lstm.num_layers, x.shape[0], self.lstm.hidden_size)
        self.output, (self.hidden_layer, self.cell_layer) = self.lstm(x, (h0, c0))
        if self.lstm.bidirectional:
            self.output = self.bilinear(self.output)
        else:
            self.output = self.linear(self.output)
        self.output = self.sigmoid(self.output)
        return self.output