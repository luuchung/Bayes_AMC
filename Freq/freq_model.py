import torch.nn as nn
class Model(nn.Module):
  def __init__(self, features = 2, hidden1 = 128,hidden2 = 64, classes = 23, **kwargs):
    super(Model, self).__init__()
    self.features = features
    self.hidden1 = hidden1
    self.hidden2 = hidden2
    self.classes = classes

    self.lstm1 = nn.LSTM(input_size = self.features, hidden_size = self.hidden1, num_layers = 2, batch_first=True)
    self.linear1 = nn.Linear(self.hidden1, self.hidden2)
    self.linear2 = nn.Linear(self.hidden2, self.classes)

  def forward(self, x):
    x, _ = self.lstm1(x)
    x = self.linear1(x)
    x = self.linear2(x)
    return x