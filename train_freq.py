import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import numpy as np
import h5py
import mltools
from freq_model import *
from tqdm import tqdm
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
classes = ['OOK','4ASK','8ASK','BPSK', 'QPSK', '8PSK', '16PSK', '32PSK', '16APSK', '32APSK', '64APSK', '128APSK', '16QAM', '32QAM', '64QAM',
               '128QAM', '256QAM', 'AM-SSB-WC', 'AM-SSB-SC', 'AM-DSB-WC', 'AM-DSB-SC', 'FM', 'GMSK']
# Load data
print('.....Start loading data.....')
# Load data

X_train = torch.tensor(X_train, dtype=torch.float32)
Y_train = torch.tensor(Y_train, dtype=torch.float32)

X_test = torch.tensor(X_val, dtype=torch.float32)
Y_test = torch.tensor(Y_val, dtype=torch.float32)



print('Training data full snrs: ', X_train.shape)

print('Testing data: ', X_test.shape)

Z_array = Z_train[:,0]
snrs = [-6, -4, -2, 0,  2,  4,  6,  8, 10, 12, 14]
print('Train snrs: ', snrs)
test_snr_index = []
for snr in snrs:
    test_snr_index += list(np.where(Z_array == snr)[0])
X_train = X_train[test_snr_index]
Y_train = Y_train[test_snr_index]
print('Training data: ', X_train.shape)

batch_train = 1196
batch_val = 598
train_loader = data.DataLoader(data.TensorDataset(X_train,Y_train), shuffle = True, batch_size = batch_train)
test_loader = data.DataLoader(data.TensorDataset(X_test,Y_test), shuffle = True, batch_size = batch_val)

batch_test = batch_val
normalize_loss = len(X_test)/batch_val
print('.....Finish loading data.....')

n_epochs = 200  # number of epochs to train on 50 epoch

net = Model(hidden1 = 128, classes = 23).to(device)
criterion = nn.CrossEntropyLoss()
opt = optim.Adam(net.parameters(), lr=0.001) # start with lr = 1e-3

print('.....Start training.....')
for epoch in range(n_epochs):
  net.train()
  for x_batch, y_batch in tqdm(train_loader):
    opt.zero_grad()
    y_hat = net(x_batch.to(device))
    loss = 0
    for i in range(y_hat.shape[1]):
         loss += criterion(y_hat[:, i, :], y_batch.to(device))
    #loss = criterion(y_hat, y_batch.to(device))
    loss.backward()
    opt.step()
  if (epoch+1) % 20 == 0:
    #continue
    net.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for x_val_batch, y_val_batch in test_loader:

            out_hat = net(x_val_batch.to(device))[:, -1, :]

            confnorm, cor, ncor = mltools.calculate_confusion_matrix(y_val_batch,
                                                               out_hat.detach().cpu().numpy(),
                                                               classes)

            correct += cor/(cor+ncor)
            val_loss = criterion(out_hat, y_val_batch.to(device))

    print("Epoch: %d |Training Loss = %.4f |Validation Loss = %.4f|Validation accuracy = %.4f" % (epoch, loss.item(), val_loss.item() ,correct / normalize_loss))






