import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import numpy as np
import h5py
import mltools
from bnn_model import VI
from Freq.freq_model import Model
from tqdm import tqdm
import pickle
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


classes = ['OOK','4ASK','8ASK','BPSK', 'QPSK', '8PSK', '16PSK', '32PSK', '16APSK', '32APSK', '64APSK', '128APSK', '16QAM', '32QAM', '64QAM',
               '128QAM', '256QAM', 'AM-SSB-WC', 'AM-SSB-SC', 'AM-DSB-WC', 'AM-DSB-SC', 'FM', 'GMSK']
# Load data
# Load data

X_train = torch.tensor(X_train, dtype=torch.float32)
Y_train = torch.tensor(Y_train, dtype=torch.float32)

X_test = torch.tensor(X_val, dtype=torch.float32)
Y_test = torch.tensor(Y_val, dtype=torch.float32)

train_length = int(X_train.shape[1]/1)
print('The training length: ', train_length )

X_train = X_train[:, :train_length, :]
X_test = X_test[:, :train_length, :]

print('Training data full snrs: ', X_train.shape)

print('Testing data: ', X_test.shape)

Z_array = Z_train[:,0]
snrs = [-6, -4, -2, 0,  2,  4,  6,  8, 10, 12, 14]
test_snr_index = []
for snr in snrs:
    test_snr_index += list(np.where(Z_array == snr)[0])
X_train = X_train[test_snr_index]
Y_train = Y_train[test_snr_index]
print('Training data: ', X_train.shape)

train_loader = data.DataLoader(data.TensorDataset(X_train,Y_train), shuffle = True, batch_size = 1196)
test_loader = data.DataLoader(data.TensorDataset(X_test,Y_test), shuffle = True, batch_size = 598)

batch_test = 598
normalize_loss = len(X_test)/598
complexity_weight = 800
number_batch = int(len(X_train)/1196)
print('.....Finish loading data.....')


#net = VI(hidden1 = 128,hidden2 = 64, classes = 23).to(device)
#net.load_state_dict(torch.load('model_iq_signal_fm_out/vi_23classes_07102023_data2018_64last_sumall_pi5_epoch_10.pt'))

net = VI().to(device)
model_dict = net.state_dict()
pretrained = Model(hidden1 = 128, hidden2 = 64 ,classes = 23).to(device)
pretrained.load_state_dict(torch.load('..'))

pretrained_dict = pretrained.state_dict()
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

model_dict.update(pretrained_dict)
net.load_state_dict(model_dict)

#free lstm layers
params_freeze = ['lstm1.weight_ih_l0', 'lstm1.weight_hh_l0', 'lstm1.bias_ih_l0', 'lstm1.bias_hh_l0', 'lstm1.weight_ih_l1',
                 'lstm1.weight_hh_l1', 'lstm1.bias_ih_l1',  'lstm1.bias_hh_l1']
for name, param in net.named_parameters():
    # Set True only for params in the list 'params_to_train'
    param.requires_grad = False if name in params_freeze else True

for parameter in net.parameters():
    print(parameter.requires_grad)

criterion = nn.CrossEntropyLoss()
opt = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr = 1e-3)
print('.....Start training model.....')
n_epochs = 5
for epoch in range(n_epochs):
  print('Epoch: ', epoch)
  net.train()
  for batch_index, (x_batch, y_batch) in enumerate(train_loader):
    opt.zero_grad()

    pi_weight = mltools.minibatch_weight(batch_idx=batch_index, num_batches=number_batch)
    loss = net.sample_elbo(inputs=x_batch.to(device),
                               labels=y_batch.to(device),
                               criterion=criterion,
                               sample_nbr=2,
                               complexity_cost_weight=pi_weight)
    loss.backward()
    opt.step()
  if (epoch+1) % 2 == 0:
      #continue
    net.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
      for x_val_batch, y_val_batch in test_loader:

          confnorm, cor, ncor = mltools.calculate_confusion_matrix(y_val_batch,
                                                           net(x_val_batch.to(device))[:, -1, :].detach().cpu().numpy(), classes)
          correct += cor / (cor + ncor)
      print("Epoch: %d | Validation accuracy = %.4f" % (epoch, correct / normalize_loss))

