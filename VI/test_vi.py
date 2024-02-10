import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from bnn_model import *
import mltools
from tqdm import tqdm
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device: ', device)
print('.....Start loading data.....')
# Load data


class MyDataset(Dataset):
    def __init__(self, X, y):
        self.dat = torch.utils.data.TensorDataset(X,y)
    def __getitem__(self, index):
        data, target = self.dat[index]
        return data, target, index

    def __len__(self):
        return len(self.dat)

X_test = torch.load('')
Y_test = torch.load('')
Z_test = torch.load('')

dataset = MyDataset(X_test, Y_test)
loader = DataLoader(dataset,
                    batch_size=2990,
                    shuffle=True)

classes = ['OOK','4ASK','8ASK','BPSK', 'QPSK', '8PSK', '16PSK', '32PSK', '16APSK', '32APSK', '64APSK', '128APSK', '16QAM', '32QAM', '64QAM',
               '128QAM', '256QAM', 'AM-SSB-WC', 'AM-SSB-SC', 'AM-DSB-WC', 'AM-DSB-SC', 'FM', 'GMSK']



net = VI(hidden1 = 128,hidden2 = 64, classes = 23).to(device)
#net.load_state_dict(torch.load('...'))

Z_array = Z_test[:,0]
snrs = sorted(list(set(Z_array)))
out_batch = np.zeros((1, 23))
acc_batch = {i: 0 for i in snrs}
confidence = {i: 0 for i in snrs}
num_test = 1

def out_average(model, data_snr, num_test):
    test_Y_hat = 0
    for i in range(num_test):
        test_Y_hat +=  model(data_snr[:, :, :].to(device))[:, -1, :]
        #test_Y_hat += model(data_snr[:, :, :].to(device))
    return test_Y_hat/num_test

for snr in snrs:
    print('SNR: ', snr)
    normalize = 0
    for data, target, index in loader:
            normalize += 1
            Z_snr = Z_array[index]
            X_test_snr = data[np.where(Z_snr == snr)]
            test_Y_hat = out_average(net, X_test_snr, num_test = 100)
            _, right, wrong = mltools.calculate_confusion_matrix(target[np.where(Z_snr == snr)].detach().cpu().numpy(), test_Y_hat.detach().cpu().numpy(), classes)

            acc_batch[snr] += round(1.0 * right / (right + wrong), 3)
            confidence[snr] +=  round(np.mean(torch.softmax(test_Y_hat, dim=1).detach().cpu().numpy().max(axis = 1)), 3)

    acc_batch[snr] = round(acc_batch[snr]/normalize, 3)
    confidence[snr] = round(confidence[snr]/normalize, 3)


print('Accuracy snr: ')
print(acc_batch)
print('Confidence snr: ')
print(confidence)

max_len = 1040
min_len = 16
step = 32
test_snr = 10
acc_length = {}
seq_len = list(range(min_len, max_len+1, step))
confidence_vi = []
output_11 = np.zeros((1, 23))
con_11 = np.zeros((1, 23))
acc_batch = {i: 0 for i in seq_len}
confidence = {i: 0 for i in seq_len}

for sequence in seq_len:
    normalize = 0
    for data, target, index in loader:
        normalize += 1
        Z_snr = Z_array[index]
        X_test_snr = data[np.where(Z_snr == test_snr)]
        test_X_i = X_test_snr[:, :sequence, :]
        test_Y_hat = out_average(net, test_X_i, num_test=100)
        # estimate classes
        cm, right, wrong = mltools.calculate_confusion_matrix(target[np.where(Z_snr == test_snr)].detach().cpu().numpy(),test_Y_hat.detach().cpu().numpy(), classes)
        acc_batch[sequence] += round(1.0 * right / (right + wrong), 3)
        confidence[sequence] +=  round(np.mean(torch.softmax(test_Y_hat, dim=1).detach().cpu().numpy().max(axis = 1)), 3)
    acc_batch[sequence] = round(acc_batch[sequence]/normalize, 3)
    confidence[sequence] = round(confidence[sequence]/normalize, 3)

print('Accuracy length: ')
print(acc_batch)
print('Confidence length: ')
print(confidence)

