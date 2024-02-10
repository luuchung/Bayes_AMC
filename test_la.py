import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np
import mltools
from Freq.freq_model import Model
import pickle
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.distributions.multivariate_normal import MultivariateNormal

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class MyDataset(Dataset):
    def __init__(self, X, y):
        self.dat = torch.utils.data.TensorDataset(X,y)
    def __getitem__(self, index):
        data, target = self.dat[index]

        return data, target, index

    def __len__(self):
        return len(self.dat)

classes = ['OOK','4ASK','8ASK','BPSK', 'QPSK', '8PSK', '16PSK', '32PSK', '16APSK', '32APSK', '64APSK', '128APSK', '16QAM', '32QAM', '64QAM',
               '128QAM', '256QAM', 'AM-SSB-WC', 'AM-SSB-SC', 'AM-DSB-WC', 'AM-DSB-SC', 'FM', 'GMSK']

criterion = nn.CrossEntropyLoss()
net = Model(hidden1 = 128, hidden2 = 64, classes = 23).to(device)
net.load_state_dict(torch.load('pre_trained_frequentist'))


var0 = 4.5e-2 #

W = list(net.parameters())[-2]
shape_W = W.shape
b = list(net.parameters())[-1]
shape_b = b.shape

W_first = list(net.parameters())[-4]
shape_W_first = W_first.shape
b_first = list(net.parameters())[-3]
shape_b_first = b_first.shape


Prec_post = torch.load('prec_post_sum.pt')
Prec_post_b = torch.load('prec_post_b.pt')

Prec_post_first = torch.load('prec_post_first_sum_20.pt')
Prec_post_b_first = torch.load('prec_post_b_first_sum.pt')

print('Hessian shape: ', Prec_post.shape)
print('Hessian bias shape: ', Prec_post_b.shape)

print('Hessian_first shape: ', Prec_post_first.shape)
print('Hessian_first bias shape: ', Prec_post_b_first.shape)

Cov_post = torch.inverse(1/var0 * torch.eye(W.numel()) + Prec_post)
Cov_post_b = torch.inverse(1/var0 * torch.eye(b.numel()) + Prec_post_b)

Cov_post_first = torch.inverse(1/var0 * torch.eye(W_first.numel()) + Prec_post_first)
Cov_post_b_first = torch.inverse(1/var0 * torch.eye(b_first.numel()) + Prec_post_b_first)

def predict_acc(x, model, n_samples=10):
    with torch.no_grad():
        model.eval()
        out, _ = model.lstm1(x)
        W_post = MultivariateNormal(W.flatten(), Cov_post.to(device))
        b_post = MultivariateNormal(b.flatten(), Cov_post_b.to(device))

        W_post_first = MultivariateNormal(W_first.flatten(), Cov_post_first.to(device))
        b_post_first = MultivariateNormal(b_first.flatten(), Cov_post_b_first.to(device))
        py = 0
        for i in range(n_samples):
            W_sample_first = W_post_first.rsample().view(shape_W_first)
            b_sample_first = b_post_first.rsample().view(shape_b_first)

            phi = out[:, -1, :] @ W_sample_first.t() + b_sample_first

            W_sample = W_post.rsample().view(shape_W)
            b_sample = b_post.rsample().view(shape_b)
            py += torch.softmax(phi @ W_sample.t() + b_sample, 1)
        py /= n_samples

    return py

# X_out = torch.load('out_data_oqpsk_1000_snr10.pt')
# index_data_out = torch.load('index_out_snr_oqpsk_1000_snr10.pt')
# Z_out = torch.load('snr_out_oqpsk_1000_snr30.pt')
# Z_out = Z_out[index_data_out]
# X_out = X_out[index_data_out]
# output_11 = np.zeros((1, 23))
# X_t = X_out[np.where(Z_out == 30)]
# with torch.no_grad():
#     #mean_predictions  = predict_batch(X_t.to(device), net, mean_W, mean_b, var_W, var_b, n_samples=20)
#     mean_predictions = predict_batch(X_t.to(device), net, n_samples=100)
#     print('Mean predictions shape: ', mean_predictions.shape)
#     mean_prob = mltools.count_average_prob(mean_predictions.detach().cpu().numpy())
# print(torch.mean(mean_predictions, 0))
# print(torch.tensor(mean_prob))

# out_hess = predict_batch(X_t.to(device), net, n_samples=100)
# torch.save(out_hess, 'Result/out_hess_1000_snr30.pt')

# max_len = 1040
# min_len = 16
# step = 32
# test_snr = 10
# seq_len = list(range(min_len, max_len+1, step))
# acc_length = {i: 0 for i in seq_len}
#
# with torch.no_grad():
#     for sequence in seq_len:
#         test_X_i = X_t[:, :sequence, :]
#         mean_predictions = predict_batch(test_X_i.to(device), net, n_samples=100)
#         acc_length[sequence] = round(np.mean(mean_predictions.detach().cpu().numpy().max(axis = 1)), 3)
#
# print(acc_length)

# snrs = [-20, -18, -16, -14, -12, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]
# sample_per_snr_out = [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
#         100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
# index_out = [1, 100,  200,  300,  400,  500,  600,  700,  800,  900, 1000,
#         1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100,
#         2200, 2300, 2400, 2500]
# Load data



X_test = X_test[test_index]
Y_test = Y_test[test_index]
Z_test = Z_test[test_index]

Z_array = Z_test[:,0]
snrs = sorted(list(set(Z_array)))
#

#
dataset = MyDataset(X_test, Y_test)
loader = DataLoader(dataset,
                    batch_size=2990,
                    shuffle=True)

Z_array = Z_test[:,0]
snrs = sorted(list(set(Z_array)))
out_batch = np.zeros((1, 23))
acc_batch = {i: 0 for i in snrs}
confidence = {i: 0 for i in snrs}
for snr in snrs:
    normalize = 0
    for data, target, index in loader:
        normalize += 1
        Z_snr = Z_array[index]
        X_test_snr = data[np.where(Z_snr == snr)]

        test_Y_hat = predict_acc(X_test_snr.to(device), net, n_samples=100)

        _, right, wrong = mltools.calculate_confusion_matrix(target[np.where(Z_snr == snr)].detach().cpu().numpy(), test_Y_hat.detach().cpu().numpy(), classes)

        acc_batch[snr] += round(1.0 * right / (right + wrong), 3)
        confidence[snr] +=  round(np.mean(test_Y_hat.detach().cpu().numpy().max(axis = 1)), 3)

    acc_batch[snr] = round(acc_batch[snr]/normalize, 3)
    confidence[snr] = round(confidence[snr]/normalize, 3)

print('Accuracy: ')
print(acc_batch)
print('Confidence: ')
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
        test_Y_hat = predict_batch(test_X_i.to(device), net,  n_samples=100)
        # estimate classes
        cm, right, wrong = mltools.calculate_confusion_matrix(target[np.where(Z_snr == test_snr)].detach().cpu().numpy(),test_Y_hat.detach().cpu().numpy(), classes)
        acc_batch[sequence] += round(1.0 * right / (right + wrong), 3)
        confidence[sequence] +=  round(np.mean(test_Y_hat.detach().cpu().numpy().max(axis = 1)), 3)
    acc_batch[sequence] = round(acc_batch[sequence]/normalize, 3)
    confidence[sequence] = round(confidence[sequence]/normalize, 3)

print('Accuracy: ')
print(acc_batch)
print('Confidence: ')
print(confidence)

