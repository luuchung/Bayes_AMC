import torch
import torch.utils.data as data
import numpy as np
from Freq.freq_model import Model
import pickle
from tqdm import tqdm
from torch.utils.data import Dataset
from torch import cat, zeros
from torch.autograd import grad
from tqdm import tqdm
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MyDataset(Dataset):
    def __init__(self, X, y):
        self.dat = torch.utils.data.TensorDataset(X,y)
    def __getitem__(self, index):
        data, target = self.dat[index]

        return data, target, index

    def __len__(self):
        return len(self.dat)

def exact_hessian(f, parameters, show_progress=False):
    params = list(parameters)
    if not all(p.requires_grad for p in params):
        raise ValueError("All parameters have to require_grad")
    df = grad(f, params, create_graph=True)
    # flatten all parameter gradients and concatenate into a vector
    dtheta = None
    for grad_f in df:
        dtheta = (
            grad_f.contiguous().view(-1)
            #grad_f.view(-1)
            if dtheta is None
            else cat([dtheta, grad_f.contiguous().view(-1)])
        )
    # compute second derivatives
    hessian_dim = dtheta.size(0)
    hessian = zeros(hessian_dim, hessian_dim)
    progressbar = tqdm(
        iterable=range(hessian_dim),
        total=hessian_dim,
        desc="[exact] Full Hessian",
        disable=(not show_progress),
    )
    for idx in progressbar:
        df2 = grad(dtheta[idx], params, create_graph=True)
        d2theta = None
        for d2 in df2:
            d2theta = (
                d2.contiguous().view(-1)
                #d2.view(-1)
                if d2theta is None
                else cat([d2theta, d2.contiguous().view(-1)])
            )
        hessian[idx] = d2theta
    f.backward()
    return hessian
classes = ['OOK','4ASK','8ASK','BPSK', 'QPSK', '8PSK', '16PSK', '32PSK', '16APSK', '32APSK', '64APSK', '128APSK', '16QAM', '32QAM', '64QAM',
               '128QAM', '256QAM', 'AM-SSB-WC', 'AM-SSB-SC', 'AM-DSB-WC', 'AM-DSB-SC', 'FM', 'GMSK']
print('.....Start loading data.....')
# Load data

X_train = torch.tensor(X_train, dtype=torch.float32)
Y_train = torch.tensor(Y_train, dtype=torch.float32)
#
#
#
Z_array = Z_train[:,0]
snrs = [-6, -4, -2, 0,  2,  4,  6,  8, 10, 12, 14]

test_snr_index = []
for snr in snrs:
    test_snr_index += list(np.where(Z_array == snr)[0])

X_train = X_train[test_snr_index]
Y_train = Y_train[test_snr_index]
print('Training data: ', X_train.shape)

train_loader = data.DataLoader(data.TensorDataset(X_train,Y_train), shuffle = True, batch_size = 200)

net = Model().to(device)
net.load_state_dict(torch.load('pre_trained_frequentist_model'))

W = list(net.parameters())[-2] # or -4 for the first layer
shape_W = W.shape
print(shape_W)
for inputs, labels in tqdm(train_loader):
    net.zero_grad()
    logits = net(inputs.to(device))
    loss = F.cross_entropy(logits, labels.to(device))
    Prec_post += exact_hessian(loss, [W])

b = list(net.parameters())[-1]# or -3 for the first layer
shape_b = b.shape
Prec_post_b = 0
for inputs, labels in tqdm(train_loader):
    net.zero_grad()
    logits = net(inputs.to(device))
    loss = F.cross_entropy(logits, labels.to(device))
    Prec_post_b += exact_hessian(loss, [b])


