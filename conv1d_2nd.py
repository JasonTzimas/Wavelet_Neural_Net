import pywt as wt
import numpy as np
import torch
import pandas as pd
from matplotlib import pyplot as plt
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


w = wt.Wavelet('db2')
dec_lo, dec_hi, rec_lo, rec_hi = w.filter_bank

def WaveletLoss(ho, go, energy_coefficients):
    # Define loss function members
    L1 = (torch.sum(ho) - np.sqrt(2))**2   # Loss for S(H) = sqrt(2)
    L1 = L1 + (torch.sum(go)**2)              # Loss for S(G) = 0
    hb = torch.cat((ho, torch.zeros(1, 1, ho.shape[2] - 2)), 2)
    opts = torch.zeros(1, 1, int(ho.shape[2]/2))
    opts[0, 0, 0] = 1
    conv = F.conv1d(hb, ho)
    conv = conv[:, :, 0::2]
    L1 = L1 + torch.sum((conv - opts)**2)      # Loss for H orthnormality
    gb = torch.cat((go, torch.zeros(1, 1, go.shape[2] - 2)), 2)
    conv = F.conv1d(gb, go)
    conv = conv[:, :, 0::2]
    L1 = L1 + torch.sum((conv - opts)**2)      # Loss for G orthonormality
    conv = F.conv1d(gb, ho)
    conv = conv[:, :, 0:2]
    opts = torch.zeros(1, 1, conv.shape[2])
    L1 = L1 + (torch.sum((conv - opts)**2))

    # Gain for quadratic orthonormality constraints
    K1 = 10
    L1 = K1*L1

    # Wavelet Energy Entropy

    ### 111 ### Total Energy calculation
    total_energy = torch.sum(energy_coefficients**2, 2, keepdim=True)
    print(energy_coefficients.shape)
    print(total_energy.shape)
    energy_coefficients = energy_coefficients / total_energy
    print(energy_coefficients.shape)
    logj = torch.log(energy_coefficients)
    print(logj.shape)
    indiv_entropy = energy_coefficients * logj
    print(indiv_entropy.shape)
    s = - torch.sum(indiv_entropy, 2, keepdim=True)
    print(s.shape)
    L2 = (torch.mean(s, 0)**2)[0, 0]
    print(L2.shape)
    # Gain for entropy loss term
    K2 = 1
    L2 = K2*L2

    # Adding the terms together
    L = L1 + L2

    return L




class TimeSeries(Dataset):
    def __init__(self, datain, window):
        #Data loading
        self.window = window
        self.dat = torch.reshape(torch.from_numpy(datain), (1, datain.shape[0]))
        self.shape = self.__getshape__()

    def __getitem__(self, index):
        return self.dat[[0], index:index+1024]

    def __len__(self):
        return self.dat.shape[1] - self.window

    def __getshape__(self):
        return self.dat.shape


class WaveletModel(torch.nn.Module):
    def __init__(self, filter_length, levels, batch_size, data_length):
        super().__init__()
        # Initialize decomposition and reconstruction filters and the parameters of the model
        #self.ho = torch.randn((1, 1, filter_length), requires_grad=True, dtype=torch.float64)
        self.ho = torch.randn((1, 1, filter_length), requires_grad=True, dtype=torch.float64)
        self.go = torch.randn((1, 1, filter_length), requires_grad=False, dtype=torch.float64)
        self.Ho = torch.randn((1, 1, filter_length), requires_grad=False, dtype=torch.float64)
        self.Go = torch.randn((1, 1, filter_length), requires_grad=False, dtype=torch.float64)

        #self.Ho = torch.randn((1, 1, filter_length), requires_grad=True, dtype=torch.float64)
        self.levels = levels
        self.filter_length = filter_length
        self.batch_size = batch_size
        self.data_length = data_length

        twos = 2*torch.ones(self.levels)
        c = torch.arange(self.levels)
        a = twos**c
        # self.approxs = torch.empty(self.batch_size, 1, int((data_length*torch.sum(a))/2**(self.levels)))
        # self.details = torch.empty(self.batch_size, 1, int((data_length*torch.sum(a))/2**(self.levels)))
        # print(self.approxs.shape, self.details.shape)
        self.register_parameter(name='ho_filter', param=torch.nn.Parameter(self.ho))

    def UpdateFilters(self):
        # Define the rest of the filters
        self.Ho = torch.flip(self.ho, [0, 1, 2])
        self.go = torch.flip(self.ho, [0, 1, 2])
        for id, el in enumerate(self.go[0, 0, :]):
            self.go[-1, -1, id] = (-1)**(id+1)*self.go[-1, -1, id]

        self.Go = torch.flip(self.go, [0, 1, 2])
    def forward(self, x):
        self.UpdateFilters()
        pos = 0
        for i in range(self.levels):
            # Appropriately pad data
            if self.filter_length % 2 == 0:
                x = torch.cat((x[:, :, int(-self.filter_length/2):], x, x[:, :, :int(self.filter_length/2 - 1)]), 2)
            else:
                x = torch.cat((x[:, :, int(-self.filter_length//2):], x, x[:, :, :int(self.filter_length//2)]), 2)

            #print(i)
            approx = F.conv1d(x, self.ho)
            approx = approx[:, :, 1::2]
            detail = F.conv1d(x, self.go)
            detail = detail[:, :, 1::2]
            x = approx
            len = approx.shape[2]
            if i==0:
                approxs = approx
                details = detail
            else:
                approxs = torch.cat((approxs, approx), 2)
                details = torch.cat((details, detail), 2)
            pos = pos + self.data_length/(2**(i+1))

        # For loop initial values
        size = int(self.data_length/(2**self.levels))
        appr_loop = approxs[:, :, -int(size):]
        # Do reconstruction
        two_powers = size
        print(size)
        first = True
        for i in range(self.levels):
            # Data padding
            an_rec = torch.zeros((self.batch_size, 1, 2*two_powers), dtype=torch.float64)
            an_rec[:, :, 0::2] = appr_loop
            dn_rec = torch.zeros((self.batch_size, 1, 2*two_powers), dtype=torch.float64)
            #print(torch.FloatTensor(details[0]))
            if first == True:
                dn_rec[:, :, 0::2] = details[:, :, -int(size):]
                first = False
            else:
                dn_rec[:, :, 0::2] = details[:, :, -int(size):int(-size+two_powers)]

            # Periodic padding
            n = self.filter_length
            if n % 2 == 0:
                dn_rec = torch.cat((dn_rec[:, :, int(-n / 2):], dn_rec, dn_rec[:, :, :int(n / 2 - 1)]), 2)
                an_rec = torch.cat((an_rec[:, :, int(-n / 2):], an_rec, an_rec[:, :, :int(n / 2 - 1)]), 2)
            else:
                dn_rec = torch.cat((dn_rec[:, :, int(-n // 2):], dn_rec, dn_rec[:, :, :int(n // 2)]), 2)
                an_rec = torch.cat((an_rec[:, :, int(-n // 2):], an_rec, an_rec[:, :, :int(n // 2 - 1)]), 2)

            # Reconstruction
            an_rec = F.conv1d(an_rec, self.Ho)
            dn_rec = F.conv1d(dn_rec, self.Go)
            appr_loop = an_rec + dn_rec
            size += 2*two_powers
            two_powers *= 2
            print(size, two_powers)
        # Save the final reconstructed signal
        reconstructed_signal = appr_loop
        energy_coeffs = torch.cat((details, approxs[:, :, -int(self.data_length/(2**self.levels)):]), 2)
        return energy_coeffs, reconstructed_signal


# Data loading and formatting
data = np.loadtxt('data/emg/imu_emg_medium_2.dat', usecols=1,  delimiter=',', dtype=np.float64)
dat = data.T
window = 1024
dat = dat[0:1024]
#print(dat.shape)

dat = torch.from_numpy(dat)
dat = torch.reshape((dat), (1, 1, dat.shape[0]))
#print(dat.shape)

# Dataset creation using custom TimeSeries class
dataset = TimeSeries(data.T, 1024)
train_dl = DataLoader(dataset, batch_size=4, shuffle=True)
iterator = iter(train_dl)
batch = next(iterator)

# Train network
model = WaveletModel(4, 5, batch.shape[0], batch.shape[2])

optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
EPOCHS = 2

for epoch in range(EPOCHS):
    for batch in train_dl:
        energy_coeffs, reconstructed_signal = model(batch)
        loss = WaveletLoss(model.ho, model.go, energy_coeffs)
        loss.backward(retain_graph=False)
        optimizer.step()
        model.ho.grad.zero_()


    print(loss)

