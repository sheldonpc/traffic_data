import numpy as np
import pandas as pd

year = '2017'  # please specify the year, our experiments use 2019

gba_his = pd.read_hdf('gba_his_' + year +'.h5')
value = gba_his.values
value = value[:(31+28)*288, :]
print(value.shape, value[value==0].shape)

# np.savez('CA_District5.npz', data=value)
datas = np.load('CA_District5.npz')
print(datas.files)
print(datas['data'])
print(datas['data'].shape)

adj = np.load('gba_rn_adj.npy')
print(adj)
print(adj.shape, adj.dtype)