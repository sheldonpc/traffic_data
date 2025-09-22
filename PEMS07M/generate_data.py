import numpy as np
import pandas as pd


file_path = 'V_228.csv'
df=pd.read_csv(file_path)
traffic_speed = df.values
traffic_speed = np.concatenate([traffic_speed, traffic_speed[0:1, :]], axis=0)
np.savez('PEMS07M.npz', data=traffic_speed)
datas = np.load('PEMS07M.npz')
print(datas.files)
print(datas['data'])
print(datas['data'].shape, traffic_speed.shape)

