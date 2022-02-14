import yaml
import os
import csv
import numpy as np
import yaml
from utils import *

# load config
config = yaml.load(open('config.yml'))

# build path
base_path = os.path.join('./data/', config['dataset'])
train_base_path = os.path.join(base_path, 'train')
train_name = 'train'+'.npy'
test_name = 'test'+'.npy'
train_save_path = os.path.join(base_path, train_name)
test_save_path = os.path.join(base_path, test_name)

# load data
num = len(os.listdir(train_base_path)) - 1
# data = np.zeros(shape=(num, config['node_num'], config['node_num']), dtype=np.float32)
data = np.zeros(shape=(num, config['node_num'], config['node_num'], 6), dtype=np.float32)

# data = [[[[[0,0,0],[0,0,0],0] for _ in range(config['node_num'])] for _ in range(config['node_num'])] for _ in range(num)]


#### two snapshots need to be combined #####
for i in range(num):
    path = os.path.join(train_base_path, 'x_' + str(i) + '.csv')
    # data[i] = get_snapshot(path, config['node_num'])
    data[i] = get_snapshot_self(path, config['node_num'])
    # print(data)
    # print(self_data)

total_num = num - config['window_size']
test_num = int(config['test_rate'] * total_num)
train_num = total_num - test_num
train_data = data[0: train_num + config['window_size']]
test_data = data[train_num: num]

# save data
np.save(train_save_path, train_data)
np.save(test_save_path, test_data)