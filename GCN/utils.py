import torch
from torch.utils.data import Dataset
import numpy as np
import csv
import yaml

config = yaml.load(open('config.yml'))

#### Still need to be modified###
def get_snapshot_self(path, node_num):
    file = open(path, 'r', encoding='utf-8')
    snapshot = np.zeros(shape=(node_num, node_num, 6), dtype=np.float32)
    csv_reader_lines = csv.reader(file)
    for line in csv_reader_lines:
        node1 = int(line[0])
        node2 = int(line[7])
        for i in range(6):
            snapshot[node1, node2, i] = float(line[i+1])
    return snapshot

def get_snapshot(path, node_num):
    file = open(path, 'r', encoding='utf-8')
    snapshot = np.zeros(shape=(node_num, node_num), dtype=np.float32)
    # snaplist = [[[[0,0,0],[0,0,0],0] for _ in range(node_num)] for _ in range(node_num)]
    csv_reader_lines = csv.reader(file)
    select = config['select']
    for line in csv_reader_lines:
        node1 = int(line[0])
        node2 = int(line[7])
        edge = float(line[6])
        snapshot[node1, node2] = edge
        snapshot[node2, node1] = edge
    return snapshot

class LPDataset(Dataset):

    def __init__(self, path, window_size):
        super(LPDataset, self).__init__()
        self.data = torch.from_numpy(np.load(path))
        self.window_size = window_size
        self.num = self.data.size(0) - window_size

    def __len__(self):
        return self.num

    def __getitem__(self, item):
        return self.data[item: item + self.window_size], self.data[item + self.window_size]

def MSE(input, target):
    num = 1
    for s in input.size():
        num = num * s
    return (input - target).pow(2).sum().item() / num

def EdgeWiseKL(input, target):
    num = 1
    for s in input.size():
        num = num * s
    mask = (input > 0) & (target > 0)
    input = input.masked_select(mask)
    target = target.masked_select(mask)
    kl = (target * torch.log(target / input)).sum().item() / num
    return kl

def MissRate(input, target):
    num = 1
    for s in input.size():
        num = num * s
    mask1 = (input > 0) & (target == 0)
    mask2 = (input == 0) & (target > 0)
    mask = mask1 | mask2
    return mask.sum().item() / num