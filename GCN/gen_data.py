import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import csv
import math
import yaml
import os
import pandas as pd


# Gaus function
def gaus(mu, sigma, no):
    num = np.random.normal(mu, sigma, no)
    return num

# Generate common data
def gen_theta(length, wrange, theta, no):
    data = [0 for _ in range(length)]
    for index in range(0,length):
        if index == 0:
            temp = (gaus(0,0.25,no))*2*math.pi
            data[0] = temp.item()
        else:
            temp = (gaus(0,0.25,no))*theta
            data[index] = data[index-1] + wrange*temp.item()
    return data

# Generate alpha and beta 
def albe(length):
    ab = np.empty((2,length))
    alpha = np.array(gen_theta(length, 1, math.pi/3, 1))
    beta = np.array(gen_theta(length, 1, math.pi/3, 1))
    ab[0, :] = alpha[:]
    ab[1, :] = beta[:]
    return ab

# Generate yaw pitch and roll
def tri(length):
    three = np.empty((3, length))
    yaw = [np.array(gen_theta(1, 1, 0, 1))/36 for _ in range(length)]
    pitch = [np.array(gen_theta(1, 1, 0, 1))/8 for _ in range(length)]
    roll = [np.array(gen_theta(1, 1, 0, 1))/18 for _ in range(length)]
    three[0, :] = yaw[:]
    three[1, :] = pitch[:]
    three[2, :] = roll[:]
    return three

# Generate lines
def gen_line(length, albe, index):
    temp = gaus(0,0.25,3)
    line_data = np.empty((3, length))
    line_data[0, 0] = 20*(temp[0])
    line_data[1, 0] = 20*(temp[1])
    line_data[2, 0] = 20*(temp[2])
    for leng in range(1, length):
        step = [np.cos(albe[index][0][leng-1]), np.sin(albe[index][0][leng-1]), np.cos(albe[index][1][leng-1])]
        line_data[:, leng] = line_data[:, leng-1] + step
    return line_data

# Calculate Distance from X and Y no Z
def DIS(data, j, k, i):
    dis = np.sqrt(np.square(data[j][0][i]-data[k][0][i])+np.square(data[j][1][i]-data[k][1][i]))
    return dis

# Calculate Slope 
def SLO(data, j, k, i):
    slo = array([data[j][0][i]-data[k][0][i], data[j][1][i]-data[k][1][i], data[j][2][i]-data[k][2][i]])
    return slo

# Check repeat
def REP(slope, nodes, k):
    flag = {}
    for j in range(nodes-1):
        for k in range(j+1, nodes-1):
            for a in range(k+1, nodes):
                flag.append = slope[j][k]/slope[j][a]
    flag = data
    return flag

# Generate SNR
def SNR_gaus(dis):
    snr = float(gaus(1,0.1,1)*dis + gaus(0,0.1,1))
    return snr

# Update lines
def update_lines(num, data_lines, lines):
    for line, data in zip(lines, data_lines):
        line.set_data(data[0:2, :num])
        line.set_3d_properties(data[2, :num])
    return lines
