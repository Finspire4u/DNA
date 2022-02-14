import numpy as np
import csv
import math
import yaml
import os
import pandas as pd

# find data
base_path = './data/P'
raw_base_path = os.path.join(base_path, 'raw')
nodes = 5
length = 21
antenna_dis = 20


def DIS(a,b,x,y):
	res = np.sqrt(np.square(a-x) + np.square(b-y))
	return res

def antenna_gain(a):
	res = round(abs(a/antenna_dis),2)
	return res 

# create the antenna open and close labels based on the angles
def angle(x,y,a,b):
	antenna = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[1,1,0,0],[0,1,1,0],[0,0,1,1],[1,0,0,1]]
	x0 = a - x
	y0 = b - y
	if x0 == 0:
		if y0 > 0:
			return [antenna[1],[0,antenna_gain(y0),0,0]]
		return [antenna[3],[0,0,0,antenna_gain(y0)]]
	angle = (np.arctan(y0/x0) + math.pi/2)*360/math.pi/2

	if y0 >= 0:
		if 0 <= angle <= 10:
			return [antenna[0],[antenna_gain(x0),0,0,0]]
		elif 10 < angle < 80:
			return [antenna[4],[antenna_gain(x0),antenna_gain(y0),0,0]]
		elif 80 <= angle <= 100:
			return [antenna[1],[0,antenna_gain(y0),0,0]]
		elif 100 < angle < 170:
			return [antenna[5],[0,antenna_gain(y0),antenna_gain(x0),0]]
		elif 170 <= angle <= 180:
			return [antenna[2],[0,0,antenna_gain(x0),0]]
	if y0 < 0:
		if 0 <= angle <= 10:
			return [antenna[2],[0,0,antenna_gain(x0),0]]
		elif 10 < angle < 80:
			return [antenna[6],[0,0,antenna_gain(x0),antenna_gain(y0)]]
		elif 80 <= angle <= 100:
			return [antenna[3],[0,0,0,antenna_gain(y0)]]
		elif 100 < angle < 170:
			return [antenna[7],[antenna_gain(x0),0,0,antenna_gain(y0)]]
		elif 170<= angle <= 180:
			return [antenna[0],[antenna_gain(x0),0,0,0]]


# write label.csv
label_path = os.path.join(raw_base_path, 'label.csv')
file = open(label_path, 'w+', newline ='')
with file:
	write = csv.writer(file)
	write.writerow(['Time','No1','No2','X1','Y1','X2','Y2','Antenna','Gain'])

for i in range(length):
	data_path = os.path.join(raw_base_path, 'x_' + str(i) + '.csv')
	data = pd.read_csv(data_path)
	No = data['No.']
	X = data['X']
	Y = data['Y']
	for a in range(nodes):
		for b in range(a+1, nodes):
			if DIS(X[a],Y[a],X[b],Y[b]) > antenna_dis:
				break
			label_path = os.path.join(raw_base_path, 'label.csv')
			file = open(label_path, 'a+', newline ='')
			with file:
				write = csv.writer(file)
				write.writerow([float(i),float(No[a]),float(No[b]),float(X[a]),float(Y[a]),
					float(X[b]),float(Y[b]),angle(X[a],Y[a],X[b],Y[b])[0],angle(X[a],Y[a],X[b],Y[b])[1]])