#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Implement by Zhe Chu, Jun 17,2021
# This python script file implement the THz Channel model introduced by
#A. Saeed, O. Gurbuz, A. O. Bicen and M. A. Akkas, 
#"Variable-Bandwidth Model and Capacity Analysis for Aerial Communications in the Terahertz Band," 
#in IEEE Journal on Selected Areas in Communications, vol. 39, no. 6, pp. 1768-1784, June 2021, doi: 10.1109/JSAC.2021.3071831.


# In[2]:


import matplotlib.pyplot as plt

import numpy as np

import math
from matplotlib.pyplot import figure
import matplotlib.pylab as pylab


# In[3]:


# Our model focusing on altitude between 10km to 16km. US standard 1976 is our atmospheric model.
# Antenna transmission power = 37 dBm, gain = 80 dBi.
# band (0.75-10THz) in the paper; however, we only take 10% of the whole band.


# In[4]:


# band (0.75-10THz)
# 10km alitutde model
altitude = 10 #km

transmission_distance = [1,2,3,4,5,6,7,8,9,10] #km

capacity =[11.53, 3.5, 1.5, 1, 0.5, 0.3, 0.2, 0.1, 0.1, 0] #Tbps 

zenith_angle = [0,30,60,90,120,150,180]

zenith_capacity = [14.1, 14, 13.5, 11.53, 10, 8.5, 8.1] # with distance = 1km


# In[5]:


# I divided it by 10 right here, sice the real bandwidth can be use is much smaller

capacity_copy = [element * 0.027 for element in capacity]

zenith_capacity_copy = [element * 0.027 for element in zenith_capacity]

capacity = capacity_copy

zenith_capacity = zenith_capacity_copy


# In[6]:


def estimate_data_at_0_distance(transmission_distance,capacity):
    
    if(transmission_distance[0]!=0):

        slop = (capacity[1] - capacity[0])/(transmission_distance[1]-transmission_distance[0])

        value_at_zero = capacity[0]-slop*transmission_distance[0]
    
        #print(slop,value_at_zero)
    
        transmission_distance.insert(0,0)
    
        capacity.insert(0,value_at_zero)
    
    return([transmission_distance,capacity])
    
    


# In[7]:


[transmission_distance,capacity] = estimate_data_at_0_distance(transmission_distance,capacity)

#print(transmission_distance)
#print(capacity)


# In[8]:


def plot_channel_capacity(altitude, transmission_distance,capacity,zenith_angle=[],zenith_capacity=[] ):

    params = {'legend.fontsize': 'x-large',
              'figure.figsize': (10, 5),
              'axes.labelsize': 'x-large',
              'axes.titlesize':'x-large',
              'xtick.labelsize':'x-large',
              'ytick.labelsize':'x-large'}
    pylab.rcParams.update(params)
    
    plt.rcParams.update({'font.size': 11})
    
    # =========channel capacity vs distance=========
    plot1 = plt.figure(1)
    
    plt.plot(transmission_distance, capacity)
    # naming the x axis
    plt.xlabel('Transmission Distance (km)')
    # naming the y axis
    plt.ylabel('Capacity(Tbps)')
    # giving a title to my graph
    title_name = 'Capacity(Tbps) vs Distance at Altitude '+str(altitude)+ ' km'

    plt.title(title_name)

    plt.ylim(0,)
    plt.xlim(1,)
    plt.grid()

    for a,b in zip(transmission_distance,capacity):
        plt.text(a, b, str(b))

    plt.xticks(np.arange(min(transmission_distance), max(transmission_distance)+1, 1.0))
    
    # =========channel capacity vs zenith angle===========
    if(len(zenith_capacity)!=0 and len(zenith_angle)!=0):
    
        plot1 = plt.figure(2)
    
        plt.plot(zenith_angle, zenith_capacity)
        # naming the x axis
        plt.xlabel('Zenith Angle (degrees)')
        # naming the y axis
        plt.ylabel('Capacity(Tbps)')
        # giving a title to my graph
        title_name = 'Capacity(Tbps) vs Zenith Angle (degrees) at Altitude '+str(altitude)+ ' km'    
    
        plt.title(title_name)
    
        plt.ylim(zenith_capacity[-1]*0.9,zenith_capacity[0]*1.1)
        plt.xlim(0,180)
        plt.grid()
    
        for a,b in zip(zenith_angle, zenith_capacity):
            plt.text(a, b, str(b))
    
        plt.xticks(np.arange(min(zenith_angle), max(zenith_angle)+1,30))
    
        plt.show()
    
#plot_channel_capacity(altitude, transmission_distance,capacity,zenith_angle,zenith_capacity)


# In[9]:


def Reverse(lst):
    return [ele for ele in reversed(lst)]


# In[10]:


class channels:
    
    def __init__(self, transmission_distance, capacity, zenith_angle, zenith_capacity, altitude):
        
        self.transmission_distance = transmission_distance
        
        self.capacity = capacity
        
        self.zenith_angle = zenith_angle
        
        self.zenith_capacity = zenith_capacity
        
        self.altitude = altitude
             


# In[11]:


jet_channel =  channels(transmission_distance, capacity, zenith_angle, zenith_capacity, altitude)


# In[12]:


# 16km alitutde model
# UAV altitude data


# In[13]:


# 33.6 dBm, 80 dBi #ignor the minor changes in antenna dBm
# band (0.75-10THz)

altitude = 16 #km

transmission_distance = [1,2,3,4,5,6,7,8,9,10] #km

capacity =[34.65, 15, 7.5, 4, 3.5, 2, 1.5, 1, 0.5, 0] #Tbps

zenith_angle = [0,30,60,90,120,150,180]

zenith_capacity = [36.3, 36.2, 35.5, 34.65,33.4, 32.5, 32.18] # with distance = 1km


# In[14]:


# I divided it by 10 right here, sice the real bandwidth can be use is much smaller

capacity_copy = [element * 0.027 for element in capacity]

zenith_capacity_copy = [element * 0.027 for element in zenith_capacity]

capacity = capacity_copy

zenith_capacity = zenith_capacity_copy


# In[15]:


[transmission_distance,capacity] = estimate_data_at_0_distance(transmission_distance,capacity)


# In[16]:


#plot_channel_capacity(altitude, transmission_distance,capacity,zenith_angle,zenith_capacity)


# In[17]:


UAV_channel =  channels(transmission_distance, capacity, zenith_angle, zenith_capacity, altitude)


# In[18]:


#print(UAV_channel.zenith_capacity)
#print(jet_channel.zenith_capacity)


# In[19]:


# Accroding to fig 6 in paper, channel capacity is almost linearly increasing vs altitude increasing
# Hence, channel capacitance at each specific altitude can be estimated
# However, what we want to have is the channel capacitance for two nodes at different altitude.
# Capacitance vs Zenith Angle graphy provide us the vertical(altitude) direction information
# We want to estimate Capacitance vs Zenith Angle at each altitude


# In[20]:


def cap_vs_zenith_at_target_altitude(target_altitude,                                     low_altitude_channel_model = jet_channel,                                     high_altitude_channel_model = UAV_channel ):
    
    
    angles = low_altitude_channel_model.zenith_angle

    start_altitude = low_altitude_channel_model.altitude

    end_altitude = high_altitude_channel_model.altitude

    start_cap_vs_znith_curve = low_altitude_channel_model.zenith_capacity

    end_cap_vs_znith_curve   = high_altitude_channel_model.zenith_capacity

    target_altitude = target_altitude


    cap_vs_zenith_at_target_altitude = []

    for i in range(len(start_cap_vs_znith_curve)):
    
        bot_val = start_cap_vs_znith_curve[i]
    
        up_val = end_cap_vs_znith_curve[i]
    
        slope = (up_val-bot_val)/(end_altitude-start_altitude)
    
        #print(slope)
    
        new_val = bot_val + slope* (target_altitude - start_altitude)
    
        new_val = max(0,new_val)
    
        cap_vs_zenith_at_target_altitude.append(new_val)
    
    return(cap_vs_zenith_at_target_altitude)    
    


# In[21]:


#test_cap_vs_zenith = cap_vs_zenith_at_target_altitude(14)

#print(test_cap_vs_zenith)


# In[22]:


#plot
'''
for test_altitude in range(jet_channel.altitude - 1, UAV_channel.altitude + 1):
    
    params = {'legend.fontsize': 'x-large',
              'figure.figsize': (30, 20),
              'axes.labelsize': 'x-large',
              'axes.titlesize':'x-large',
              'xtick.labelsize':'x-large',
              'ytick.labelsize':'x-large'}
    pylab.rcParams.update(params)
    
    plt.rcParams.update({'font.size': 22})
    
    plot1 = plt.figure(1)
    
    test_cap_vs_zenith = cap_vs_zenith_at_target_altitude(test_altitude)
    
    label = "Altitude = "+str(test_altitude)+"km"
    
    plt.plot(zenith_angle, test_cap_vs_zenith, label=label)
    
    for e in range(len(test_cap_vs_zenith)):
        
        test_cap_vs_zenith[e] = round(test_cap_vs_zenith[e],2)
    
    for a,b in zip(zenith_angle, test_cap_vs_zenith):
        plt.text(a, b, str(b))
        
    #print(test_cap_vs_zenith[0] - test_cap_vs_zenith[-1])
    
plt.grid()
    
plt.xticks(np.arange(min(zenith_angle), max(zenith_angle)+70,30))
        
plt.legend()

plt.show()
'''


# In[23]:


def find_channel_cap_vs_dsitance_at_specific_altitude(altitude,                                                      low_altitude_channel_model= jet_channel,                                                      high_altitude_channel_model = UAV_channel):

    bottom_cureve = low_altitude_channel_model.capacity
    
    upper_cureve  = high_altitude_channel_model.capacity
    
    start_altitude = low_altitude_channel_model.altitude

    end_altitude = high_altitude_channel_model.altitude
    
    alt_dif = end_altitude - start_altitude
    
    cureve_at_traget_altitude = []
    
    for i in range(len(bottom_cureve)):
        
        bottom_cap = bottom_cureve[i]
        
        upper_cap = upper_cureve[i]
        
        slop = (upper_cap - bottom_cap)/alt_dif
        
        estimate_cap = bottom_cap + slop*(altitude - start_altitude)
        
        cureve_at_traget_altitude.append(estimate_cap)
        
    return cureve_at_traget_altitude


# In[24]:


#test_curve = (find_channel_cap_vs_dsitance_at_specific_altitude(13))

#plot_channel_capacity(13 , transmission_distance,test_curve)


# In[25]:


# given X,Y data points, try to find y value when given a x

def estimate_value_by_cureve_slop(x_vals,y_vals,target_x):
    
    if(target_x<x_vals[0]):
        
        slop = (y_vals[1] - y_vals[0])/(x_vals[1] - x_vals[0])
        
        result = y_vals[0]-slop*(x_vals[0]-target_x)
        
        return result
    
    elif(target_x>x_vals[-1]):
        
        slop = (y_vals[-1] - y_vals[-2])/(x_vals[-1] - x_vals[-2])
        
        result = y_vals[-1]+slop*(target_x-x_vals[-1])
        
        return result 
    
    i=0
    
    while(x_vals[i]<target_x):
        
        i+=1
        
    i-=1
    
    x1 = x_vals[i]
    x2 = x_vals[i+1]
    y1 = y_vals[i]
    y2 = y_vals[i+1]
    
    slop = (y2-y1)/(x2-x1)
    
    result = y1+slop*(target_x-x1)
    
    return result
    
    
    


# In[26]:


#test = estimate_value_by_cureve_slop(x_vals = transmission_distance,y_vals = test_curve,target_x = 11)

#print(test)


# In[27]:


# intput: node1/2 psoition = [x,y,z] or [x,y], the z will be set =10 for default
#output: channel capacitance in unit Tbps

def find_channel_capacity(node1_position,node2_position,                          low_altitude_channel_model= jet_channel,                          high_altitude_channel_model = UAV_channel):
    
    global zenith_angle
    
    global transmission_distance
    
    # set up default altitude if altitude information are not given
    default_altitude = 10
    
    if(len(node1_position) <3):
        node1_position.append(default_altitude)
        
    if(len(node2_position) <3):
        node2_position.append(default_altitude)      
        
    # node 2 altitude must > node 1
    
    if(node2_position[2]<node1_position[2]):
        return find_channel_capacity(node2_position,node1_position,                                     low_altitude_channel_model,                                     high_altitude_channel_model)
    
    
    x_dif = node2_position[0] - node1_position[0]
    
    y_dif = node2_position[1] - node1_position[1]
    
    z_dif = node2_position[2] - node1_position[2]
    
    distance = (x_dif**2 + y_dif**2 + z_dif**2)**(1/2)
    
    #xy_plan_d = (x_dif**2 + y_dif**2)**(1/2)
    
    if(distance ==0):
        zenith_angle_between_nodes = 90
    else:
        zenith_angle_between_nodes = math.acos(z_dif/distance)/math.pi*180
    
    #print('distance between nodes: ',distance)
    #print('zenith angle between nodes: ',zenith_angle_between_nodes,'degree')
    
    
    #find lower altitude channel modle
    base_model_altitude = node1_position[2]
    
    lower_altitude_base_capacity_vs_distance_model = find_channel_cap_vs_dsitance_at_specific_altitude(base_model_altitude)
    
    base_capacitance = estimate_value_by_cureve_slop(x_vals = transmission_distance,y_vals = lower_altitude_base_capacity_vs_distance_model,target_x = distance)
    
    base_capacitance =max(base_capacitance,0)
    
    #print('without zenith angle, the capacitance is ',base_capacitance)
    
    # due to zenith angle, the acutal cureve will be shifted(up wards)
    cap_vs_zenith = cap_vs_zenith_at_target_altitude(base_model_altitude)
    
    zero_angle_val = estimate_value_by_cureve_slop(x_vals = zenith_angle,y_vals = cap_vs_zenith,target_x = 90)
    
    with_angle_val = estimate_value_by_cureve_slop(x_vals = zenith_angle,y_vals = cap_vs_zenith,target_x = zenith_angle_between_nodes)
    
    #print('when d=1km, without zenith, cap = ',zero_angle_val)
    #print('when d=1km, with zenith,  cap = ',with_angle_val)
    
    percentage_change = (with_angle_val-zero_angle_val)/zero_angle_val
    
    final_result = (1+percentage_change)*base_capacitance
    
    return final_result


# In[28]:


'''
node1_position = [0,0,16] #[0, 0, 0+10]

node2_position = [0,0,16] #[10, 1,3**(1/2)+10]


capacitance = find_channel_capacity(node1_position,node2_position)

print('channel capacitance is ',capacitance,'Tbps')
'''


# In[ ]:





# In[ ]:





# In[ ]:




