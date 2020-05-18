import numpy as np

def wrap_to_2pi(data):
    positive_input = (data > 0)
    zero_check = np.logical_and(positive_input,(data == 0))
    data = np.mod(data, 2*np.pi)
    data[zero_check] = 2*np.pi
    return data

def rad_to_deg(data):
    return data*180/np.pi

def groupdelay(signal): # TODO
    return

def phase(signal, deg=False, unwrap=False): # TODO
    return
