#!/usr/bin/env python
# -*- coding: utf-8

import matplotlib.pyplot as plt
import numpy as np

def sigmoid(x):  #sigmoid函数
    return 1.0/(1+np.exp(-x))

plt.figure(figsize=(8,6))
    
x = np.linspace(-20,20)
y = sigmoid(x)
plt.plot(x,y,color="green",label="Sigmoid Line",linewidth=2) 
    
plt.legend()
plt.show()