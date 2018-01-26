#!/usr/bin/env python
# -*- coding: utf-8 

from numpy import *

def sigmoid(x):  #sigmoid函数
    return 1.0/(1+exp(-x))