#!/usr/bin/env python
# -*- coding: utf-8

from numpy import *
from LR.lr_model import sigmoid

def stoc_grad_ascent(dataMat, labelMat):  #随机梯度上升，当数据量比较大时，每次迭代都选择全量数据进行计算，计算量会非常大。所以采用每次迭代中一次只选择其中的一行数据进行更新权重。
    dataMatrix=mat(dataMat)
    classLabels=labelMat
    m,n=shape(dataMatrix)
    alpha=0.01
    maxCycles = 500
    weights=ones((n,1))
    for k in range(maxCycles):
        for i in range(m): #遍历计算每一行
            h = sigmoid(sum(dataMatrix[i] * weights))
            error = classLabels[i] - h
            weights = weights + alpha * error * dataMatrix[i].transpose()
    return weights