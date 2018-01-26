#!/usr/bin/env python
# -*- coding: utf-8 

def load_dataSet(DIR_DATASET):
    dataMat = []
    labelMat = []
    fr = open(DIR_DATASET)
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])   #前面的1，表示方程的常量。比如两个特征X1,X2，共需要三个参数，W1+W2*X1+W3*X2
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat