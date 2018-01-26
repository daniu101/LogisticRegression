#!/usr/bin/env python
# -*- coding: utf-8

from LR.load_data_module import load_dataSet
from LR.training_grad_scent_model import grad_ascent
from LR.training_stoc_grad_ascent_model import stoc_grad_ascent
from LR.training_stoc_grad_ascent_upgraded_model import stoc_grad_ascent_upgraded
from LR.matplotlib_model import matplotlib_show

DIR_DATASET = "D:/LoR.txt"
dataMat, labelMat = load_dataSet(DIR_DATASET)
    
weights = grad_ascent(dataMat, labelMat).getA()
# weights = stoc_grad_ascent(dataMat, labelMat).getA()
# weights = stoc_grad_ascent_upgraded(dataMat, labelMat).getA()
    
matplotlib_show(weights,dataMat,labelMat)