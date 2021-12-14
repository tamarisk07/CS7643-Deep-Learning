# -*- coding: utf-8 -*-
"""
Created on Sun Sep  5 17:16:07 2021

@author: Daisy
"""


import numpy as np

x = np.array([[-1.48839468, -0.31530738],
              [-0.28271176, -1.00780433],
              [0.66435418, 1.2537461],
              [-1.64829182, 0.90223236]])

exps = np.exp(x - np.max(x))
prob = exps / np.sum(exps)



y = np.array([[0.23629739, 0.76370261],
              [0.67372745, 0.32627255],
              [0.35677439, 0.64322561],
              [0.07239128, 0.92760872]])