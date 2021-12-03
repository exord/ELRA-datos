#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 17:46:11 2021

@author: rodrigo
"""

import matplotlib.pyplot as plt
import numpy as np
# from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x = np.linspace(-2, 2, 100)

X, Y = np.meshgrid(x, x)

def func(X, Y):
    return 2*X**2 + Y**2 + 5

def grad(X, Y):
    return np.array([4*X, 2*Y])

ax.plot_surface(X, Y, func(X, Y), cmap='jet', alpha=0.4)
ax.scatter(0, 0, 5, c='r', marker='X')

yi = 0
xi = 1.5
eps = 0

ax.scatter(xi, yi, func(xi, yi) + eps, s=10, c='k', zorder=20, alpha=1.0)

eta = 0.05
# eta = 0.52
for i in range(10):
    
    # Compute gradient
    gx, gy = -grad(xi, yi)
    xf, yf = np.array([xi, yi]) + eta * np.array([gx, gy])
    
    deltaf = func(xf, yf) - func(xi, yi)
    ax.quiver(xi, yi, func(xi, yi), eta*gx, eta*gy, deltaf,
              arrow_length_ratio=0, color='k', lw=1)
    
    ax.scatter(xf, yf, func(xf, yf), s=10, c='k')
    
    xi, yi = xf, yf
    