#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 13:22:56 2018

@author: sotzee
"""

from scipy.spatial import ConvexHull
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def transform_trivial(points):
    return points

def transform_circle_up(points,R):
    points_nor=(points-points.min(0)+0.001)/(points.max(0)-points.min(0)+0.001)
    theta=(points_nor[:,0]+points_nor[:,1])*np.pi/2.
    phi=np.where(points_nor[:,0]<points_nor[:,1],-np.pi*points_nor[:,0]/points_nor[:,1],np.pi*points_nor[:,1]/points_nor[:,0])
    r=points_nor[:,2]-1+R
    return np.array([r*np.sin(theta)*np.cos(phi),r*np.sin(theta)*np.sin(phi),r*np.cos(theta)]).transpose()

def transform_circle_down(points,R):
    points_nor=(points-points.min(0)+0.001)/(points.max(0)-points.min(0)+0.001)
    theta=(points_nor[:,0]+points_nor[:,1])*np.pi/2.
    phi=np.where(points_nor[:,0]<points_nor[:,1],-np.pi*points_nor[:,0]/points_nor[:,1],np.pi*points_nor[:,1]/points_nor[:,0])
    r=1-points_nor[:,2]+R
    return np.array([r*np.sin(theta)*np.cos(phi),r*np.sin(theta)*np.sin(phi),r*np.cos(theta)]).transpose()

def hull3d(points,list_01,**kwargs):
    transform_f, R = kwargs.get('transform_f',transform_trivial), kwargs.get('R',1)
    points = transform_f(points)
    transform_circle_01=[transform_circle_down,transform_circle_up]
    vertices_list=[]
    for i in list_01:
        vertices_i=list(ConvexHull(transform_circle_01[i](points,R)).vertices)
        vertices_list.append(vertices_i)
    return vertices_list

def plot_hull3d(points,hull_vertices):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    for i in range(len(hull_vertices)):
        ax.plot_trisurf(points[hull_vertices[i],0], points[hull_vertices[i],1], points[hull_vertices[i],2], linewidth=0.2, antialiased=True,cmap='jet')
    plt.show()

#test:
#from mpl_toolkits.mplot3d import Axes3D
pts = (np.random.rand(100000,3)-np.array([0.5,0.5,0.5]))*2
pts = pts[pts[:,0]**2+pts[:,1]**2+pts[:,2]**2<1]
vertices_list=hull3d(pts,[1])
plot_hull3d(pts,vertices_list)