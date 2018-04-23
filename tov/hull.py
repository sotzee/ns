#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 13:31:06 2018

@author: sotzee
"""

from scipy.spatial import ConvexHull
import numpy as np
import matplotlib.pyplot as plt

def transform_fit(points):
    import scipy.optimize as optimization
    def func(x,a,b,c,d,e):
        return a+b*x+c*x**2+d*x**3+e*x**5
    def func_(x,fit_para):
        return fit_para[0]+fit_para[1]*x+fit_para[2]*x**2+fit_para[3]*x**3+fit_para[4]*x**5
    x0=np.array([1.,1.,1.,1.,1.])
    fit_result_all=optimization.curve_fit(func, points[:,0],points[:,1], x0)
    tmp=func_(points[:,0],fit_result_all[0])
    #points_new=np.array([points[:,0],points[:,1]-tmp]).transpose()
    #plt.plot(points_new[:,0], points_new[:,1], 'o')
    #plt.plot(points[:,0], points[:,1], 'o')
    return np.array([points[:,0]-points[:,0].min(),points[:,1]-tmp-np.min(points[:,1]-tmp)]).transpose()

def transform_shift(points):
    return np.array([points[:,0]-points[:,0].min(),points[:,1]-np.min(points[:,1])]).transpose()

def transform_circle_down(points,R):
    points_nor=points/points.max(0)
    x=(R-points_nor[:,1])*np.sin(2*np.pi*points_nor[:,0])
    y=-(R-points_nor[:,1])*np.cos(2*np.pi*points_nor[:,0])
    return np.array([x,y]).transpose()

def transform_circle_right(points,R):
    points_nor=points/points.max(0)
    x=(R-1+points_nor[:,0])*np.sin(2*np.pi*points_nor[:,1])
    y=-(R-1+points_nor[:,0])*np.cos(2*np.pi*points_nor[:,1])
    return np.array([x,y]).transpose()

def transform_circle_up(points,R):
    points_nor=points/points.max(0)
    x=(R-1+points_nor[:,1])*np.sin(2*np.pi*(1-points_nor[:,0]))
    y=-(R-1+points_nor[:,1])*np.cos(2*np.pi*(1-points_nor[:,0]))
    return np.array([x,y]).transpose()

def transform_circle_left(points,R):
    points_nor=points/points.max(0)
    x=(R-points_nor[:,0])*np.sin(2*np.pi*(1-points_nor[:,1]))
    y=-(R-points_nor[:,0])*np.cos(2*np.pi*(1-points_nor[:,1]))
    return np.array([x,y]).transpose()

def hull(points,list_0123,transform_f,R):
    points = transform_f(points)
    transform_circle_0123=[transform_circle_down,transform_circle_right,transform_circle_up,transform_circle_left]
    vertices_list=[]
    for i in list_0123:
        vertices_i=list(ConvexHull(transform_circle_0123[i](points,R)).vertices)
        if(i==0):
            index_start=list(points[vertices_i,0]).index(np.min(points[vertices_i,0]))
            vertices_i=vertices_i[index_start:]+vertices_i[:index_start]
            vertices_list.append(vertices_i)
        elif(i==1):
            index_start=list(points[vertices_i,1]).index(np.min(points[vertices_i,1]))
            vertices_i=vertices_i[index_start:]+vertices_i[:index_start]
            vertices_list.append(vertices_i)
        elif(i==2):
            index_start=list(points[vertices_i,0]).index(np.max(points[vertices_i,0]))
            vertices_i=vertices_i[index_start:]+vertices_i[:index_start]
            vertices_list.append(vertices_i)
        elif(i==3):
            index_start=list(points[vertices_i,1]).index(np.max(points[vertices_i,1]))
            vertices_i=vertices_i[index_start:]+vertices_i[:index_start]
            vertices_list.append(vertices_i)
        else:
            print 'Error at hull(points,list_0123,transform_f,R) !!!!!!!!'
            
    return vertices_list

def plot_hull(points,hull_vertices):
    for i in range(len(hull_vertices)):
        plt.plot(points[hull_vertices[i],0], points[hull_vertices[i],1], 'k--', lw=2)


# =============================================================================
# #test:
# import matplotlib.pyplot as plt
# points = np.random.rand(100000, 2)
# points = points[(points[:,1]-0.5)**2+(points[:,0])**2>0.1]
# points = points[(points[:,1]-0.5)**2+(points[:,0]-1)**2>0.1]
# points = points[(points[:,1])**2+(points[:,0]-0.5)**2>0.1]
# points = points[(points[:,1]-1)**2+(points[:,0]-0.5)**2>0.1]
# =============================================================================
# =============================================================================
# import pickle
# f_check=open('./check_dat/hull_of_this_array','rb')
# points=pickle.load(f_check)
# f_check.close()
# =============================================================================

# =============================================================================
# plt.plot(points[:,0], points[:,1], 'o')
# hull4_vertices=hull(points,[0],transform_fit,max([1,len(points)/1000]))
# plot_hull(points,hull4_vertices)
# =============================================================================




# =============================================================================
# def transform_exp(points,para):
#     return 2000**(points/points.max(0))
# 
# def transform_fit(points,para):
#     import scipy.optimize as optimization
#     def func(x,a,b,c,d,e):
#         return a+b*x+c*x**2+d*x**3+e*x**5
#     def func_(x,fit_para):
#         return fit_para[0]+fit_para[1]*x+fit_para[2]*x**2+fit_para[3]*x**3+fit_para[4]*x**5
#     x0=np.array([1.,1.,1.,1.,1.])
#     fit_result_all=optimization.curve_fit(func, points[:,0],points[:,1], x0)
#     tmp=func_(points[:,0],fit_result_all[0]-para)
#     points_new=np.array([points[:,0],points[:,1]-tmp]).transpose()
#     plt.plot(points_new[:,0], points_new[:,1], 'o')
#     plt.plot(points[:,0], points[:,1], 'o')
#     return np.array([points[:,0],points[:,1]-tmp]).transpose()
# =============================================================================
    
# =============================================================================
# def hull(points,transform,para):
#     #hull1 = ConvexHull(points)
#     hull1 = ConvexHull(transform(points,-para))
#     hull2 = ConvexHull(transform(points,para))
#     vertices=merge_vertices(list(hull1.vertices),list(hull2.vertices))
#     #plt.plot(points[vertices,0], points[vertices,1], 'r--', lw=2)
#     return vertices
# 
# def aline_list(list1,list2):
#     for i in list1:
#         if i in list2:
#             list1=list1[list1.index(i):]+list1[:list1.index(i)]
#             list2=list2[list2.index(i):]+list2[:list2.index(i)]
#             return list1,list2
# 
# def merge_vertices(list1,list2):
#     i1=0
#     i2=0
#     list_result=[]
#     list1,list2=aline_list(list1,list2)
#     while(i1<len(list1) and i2<len(list2)):
#         if(list1[i1]==list2[i2]):
#             list_result.append(list1[i1])
#         else:
#             if list1[i1] in list2:
#                 index_tmp=list2.index(list1[i1])+1
#                 list_result+=list2[i2:index_tmp]
#                 i2=index_tmp-1
#             elif list2[i2] in list1:
#                 index_tmp=list1.index(list2[i2])+1
#                 list_result+=list1[i1:index_tmp]
#                 i1=index_tmp-1
#             else:
#                 print 'BUG!!! unexpected case!!!'
#                 break
#         if(i1==len(list1)):
#             list_result+=list2[i2:]
#             break
#         if(i2==len(list2)):
#             list_result+=list1[i1:]
#             break
#         i1+=1
#         i2+=1
#     list_result.append(list_result[0])
#     return list_result
# =============================================================================



# =============================================================================
# stupid trial: want to combine the four vertices array to one, which is extremely difficult
# vertices=[]
# for vertice_point in vertices1:
#     if(vertice_point not in vertices2):
#         vertices.append(vertice_point)
#     else:
#         break
# def add_vortices(points,vertice_point,vertices,vertices_now, vertices_next):
#     if vertice_point not in vertices_now :
#         points_distance=(points[vertices_now]-points[vertice_point])**2
#         points_distance=points_distance[:,0]+points_distance[:,1]
#         index_close=list(points_distance).index(np.min(points_distance))
#         print '1'
#     else:
#         print '2'
#         index_close=vertices_now.index(vertice_point)
#     vertices_now=vertices_now[index_close:]+vertices_now[:index_close]
#     for vertice_point in vertices_now:
#         if(vertice_point not in vertices_next):
#             vertices.append(vertice_point)
#         else:
#             break
#     return vertice_point
# 
# print vertices
# vertice_point=add_vortices(points,vertice_point,vertices,vertices2, vertices3)
# print vertices
# vertice_point=add_vortices(points,vertice_point,vertices,vertices3, vertices4)
# print vertices
# vertice_point=add_vortices(points,vertice_point,vertices,vertices4, vertices1)
# if vertice_point not in vertices:
#     vertice_point=add_vortices(points,vertice_point,vertices,vertices1, vertices)
# vertices.append(vertice_point)
# plt.plot(points[vertices,0], points[vertices,1], 'k--', lw=2)
# =============================================================================
