
# coding: utf-8

# In[1]:


import numpy as np

from os import listdir
import pandas as pd

import pickle
from cell_class import single_cell
import sqlite3
import argparse

from skimage.io import imread
from matplotlib import pyplot as plt

import copy
import contour_class
import utility_tools
from sklearn import decomposition,cluster,manifold
from sklearn.preprocessing import StandardScaler

from contour_tool import df_find_contour_points,find_contour_points,generate_contours,align_contour_to,align_contours


# In[2]:


# args = argparse.ArgumentParser()
# args.add_argument('--input_path', type=str, default='/home/zoro/Desktop/GUI/input/')
# args.add_argument('--output_path', type=str, default='/home/zoro/Desktop/GUI/output/')
# args.add_argument('--pts_num', type=str, default=150)


# args = args.parse_args()

# input_path=args.input_path
# output_path=args.output_path
#pts_num=args.pts_num

input_path='/home/zoro/Desktop/GUI/input/'
output_path='/home/zoro/Desktop/GUI/output/'
pts_num=150


# In[3]:

def Morphology_analysis(input_path, output_path, pts_num, ax1, ax2, ax3):
    sample_path=input_path+'/sample/'
    seg_path=input_path+'/seg/'


    # In[4]:


    conn = sqlite3.connect(input_path + '/cell_track.db')
    df=pd.read_sql_query('SELECT * FROM Per_Object',conn)
    relation_df=pd.read_sql_query('SELECT * FROM Per_Relationships',conn)
    t_span=max(df['ImageNumber'])

    cells= [single_cell(img_num=df.loc[i,'ImageNumber'],obj_num=df.loc[i,'ObjectNumber']) for i in range(len(df))]
    for i in range(len(cells)):
        img_num,obj_num=cells[i].img_num,cells[i].obj_num
        #set_cell_feaures
        cells[i].set_cell_features(df.columns[3:22],df.loc[i,'Cell_AreaShape_Area':'Cell_AreaShape_Solidity'].values.tolist())
        #set_traj_label
        cells[i].set_traj_label(np.asscalar(df.loc[(df['ImageNumber']==img_num)&(df['ObjectNumber']==obj_num),'Cell_TrackObjects_Label'].values))


    with open(output_path+'/cells', 'wb') as fp:
        pickle.dump(cells, fp)



    # In[5]:


    sample_img_list=sorted(listdir(sample_path))
    contour_points_and_obj=find_contour_points(sample_path,sample_img_list,contour_value=0.5)

    cell_contours,sort_obj_arr=generate_contours(contour_points_and_obj,closed_only = True, min_area = None, max_area = None, axis_align = True)
    for i in range(len(cell_contours)):
        cell_contours[i].resample(num_points=pts_num)
        cell_contours[i].axis_align()
        points=cell_contours[i].points
        ax1.plot(points[:, 0], points[:, 1], '.')
    ax1.set_xlabel('X',fontsize=16)
    ax1.set_ylabel('Y',fontsize=16)
    # ax1.set_title('contours',fontsize=16)
    # plt.show()

    mean_contour,iters=align_contours(cell_contours,allow_reflection = True,allow_scaling = False,max_iters = 20)
    ax2.plot(mean_contour.points[:, 0], mean_contour.points[:, 1], '.')
    # plt.title('mean cell contour',fontsize=16)
    ax2.set_xlabel('X',fontsize=16)
    ax2.set_ylabel('Y',fontsize=16)
    # plt.show()

    with open(output_path+'/mean_cell_contour', 'wb') as fp:
        pickle.dump(mean_contour, fp)


    # In[6]:


    seg_img_list=sorted(listdir(seg_path))
    cell_contour_points_and_cell=df_find_contour_points(df,seg_path,seg_img_list,contour_value=0.5)
    cell_contours,sort_cell_arr=generate_contours(cell_contour_points_and_cell,closed_only = True, min_area = None, max_area = None, axis_align = False)
    for i in range(sort_cell_arr.shape[0]):
        img_num,obj_num=sort_cell_arr[i,0],sort_cell_arr[i,1]
        ind=df.loc[(df['ImageNumber']==img_num)&(df['ObjectNumber']==obj_num)].index.tolist()[0]

        cell_contours[i].resample(num_points=150)
        cell_contours[i].axis_align()
        align_contour_to(cell_contours[i], mean_contour, allow_reflection = True,allow_scaling = True)
        scale_back=utility_tools.decompose_homogenous_transform(cell_contours[i].to_world_transform)[1]
        cell_contours[i].scale(scale_back)

        cells[ind].set_cell_contour(cell_contours[i])

    with open(output_path+'/cells', 'wb') as fp:
        pickle.dump(cells, fp)


    # In[8]:


    data=np.array([single_cell.cell_contour.points for single_cell in cells if hasattr(single_cell,'cell_contour')])

    X= data
    X, data_point_shape = utility_tools.flatten_data(X)
    X=X.astype(np.float)

    pca = decomposition.PCA(n_components =0.98,svd_solver= 'full')
    Y = pca.fit_transform(X)
    # plt.title('Morphology PCA',fontsize=16)
    ax3.scatter(Y[:,0],Y[:,1],color='blue',edgecolor='k')
    ax3.set_xlabel('PC1',fontsize=16)
    ax3.set_ylabel('PC2',fontsize=16)
    # plt.show()


    with open(output_path+'/morph_pca', 'wb') as fp:
        pickle.dump(pca, fp)


    for i in range(len(cells)):
        if hasattr(cells[i],'cell_contour'):
            data=np.expand_dims(cells[i].cell_contour.points, axis=0)
            X, X_shape=utility_tools.flatten_data(data)
            Y=pca.transform(X)[0]
            cells[i].set_pca_cord(Y)
    with open(output_path+'/cells', 'wb') as fp:
        pickle.dump(cells, fp)


# In[ ]:





# In[ ]:




