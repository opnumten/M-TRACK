
# coding: utf-8

# In[6]:


import numpy as np

from skimage.io import imread
from matplotlib import pyplot as plt
import pickle
import os
from os import listdir
from skimage.measure import regionprops,label
from PIL import Image, ImageDraw, ImageFont

import glob
import pandas as pd
from cell_class import single_cell,fluor_single_cell
import mahotas
import mahotas.features.texture as mht
from sklearn import decomposition,cluster,manifold

from sklearn.preprocessing import StandardScaler
import sqlite3


# In[7]:


# args = argparse.ArgumentParser()
# args.add_argument('--input_path', type=str, default='/home/zoro/Desktop/GUI/input/')
# args.add_argument('--output_path', type=str, default='/home/zoro/Desktop/GUI/output/')
# args.add_argument('--fluor_interval', type=str, default=1)

# args = args.parse_args()
# input_path=args.input_path
# fluor_interval=agrs.fluor_interval

input_path='/home/zoro/Desktop/GUI/input/'
output_path='/home/zoro/Desktop/GUI/output/'
fluor_interval=2


# In[8]:





# In[9]:


def compute_fluor_info(seg,fluor_img):
    rps=regionprops(seg)
    
    mean_intensity=np.zeros((len(rps)))
    std_intensity=np.zeros((len(rps)))
    intensity_range=np.zeros((len(rps)))
    fluor_haralick=[]
    for i in range(len(rps)):
        cell_num=int(i+1)
        r=rps[i]
        cell_mask=(seg==cell_num)
        region_cell_mask=cell_mask[r.bbox[0]:r.bbox[2],r.bbox[1]:r.bbox[3]]
        
        crop_img=fluor_img[r.bbox[0]:r.bbox[2],r.bbox[1]:r.bbox[3]]
        cell_img=(fluor_img*cell_mask)[r.bbox[0]:r.bbox[2],r.bbox[1]:r.bbox[3]]
        
        mean_intensity[i]=np.sum(cell_img)*1.0/r.area
        std_intensity[i]=np.std(cell_img[region_cell_mask])
        
        
        min_value,max_value=np.amin(cell_img[region_cell_mask]),np.amax(cell_img[region_cell_mask])
        min_value=min_value-1
        
        intensity_range[i]=max_value-min_value

        
        #the haralick features have four directions, to meet rotation invariance,use average for each feature
        fl_hara=np.mean(mht.haralick(cell_img, ignore_zeros=True),axis=0)

        
        fluor_haralick.append(fl_hara)
    fluor_haralick=np.array(fluor_haralick)
    
    return mean_intensity,std_intensity,intensity_range,fluor_haralick


def Haralick_analysis(input_path, output_path, fluor_interval, ax1):
    sample_path=input_path+'/sample/'
    seg_path=input_path+'/seg/'
    fluor_img_path=input_path+'/fluor_img/'
    fluor_name='fluor'
    feature_name='haralick'


    # In[10]:


    feature_list=['mean_intensity','std_intensity','intensity_range','haralick']
    fluor_feature_name=fluor_name+'_'+feature_name


    seg_list=sorted(listdir(seg_path))
    fluor_img_list=sorted(listdir(fluor_img_path))





    conn = sqlite3.connect(input_path + '/cell_track.db')
    df=pd.read_sql_query('SELECT * FROM Per_Object',conn)
    t_span=max(df['ImageNumber'])


    with open (output_path+'/cells', 'rb') as fp:
        cells = pickle.load(fp)
    for k in range(len(cells)):
        fluor_single_cell.convert_to_class(cells[k])



    for ti in np.arange(0,t_span,fluor_interval):#
        img_num=ti+1
        seg=imread(seg_path+seg_list[ti])

        fluor_img=imread(fluor_img_path+fluor_img_list[ti])


        mean_intensity,std_intensity,intensity_range,fluor_haralick=compute_fluor_info(seg,fluor_img)

        for obj_num in np.arange(1,np.amax(seg)+1):
            ind=df.loc[(df['ImageNumber']==img_num)&(df['ObjectNumber']==obj_num)].index.tolist()[0]
            fluor_features=[mean_intensity[obj_num-1],std_intensity[obj_num-1],intensity_range[obj_num-1],fluor_haralick[obj_num-1,:]]
            cells[ind].set_fluor_features(fluor_name,feature_list,fluor_features)


    # In[13]:


    data=np.array([single_cell.fluor_feature_values[3] for single_cell in cells if hasattr(single_cell,fluor_name+'_feature_values')])

    scaler = StandardScaler()

    X=scaler.fit_transform(data)

    pca = decomposition.PCA(n_components =0.98,svd_solver= 'full')
    Y = pca.fit_transform(X)
    # print(pca.explained_variance_ratio_)
    # plt.title('Fluorescence PCA',fontsize=16)
    ax1.scatter(Y[:,0],Y[:,1],color='blue',edgecolor='k')
    ax1.set_xlabel('PC1',fontsize=16)
    ax1.set_ylabel('PC2',fontsize=16)
    # plt.show()

    with open(output_path+'/fluor_pca', 'wb') as fp:
        pickle.dump(pca, fp)

    for i in range(len(cells)):
        if hasattr(cells[i],fluor_name+'_feature_values'):
            X=np.expand_dims(cells[i].fluor_feature_values[3],axis=0)
            X=scaler.transform(X)
            Y=pca.transform(X)[0]
            cells[i].set_fluor_pca_cord(fluor_feature_name,Y)
    with open(output_path+'/fluor_cells', 'wb') as fp:
        pickle.dump(cells, fp)


    # In[ ]:




