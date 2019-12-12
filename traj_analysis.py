
# coding: utf-8

# In[1]:


from cell_class import single_cell,fluor_single_cell
import contour_class
import utility_tools
from contour_tool import df_find_contour_points,find_contour_points,generate_contours,align_contour_to,align_contours

from traj_class import single_cell_traj,fluor_single_cell_traj
import numpy as np

from skimage.io import imread
from matplotlib import pyplot as plt
import pickle
import os
from os import listdir
from skimage.measure import regionprops,label

import glob
import pandas as pd
import mahotas
import mahotas.features.texture as mht
from sklearn import decomposition,cluster,manifold

from sklearn.preprocessing import StandardScaler
import sqlite3
from traj_scale import sp_traj_scaling
import copy


# In[2]:


# args = argparse.ArgumentParser()
# args.add_argument('--input_path', type=str, default='/home/zoro/Desktop/GUI/input/')
# args.add_argument('--output_path', type=str, default='/home/zoro/Desktop/GUI/output/')
# args.add_argument('--minimum_traj_length', type=str, default=12)


# args = args.parse_args()
# input_path=args.input_path
# output_path=args.output_path
# minimum_traj_length=agrs.minimum_traj_length



input_path='/home/zoro/Desktop/GUI/input/'
output_path='/home/zoro/Desktop/GUI/output/'

minimum_traj_length=36


# In[3]:





# In[4]:


def generate_fluor_traj(df,cells,traj_idx_list,fluor_name,feature_list,fluor_feature_name):
    
    haralick_labels = ["Angular Second Moment",
                   "Contrast",
                   "Correlation",
                   "Sum of Squares: Variance",
                   "Inverse Difference Moment",
                   "Sum Average",
                   "Sum Variance",
                   "Sum Entropy",
                   "Entropy",
                   "Difference Variance",
                   "Difference Entropy",
                   "Information Measure of Correlation 1",
                   "Information Measure of Correlation 2",
                    "Maximal Correlation Coefficient"]
    
    

    
    traj_xy=[]
    traj_feature=[]
    traj_contour=[]
    traj_cord=[]
    traj_seri=[]
 
    traj_fluor_feature_values=[]
    traj_haralick=[]
    traj_fluor_pca_cord=[]

    for ind in traj_idx_list:
        img_num,obj_num=df['ImageNumber'][ind],df['ObjectNumber'][ind]
        if hasattr(cells[ind],'cell_contour') and hasattr(cells[ind],'pca_cord'):
            traj_contour.append(cells[ind].cell_contour.points.flatten())
            traj_cord.append(cells[ind].pca_cord)
            traj_seri.append([img_num,obj_num])
            traj_xy.append([df.loc[ind,'Cell_AreaShape_Center_X'],df.loc[ind,'Cell_AreaShape_Center_Y']])
            traj_feature.append(df.loc[ind,'Cell_AreaShape_Area':'Cell_AreaShape_Solidity'].values.tolist())
            if hasattr(cells[ind],fluor_name+'_feature_values'):
                exec('traj_fluor_feature_values.append(np.array(cells[ind].'+fluor_name+'_feature_values[:3]))')
                exec('traj_haralick.append(np.array(cells[ind].'+fluor_name+'_feature_values[3]))')
                exec('traj_fluor_pca_cord.append(cells[ind].'+fluor_feature_name+'_pca_cord)')
            else:                    
                traj_fluor_feature_values.append(np.zeros((3,)))
                traj_haralick.append((np.zeros(13,)))
                traj_fluor_pca_cord.append(np.zeros((4,)))

    
    traj_xy=np.asarray(traj_xy)
    traj_feature=np.asarray(traj_feature)
    traj_contour=np.asarray(traj_contour)


    traj_cord=np.asarray(traj_cord)
    traj_seri=np.asarray(traj_seri)


    
    traj_fluor_feature_values=np.asarray(traj_fluor_feature_values)
    
    traj_haralick=np.asarray(traj_haralick)

    traj_fluor_pca_cord=np.asarray(traj_fluor_pca_cord)
    
   
    return traj_feature,traj_contour,traj_cord,traj_seri,[traj_fluor_feature_values[:,0],traj_fluor_feature_values[:,1],traj_fluor_feature_values[:,2],traj_haralick],traj_fluor_pca_cord


# In[5]:

def Traj_analysis(input_path, output_path, minimum_traj_length, ax1_1, ax1_2, fig2):

    sct_path=output_path+'/single_cell_traj/'
    if not os.path.exists(sct_path):
        os.makedirs(sct_path)

    sample_path=input_path+'/sample/'
    seg_path=input_path+'/seg/'
    fluor_img_path=input_path+'/fluor_img/'
    fluor_name='fluor'
    feature_name='haralick'


    feature_list=['mean_intensity','std_intensity','intensity_range','haralick']
    fluor_feature_name=fluor_name+'_'+feature_name

    seg_list=sorted(listdir(seg_path))

    with open (output_path+'/fluor_cells', 'rb') as fp:
        cells = pickle.load(fp)

    conn = sqlite3.connect(input_path + '/cell_track.db')
    df=pd.read_sql_query('SELECT * FROM Per_Object',conn)

    t_span=max(df['ImageNumber'])
    #-------------record img_num and obj_num(or idx_num in Per_Object) in all traj into one table
    #label=row number+1
    traj_labels=df['Cell_TrackObjects_Label'].values
    traj_labels=np.sort(np.unique(traj_labels[traj_labels>0]))

    for traj_label in traj_labels:
        traj_seri=[]
        traj_idx_list=df[df['Cell_TrackObjects_Label'] == int(traj_label)].index.tolist()
        for ind in traj_idx_list:
            img_num,obj_num=df['ImageNumber'][ind],df['ObjectNumber'][ind]
            if hasattr(cells[ind],'cell_contour') and hasattr(cells[ind],'pca_cord'):
                traj_seri.append([img_num,obj_num])
        if len(traj_seri)>0:

            traj_feature,traj_contour,traj_cord,traj_seri,traj_fluor_feature_values,traj_fluor_pca_cord=generate_fluor_traj(df,cells,traj_idx_list,fluor_name,feature_list,fluor_feature_name)
            traj_sct=fluor_single_cell_traj(traj_seri,traj_contour)
            traj_sct.set_traj_feature(traj_feature)
            traj_sct.set_traj_cord(traj_cord)
            traj_sct.set_traj_fluor_features(fluor_name,feature_list,traj_fluor_feature_values)
            traj_sct.set_traj_fluor_pca_cord(fluor_feature_name,traj_fluor_pca_cord)

            with open(sct_path+str(traj_label)+'_traj', 'wb') as fp:
                pickle.dump(traj_sct, fp)


    # In[12]:


    sct_list=sorted(glob.glob(sct_path+'*traj'))

    scale_features=[]
    scale_contours=[]
    all_fluor0=[]
    all_t0=[]
    all_haralick0=[]

    all_ori_morph_cord=[]
    all_ori_fluor_cord=[]


    for i in range(len(sct_list)):
        with open (sct_list[i], 'rb') as fp:
            sct = pickle.load(fp)

        mask=sct.traj_fluor_feature_values[0]!=0
        traj_t=sct.traj_seri[mask][:,0]
        if len(traj_t)>minimum_traj_length:

            traj_morph=sct.traj_cord[mask]
            traj_fluor=sct.traj_fluor_haralick_pca_cord[mask]
            traj_fluor_haralick=sct.traj_fluor_feature_values[2][mask]

            traj_scale_contour,traj_scale_contour_with_fluor,traj_scale_haralick,scale_t=sp_traj_scaling(sct,t_range=48)

            if len(traj_scale_contour)>0:
                scale_contours.append(traj_scale_contour_with_fluor)
                all_haralick0.append(traj_scale_haralick)
                all_t0.append(traj_t)

                all_ori_morph_cord.append(traj_morph)
                all_ori_fluor_cord.append(traj_fluor)


    # In[18]:


    X1=np.vstack(scale_contours)
    morph_pca = decomposition.PCA(n_components =0.98,svd_solver= 'full')
    Y1 = morph_pca.fit_transform(X1)
    print('morphology PCA explained variance ratio:')
    print(morph_pca.explained_variance_ratio_)
    X2=np.vstack(all_haralick0)
    fluor_scaler = StandardScaler()
    X2=fluor_scaler.fit_transform(X2)
    fluor_pca = decomposition.PCA(n_components =0.98,svd_solver= 'full')
    Y2 = fluor_pca.fit_transform(X2)
    print('fluorescence PCA explained variance ratio:')
    print(fluor_pca.explained_variance_ratio_)


    variances=morph_pca.explained_variance_
    variance_ratio=morph_pca.explained_variance_ratio_
    tot_variance=np.sum(variances)
    stds=np.sqrt(variances)
    # print(variances,morph_pca.explained_variance_ratio_)

    #-------plot principal modes-------------------------
    axes  = [ax1_1, ax1_2]
    for pci in range(2):#variances.shape[0]):
        ax = axes[pci]
        cell_posi=np.zeros((variances.shape[0],))
        mode_std1=copy.copy(cell_posi)
        mode_std1[pci]=1
        mode_std2=copy.copy(cell_posi)
        mode_std2[pci]=2
        mode_std_1=copy.copy(cell_posi)
        mode_std_1[pci]=-1
        mode_std_2=copy.copy(cell_posi)
        mode_std_2[pci]=-2

        shape_array0=morph_pca.inverse_transform(cell_posi*stds)
        shape_array1=morph_pca.inverse_transform(mode_std1*stds)
        shape_array2=morph_pca.inverse_transform(mode_std2*stds)
        shape_array3=morph_pca.inverse_transform(mode_std_1*stds)
        shape_array4=morph_pca.inverse_transform(mode_std_2*stds)
        shape_arrs=[shape_array0,shape_array1,shape_array2,shape_array3,shape_array4]
        for i in [0,1,3]:
            points=shape_arrs[i]
          
        #     print(points.shape)
            #fig, ax = plt.subplots(figsize=(12, 12))

            ax.plot(points[0::2], points[1::2], '-',linewidth=5)
            ax.legend(('Mean','+1','-1'),fontsize=14,ncol=3)
        ax.set_xlabel('x',fontsize=16)
        ax.set_ylabel('y',fontsize=16)
        # plt.title('scaled pricipal mode'+str(pci+1),fontsize=16)
    #     plt.xticks(np.arange(-1,1.1,0.5),fontsize=14)
    #     plt.yticks(np.arange(-0.8,0.9,0.4),fontsize=14)
        ax.set_xlim(-1.5,1.5)
        ax.set_ylim(-1,1)
        ax.set_aspect('equal', adjustable='box')
    #     plt.savefig(result_path+'mode'+str(pci+1)+'.png',dpi=300)
    #     plt.show()


    # In[9]:


    scale_features=[]
    scale_contours=[]
    scale_fluor=[]
    for i in range(len(sct_list)):
        with open (sct_list[i], 'rb') as fp:
            sct = pickle.load(fp)
        mask=sct.traj_fluor_feature_values[0]!=0
        inds=np.where(sct.traj_fluor_feature_values[0]!=0)
        traj_t=sct.traj_seri[mask][:,0]
        traj_morph=sct.traj_cord[mask]
        traj_fluor=sct.traj_fluor_haralick_pca_cord[mask]
        if len(traj_t)>minimum_traj_length:


            scale_contour,scale_contour_with_fluor,scale_haralick,scale_t=sp_traj_scaling(sct)


            if len(scale_contour)>0:
                scale_morph_cord=morph_pca.transform(scale_contour)
                scale_fluor_cord0=fluor_pca.transform(fluor_scaler.transform(scale_haralick))
                scale_fluor_cord=np.zeros((scale_morph_cord.shape[0],scale_fluor_cord0.shape[1]))
                scale_fluor_cord[inds,:]=scale_fluor_cord0


                sct.set_traj_scale_cord(scale_morph_cord)
                sct.set_traj_fluor_pca_cord('fluor_scale_haralick',scale_fluor_cord)

                with open (sct_list[i], 'wb') as fp:
                    pickle.dump(sct,fp)
            else:
                sct.set_traj_scale_cord(None)
                sct.set_traj_fluor_pca_cord(fluor_name,None)


                with open (sct_list[i], 'wb') as fp:
                    pickle.dump(sct,fp)
        else:
            sct.set_traj_scale_cord(None)
            sct.set_traj_fluor_pca_cord(fluor_name,None)


            with open (sct_list[i], 'wb') as fp:
                pickle.dump(sct,fp)


    # In[17]:


    #----------choose a trajectory from folder
    single_cell_traj=sct_list[1]
    plottraj_message = 'trajectory plotted'
    with open (single_cell_traj, 'rb') as fp:
        sct = pickle.load(fp)
    if hasattr(sct,'traj_scale_cord') and hasattr(sct,'traj_fluor_scale_haralick_pca_cord'):
        if sct.traj_scale_cord is not None:

            mask=sct.traj_fluor_feature_values[0]!=0

            traj_t=sct.traj_seri[mask][:,0]
            morph_traj=sct.traj_scale_cord[mask]
            fluor_traj=sct.traj_fluor_scale_haralick_pca_cord[mask]

            dot_color=np.arange(morph_traj.shape[0])
            cm=plt.cm.get_cmap('jet')
            # plt.title('single cell trajectory plot',fontsize=16)
            colorbar_instance = fig2.gca().scatter(morph_traj[:,0],fluor_traj[:,0],c=dot_color,cmap=cm)
            fig2.gca().set_xlabel('Scaled Morphology PC1',fontsize=16)
            fig2.gca().set_ylabel('Scaled Fluorescence PC1',fontsize=16)
            fig2.colorbar(colorbar_instance)
            # plt.show()
        else:
            plottraj_message =  'cannot plot trajectory, sct.traj_scale_cord is None'
    else:
         plottraj_message = 'cannot plot trajectory  not hasattr(sct, "traj_scale_cord") or not hasattr(sct, "traj_fluor_scale_haralick_pca_cord")'

    return morph_pca.explained_variance_ratio_, fluor_pca.explained_variance_ratio_, plottraj_message

def Draw_chosen_traj(single_cell_traj, fig):
    print(single_cell_traj)
    with open(single_cell_traj, 'rb') as fp:
        sct = pickle.load(fp)
    if hasattr(sct, 'traj_scale_cord') and hasattr(sct, 'traj_fluor_scale_haralick_pca_cord'):
        if sct.traj_scale_cord is not None:

            mask = sct.traj_fluor_feature_values[0] != 0

            traj_t = sct.traj_seri[mask][:, 0]
            morph_traj = sct.traj_scale_cord[mask]
            fluor_traj = sct.traj_fluor_scale_haralick_pca_cord[mask]

            dot_color = np.arange(morph_traj.shape[0])
            cm = plt.cm.get_cmap('jet')
            # plt.title('single cell trajectory plot', fontsize=16)
            colorbar_instance = fig.gca().scatter(morph_traj[:, 0], fluor_traj[:, 0], c=dot_color, cmap=cm)
            fig.gca().set_xlabel('Scaled Morphology PC1', fontsize=16)
            fig.gca().set_ylabel('Scaled Fluorescence PC1', fontsize=16)
            fig.colorbar(colorbar_instance)
            # plt.show()
        else:
            return 'cannot plot trajectory, sct.traj_scale_cord is None'
    else:
        return 'cannot plot trajectory  not hasattr(sct, "traj_scale_cord") or not hasattr(sct, "traj_fluor_scale_haralick_pca_cord")'
    return 'trajectory plotted'


# In[ ]:




