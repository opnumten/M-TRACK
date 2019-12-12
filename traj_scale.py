import numpy as np
from skimage.io import imread
from matplotlib import pyplot as plt
import os
from os import listdir
import pandas as pd
import pickle
from cell_class import single_cell,fluor_single_cell
import contour_class
import utility_tools
import image_warp
from contour_tool import df_find_contour_points,find_contour_points,generate_contours,align_contour_to,align_contours
from scipy.signal import medfilt,wiener
from traj_class import single_cell_traj,fluor_single_cell_traj
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler,QuantileTransformer




key_list=['Cell_AreaShape_Area', 'Cell_AreaShape_Center_X', 'Cell_AreaShape_Center_Y', 'Cell_AreaShape_Center_Z', \
          'Cell_AreaShape_Compactness', 'Cell_AreaShape_Eccentricity', 'Cell_AreaShape_EulerNumber', \
          'Cell_AreaShape_Extent', 'Cell_AreaShape_FormFactor', 'Cell_AreaShape_MajorAxisLength',\
          'Cell_AreaShape_MaxFeretDiameter', 'Cell_AreaShape_MaximumRadius', 'Cell_AreaShape_MeanRadius', \
          'Cell_AreaShape_MedianRadius', 'Cell_AreaShape_MinFeretDiameter', 'Cell_AreaShape_MinorAxisLength', \
          'Cell_AreaShape_Orientation', 'Cell_AreaShape_Perimeter', 'Cell_AreaShape_Solidity']
key_mask=np.ones((len(key_list),),dtype=bool)
key_mask[key_list.index('Cell_AreaShape_Center_Y')]=False
key_mask[key_list.index('Cell_AreaShape_Center_X')]=False
key_mask[key_list.index('Cell_AreaShape_Center_Z')]=False
key_mask[key_list.index('Cell_AreaShape_EulerNumber')]=False
key_mask[key_list.index('Cell_AreaShape_Orientation')]=False




#--------scaling by the first stay point--------
#https://github.com/Yurui-Li/Stay-Point-Identification
def GetDistance(data,ind1,ind2,Metric_L):#Metric_L=1: Manhattan distance , 2: Euclidean distance
    distance=np.linalg.norm(data[ind2]-data[ind1],ord=Metric_L)
    return distance



def ka(dis,dc):#dc cutoff distance
    if(dis>=dc):
        return 0
    else:
        return 1

#local density
def density(data,dc,Metric_L):
    part_density=[]   #local density
    scope=[] #density range
    leftBoundary=0;rightBoundary=len(data)-1 
    for i in range(len(data)):
        traigger=True
        left=i-1
        right=i+1
        incrementLeft=1;incrementRight=1 
        while traigger:
            #extend left
            if incrementLeft!=0:                
                if left<0:
                    left=leftBoundary
                distanceLeft=GetDistance(data,left,i,Metric_L=Metric_L)
                if (distanceLeft<dc)&(left>leftBoundary):
                    left-=1
                else:
                    incrementLeft=0
            #extend right
            if incrementRight!=0:                
                if right>rightBoundary:
                    right=rightBoundary                            
                distanceRight=GetDistance(data,i,right,Metric_L=Metric_L)
                if (distanceRight<dc)&(right<rightBoundary):
                    right+=1
                else:
                    incrementRight=0
            #stop extend
            if (incrementLeft==0)&(incrementRight==0):
                traigger=False
            if (left==leftBoundary)&(incrementRight==0):
                traigger=False
            if (incrementLeft==0)&(right==rightBoundary):
                traigger=False
        if left==leftBoundary:
            scope.append([left,right-1])
            part_density.append(right-left-1)
        elif right==rightBoundary:
            scope.append([left+1,right])
            part_density.append(right-left-1)
        else:
            scope.append([left+1,right-1])
            part_density.append(right-left-2)
    part_density=np.array(part_density)
    scope=np.array(scope)
    return part_density,scope

#reverse update
def SP_search(data,part_density,scope,tc,Metric_L):
    SP=[]

    traigger=True
    used=[]
    while traigger:
        partD=max(part_density)
        index=np.argmax(part_density)
#         print('index:',index)
        start=scope[index][0]
        end=scope[index][1]
        
        if len(used)!=0:
            for i in used:
                if (scope[i][0]>start)&(scope[i][0]<end):
                    part_density[index]=scope[i][0]-start-1
                    scope[index][1]=scope[i][0]-1
#                     print("1_1")
                if (scope[i][1]>start)&(scope[i][1]<end):
                    part_density[index]=end-scope[i][1]-1
                    scope[index][0]=scope[i][1]+1
#                     print("1_2")
                if (scope[i][0]<=start)&(scope[i][1]>=end):
                    part_density[index]=0
                    scope[index][0]=0;scope[index][1]=0
#                     print("1_3")
            start=scope[index][0];end=scope[index][1]
        timeCross=end-start
#         print('time:',timeCross)
        if timeCross>tc:
            S_arrive_t=start
            S_leave_t=end
  
            SP.append(index)
            used.append(index)
            for k in range(scope[index][0],scope[index][1]+1):
                part_density[k]=0
        part_density[index]=0
        if max(part_density)==0:
            traigger=False
    SP=np.array(SP)
    return SP


#judge stay points overlap
def similar(sp,data,dc,Metric_L):
    index=sp.tolist()
    redundant=[]
    for i in index:
        for j in index:
            if i not in redundant and j>i:
                dist=GetDistance(data,i,j,Metric_L=Metric_L)
                if dist<dc:
                    redundant.append(j)
    print(index,redundant)                
    for k in set(redundant):
        index.remove(k)
    index=np.array(index)
    return index

def sp_traj_scaling(sct,t_cutoff=6,t_range=48,Metric_L=1):
    mask=sct.traj_fluor_feature_values[0]!=0
    traj_t=sct.traj_seri[mask][:,0]
    traj_morph=sct.traj_cord[mask]
    traj_fluor=sct.traj_fluor_haralick_pca_cord[mask]
    
    # morph_scaler=StandardScaler().fit(traj_morph[:,:3].flatten()[:,None])
    # fluor_scaler=StandardScaler().fit(traj_fluor[:,:3].flatten()[:,None])

    morph_scaler=MinMaxScaler().fit(traj_morph[:,:].flatten()[:,None])
    fluor_scaler=MinMaxScaler().fit(traj_fluor[:,:].flatten()[:,None])
    
    norm_traj_morph=traj_morph.copy()
    for i in range(traj_morph.shape[1]):
        norm_traj_morph[:,i]=morph_scaler.transform(traj_morph[:,i][:,None])[:,0]
    norm_traj_fluor=traj_fluor.copy()
    for i in range(traj_fluor.shape[1]):
        norm_traj_fluor[:,i]=fluor_scaler.transform(traj_fluor[:,i][:,None])[:,0]
        

    
    X=np.column_stack((norm_traj_morph,norm_traj_fluor))
    
    # dot_color=np.arange(X.shape[0])
    # cm=plt.cm.get_cmap('jet')
    # plt.scatter(norm_traj_morph[:,0],norm_traj_fluor[:,0],c=dot_color,cmap=cm,s=1)
#     plt.show()
    
    dist_cutoff=max(np.mean(np.linalg.norm(np.diff(X[:t_range],axis=0),axis=1,ord=Metric_L)),np.mean(np.linalg.norm(np.diff(X,axis=0),axis=1,ord=Metric_L)))
#     dist_cutoff=np.mean(np.linalg.norm(np.diff(X,axis=0),axis=1))
    #print(dist_cutoff)
    part_density,scope=density(X,dist_cutoff,Metric_L=Metric_L)
    SP=SP_search(X,part_density,scope,t_cutoff,Metric_L=Metric_L)
#     print(SP)
    if SP.shape[0]>0 and np.amin(SP)<t_range:
    
        scale_t=np.amin(SP)

        #     scale_t=np.argmax(part_density[:48])
        #print(scale_t,scope[scale_t])

        #plt.scatter(norm_traj_morph[SP,0],norm_traj_fluor[SP,0])
        #plt.show()

        if scope[scale_t][1]-scope[scale_t][0]<2*t_cutoff:
            st=min(max(0,scale_t-t_cutoff),scope[scale_t][0])
            et=max(scale_t+t_cutoff,scope[scale_t][1])
        else:
            st,et=scope[scale_t][0],scope[scale_t][1]
        #print(st,et)
        
        morph_scale_area=np.mean(sct.traj_feature[mask][:,key_mask][st:et,0])
        morph_scale_ar=np.mean(sct.traj_feature[mask][:,key_mask][st:et,8])#mean redius
        morph_scale_mr=np.mean(sct.traj_feature[mask][:,key_mask][st:et,9])#median redius
        
        scale_contour=sct.traj_contour/np.sqrt(morph_scale_area)
        scale_contour_with_fluor=sct.traj_contour[mask]/np.sqrt(morph_scale_area)

        fluor_scale=np.mean(sct.traj_fluor_feature_values[3][mask][st:et,:],axis=0)

        scale_haralick=sct.traj_fluor_feature_values[3][mask]-fluor_scale



    else:
        if SP.shape[0]>0:
            plt.scatter(norm_traj_morph[SP,0],norm_traj_fluor[SP,0])
            plt.show()
        scale_contour,scale_contour_with_fluor,scale_haralick,scale_t=np.array([]),np.array([]),np.array([]),np.array([])

    return scale_contour,scale_contour_with_fluor,scale_haralick,scale_t
