import numpy as np
import contour_class


class single_cell(object):
    def __init__(self, img_num,obj_num):
        self.img_num=img_num
        self.obj_num=obj_num
    #features
    def set_cell_features(self,feaure_name,feature_value):
        self.cell_features=dict(zip(feaure_name,feature_value))
    def set_traj_label(self,traj_label):
        self.traj_label=traj_label


    #cell contour
    def set_cell_contour(self,cell_contour):
        self.cell_contour=cell_contour
    
    def set_pca_cord(self,pca_cord):
        self.pca_cord=pca_cord

#----for converting parent instance to child instance----
class ConverterMixin(object):
    @classmethod
    def convert_to_class(cls,obj):
        obj.__class__=cls
        
class fluor_single_cell(ConverterMixin,single_cell):
    def __init__(self,img_num,obj_num):
        super(fluor_single_cell,self).__init__(img_num,obj_num)
        
    def set_fluor_features(self,fluor_name,feature_list,feature_values):
        exec('self.'+fluor_name+'_feature_list=feature_list')
        exec('self.'+fluor_name+'_feature_values=feature_values')

    def set_fluor_pca_cord(self,fluor_feature_name,fluor_pca_cord):
        exec('self.'+fluor_feature_name+'_pca_cord=fluor_pca_cord')
