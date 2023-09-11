import torch
from torch import nn
import numpy as np
import os
import numpy as np
import itertools
import math, random
random.seed = 42
import numpy as np
import open3d as o3d

import numpy as np
import mpl_toolkits.mplot3d
import matplotlib.pyplot as plt

import pygem
print(pygem.__version__)
from pygem import FFD
from pygem import CustomDeformation


from path import Path
import scipy.spatial.distance
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from visualize import pcshow,pc_show_multi,visualize_rotate,pcwrite
from point_utils import *
from bernsetin import *
from FFD import _calculate_ffd

import torch
from torch import nn
from model import Deform_Net,PointNetCls,Contrastive_PointNet
import re
from emd__ import emd_module

EMD = emd_module.emdModule()


def pointmixup(mixrates, xyz1, xyz2):
    # mix_rate = torch.tensor(mixrates).to(self.args.device).float()
    # mix_rate = mix_rate.unsqueeze_(1).unsqueeze_(2)
    # mix_rate_expand_xyz = mix_rate.expand(xyz1.shape).to(self.args.device)
    _, ass = EMD(xyz1, xyz2, 0.005, 300)
    xyz2 = xyz2[ass]
    xyz = xyz1 * (1 - mixrates) + xyz2 * mixrates

    return xyz


def read_off(file):
    off_header = file.readline().strip()
    if 'OFF' == off_header:
        n_verts, n_faces, __ = tuple([int(s) for s in file.readline().strip().split(' ')])
    else:
        n_verts, n_faces, __ = tuple([int(s) for s in off_header[3:].split(' ')])
    verts = [[float(s) for s in file.readline().strip().split(' ')] for i_vert in range(n_verts)]
    faces = [[int(s) for s in file.readline().strip().split(' ')][1:] for i_face in range(n_faces)]
    return verts, faces


def np_to_tensor(x):
    return torch.from_numpy(x.astype(np.float32))

def deform_point(tensor,classifier,deform_net1,deform_net2):
    classifier.eval()
    feats = classifier(tensor)   
    norm_feat = torch.nn.functional.normalize(feats[0], p=2.0, dim = 1)
    dp1 = deform_net1(norm_feat)
    dp1 = dp1.detach().numpy()


    dp2 = deform_net2(norm_feat)
    dp2 = dp2.detach().numpy()

    return dp1[0],dp2[0]




if __name__ == "__main__":



    # os.chdir('/Users/wuhongyu/code/tmp_FFD/template_FFD')

    folder_name = 'Deformed_Images'
    if not os.path.exists(folder_name):
    # If it doesn't exist, create the folder
        os.mkdir(folder_name)
        print(f"Folder {folder_name} created successfully.")


    deform_net1 =  Deform_Net(in_features=128,out_features=(5+1)**3 * 3)
    deform_net1.load_state_dict(torch.load('deform_net_1.pth.tar',map_location=torch.device('cpu'))['state_dict'],strict=False)

    deform_net2 =  Deform_Net(in_features=128,out_features=(5+1)**3 * 3)
    deform_net2.load_state_dict(torch.load('deform_net_2.pth.tar',map_location=torch.device('cpu'))['state_dict'],strict=False)



    classifier = Contrastive_PointNet()
    classifier.load_state_dict(torch.load('best_model.pth.tar',map_location=torch.device('cpu'))['state_dict'],strict=False)

    print("load model successfully")


    
    path = Path('/Users/wuhongyu/code/Dataset/ModelNet40')

    folders = [dir for dir in sorted(os.listdir(path)) if os.path.isdir(path/dir)]
    # classes

    limit_count = 0

    for cls in folders:
        if '.' in cls or limit_count>30 or 'air' in cls:
            limit_count = 0
            continue
        # get current class folder
        cls_folder_name = cls
        if not os.path.exists(folder_name+'/'+cls_folder_name):
        # If it doesn't exist, create the folder
            os.mkdir(folder_name+'/'+cls_folder_name)
            print(f"Folder {cls_folder_name} created successfully.")


        # get all the files in the folder
        file_list = os.listdir(path/cls/'train')

        for file in file_list:
            with open(path/cls/'train'/file, 'r') as f:
                data  = read_off(f)
                verts, faces  = data 

            i,j,k = np.array(faces).T
            x,y,z = np.array(verts).T
            coords = np.column_stack((x,y,z))
            # get the berstin parameter, control points
            b,p,xyz= _calculate_ffd(np.array(verts),faces,n=5)
            
            # get origin 3d img
            origin = np.matmul(b,p)
            origin_tensor = np_to_tensor(np.array(origin)).unsqueeze(0)
            origin_tensor = origin_tensor.transpose(2, 1)

            # get deformed control points
            dp1,dp2 = deform_point(origin_tensor,classifier,deform_net1,deform_net2)

            # get the new 3d img
            new1 =  np.matmul(b,p+dp1)
            new1 = Normalize()(new1)


            new2 =  np.matmul(b,p+dp2)
            new2 = Normalize()(new2)


            # 这个地方放pointmixup

            # B = new1.shape[0]
            mixrates = 0.5
            new3 = pointmixup(mixrates=mixrates,xyz1=new1,xyz2=new2)
            new3 = Normalize()(new3)


            file_name = file.split('.')[0]
            img_path = folder_name+'/'+cls_folder_name+'/'+file_name            
            # save img here
            pcwrite(img_path+'_deform1',*(new1).T)
            pcwrite(img_path+'_deform2',*(new2).T)
            pcwrite(img_path+'_mixup',*(new3).T)
            pcwrite(img_path,*(origin).T)



            limit_count +=1

            



            #create class folder
            #create img folder
            #


    

