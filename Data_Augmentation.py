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




from plyfile import PlyData
from point_utils import *
from bernsetin import *
from FFD import calculate_ffd
from path import Path
import os
os.chdir(os.getcwd())
import re
from emd__ import emd_module
import argparse
from model.model import *
parser = argparse.ArgumentParser()
parser.add_argument(
    '--dataset', type=str, required=True, help="dataset path")
parser.add_argument(
    '--target_dataset', type=str, required=True, help="target_dataset path")
parser.add_argument(
    '--model', type=str, required=False, help="model directory path")
parser.add_argument(
    '--split', type=str, required=True, help="the dataset split")
parser.add_argument(
    '--npoints', type=int, required=False, default=3072, help="the dataset split")
# parser.add_argument(
#     '--npoints', action="stroe_true",required=False, default=False, help="the dataset split")
parser.add_argument('--random', action='store_true', help="use random")
# parser.add_argument(
#     '--random', required=False, default=False, help="the dataset split")


if torch.cuda.is_available():
        device = torch.device('cuda')
        EMD = emd_module.emdModule()

else:
       device = torch.device('cpu')
      
def pointmixup(mixrates, xyz1, xyz2):
    # mix_rate = torch.tensor(mixrates).to(self.args.device).float()
    # mix_rate = mix_rate.unsqueeze_(1).unsqueeze_(2)
    # mix_rate_expand_xyz = mix_rate.expand(xyz1.shape).to(self.args.device)
    xyz1 = torch.tensor(xyz1).unsqueeze(0)
    xyz2 = torch.tensor(xyz2).unsqueeze(0)

    _, ass = EMD(xyz1, xyz2, 0.005, 300)
    ass = ass.cpu().numpy()
    xyz2 = xyz2[0][ass]
    xyz = xyz1 * (1 - mixrates) + xyz2 * mixrates

    return xyz.cpu().numpy()


def read_ply(f):
    plydata = PlyData.read(f)
    verts = np.vstack([plydata['vertex']['x'], plydata['vertex']['y'], plydata['vertex']['z']]).T
    face = plydata['face']['vertex_index'].T

    return verts, face

def np_to_tensor(x):
    return torch.from_numpy(x.astype(np.float32))

def deform_point(tensor,classifier,deform_net1,deform_net2):
    classifier.eval()
    deform_net1.eval()
    deform_net2.eval()


    feats,_,_ = classifier(tensor)   
    norm_feat = torch.nn.functional.normalize(feats[0], p=2.0, dim = -1)
    norm_feat = norm_feat.unsqueeze(0)

    dp1 = deform_net1(norm_feat)
    dp1 = dp1.detach().numpy()


    dp2 = deform_net2(norm_feat)
    dp2 = dp2.detach().numpy()

    return dp1[0],dp2[0]


def load_model(model_path):
    def find_max_layer(state_dict):
        numbers_after_fc = [int(re.search('fc(\d+)', item).group(1)) for item in state_dict if 'fc' in item]
        return max(numbers_after_fc)

    deform_net_1_path = model_path + '/deform/deform_net_1.pth.tar'
    deform_net_2_path = model_path + '/deform/deform_net_2.pth.tar'

    deform_net_sd = torch.load(deform_net_1_path, map_location=torch.device('cpu'))['state_dict']
    max_layer = find_max_layer(deform_net_sd)

    number_cp = int(model_path.split('_')[7])
    number_cp_per_axis = round(number_cp ** (1/3))

    deform_net_map = {
        1: Deform_Net_1layer,
        2: Deform_Net_2layer,
        3: Deform_Net_3layer
    }

    deform_net1 = deform_net_map[max_layer](in_features=128, out_features=(number_cp_per_axis)**3 * 3)
    deform_net1.load_state_dict(torch.load(deform_net_1_path, map_location=torch.device('cpu'))['state_dict'], strict=False)

    deform_net2 = deform_net_map[max_layer](in_features=128, out_features=(number_cp_per_axis)**3 * 3)
    deform_net2.load_state_dict(torch.load(deform_net_2_path, map_location=torch.device('cpu'))['state_dict'], strict=False)


    back_bone = model_path + '/best/best_model.pth.tar'  # Assuming back_bone path is also derived from model_path
    classifier = Contrastive_PointNet()
    classifier.load_state_dict(torch.load(back_bone, map_location=torch.device('cpu'))['state_dict'], strict=True)

    return classifier, deform_net1, deform_net2,number_cp_per_axis

def deform_ply(dataset_path,folder_path,model_path,npoints,split='train'):
        folders = [dir for dir in sorted(os.listdir(dataset_path)) if os.path.isdir(dataset_path / dir)]

        classifier, deform_net1, deform_net2,number_cp_per_axis=  load_model(model_path)

        if not os.path.exists(folder_path):
        # If it doesn't exist, create the folder
            os.mkdir(folder_path)
            print(f"Folder {folder_path} created successfully.")
        limit_count = 0
        for cls in folders:
                print(cls)

                if '.' in cls or limit_count>30:
                    limit_count = 0
                    continue
                # get current class folder
                cls_folder_name = cls
                cls_folder_path = folder_path+'/'+cls_folder_name

                if not os.path.exists(cls_folder_path):
                # If it doesn't exist, create the folder
                    os.mkdir(cls_folder_path)

                if not os.path.exists(cls_folder_path+'/'+'train'):
                    os.mkdir(cls_folder_path+'/'+'train')

                if not os.path.exists(cls_folder_path+'/'+'test'):
                    os.mkdir(cls_folder_path+'/'+'test')
                    # print(f"Folder {cls_folder_name} created successfully.")
                # get all the files in the folder
                file_list = os.listdir(dataset_path/cls/split)

                for file in file_list:
                    file_name = file.split('.')[0]
                    save_path = os.path.join(folder_path + '/' + cls +'/'+ split, file_name)
                    if os.path.exists(save_path+'.npy'):
                        continue



                    with open(os.path.join(dataset_path+'/'+cls+'/'+split,file),'rb') as f:
                                data  = read_ply(f)
                                verts, faces  = data
                                points =  PointSampler(npoints)((verts, faces))
                                b,p,xyz= calculate_ffd(np.array(points),n=number_cp_per_axis-1)
                                f.close()
                    print(os.path.join(dataset_path+'/'+cls+'/'+split,file))
                    # get origin 3d img

                    origin = np.matmul(b,p)
                    origin_tensor = np_to_tensor(np.array(origin)).unsqueeze(0)
                    origin_tensor = origin_tensor.transpose(2, 1)

                    # # get deformed control points
                    dp1,dp2 = deform_point(origin_tensor,classifier,deform_net1,deform_net2)
                    dp1 =Normalize()(dp1)
                    dp2 =Normalize()(dp2)

                    # dp_delta1 = np.random.rand(p.shape[0],p.shape[1])
                    # dp_delta2 = np.random.rand(p.shape[0],p.shape[1])

                    # get the new 3d img
                    new1 =  np.matmul(b,p+dp1)
                    new1 = Normalize()(new1)
                    new2 =  np.matmul(b,p+dp2)
                    new2 = Normalize()(new2)
                    mixup = pointmixup(0.5,new1,new2)
                    mixup = Normalize()(mixup[0])
                    if np.isnan(b[0][0]):
                        print(os.path.join(dataset_path + '/' + cls + '/' + split, file)+"is NaN")

                    if split=="test":
                        np.save(file=save_path+'.npy',arr=xyz)
                        continue
                    np.save(file=save_path + '.npy', arr=xyz)
                    np.save(file=save_path+'_ffd1.npy',arr=new1)
                    np.save(file=save_path+'_ffd2.npy',arr=new2)
                    np.save(file=save_path+'_mixup.npy',arr=mixup)


def deform_txt(dataset_path, folder_path, model_path, npoints,random,random_point_per_axis=4):
    folders = [dir for dir in sorted(os.listdir(dataset_path)) if os.path.isdir(dataset_path / dir)]
    if not random:
         classifier, deform_net1, deform_net2, number_cp_per_axis = load_model(model_path)
    else:
        number_cp_per_axis = random_point_per_axis

    if not os.path.exists(folder_path)\
            :
        # If it doesn't exist, create the folder
        os.mkdir(folder_path)
        print(f"Folder {folder_path} created successfully.")
    limit_count = 0
    for cls in folders:
        print(cls)

        if '.' in cls or limit_count > 30:
            limit_count = 0
            continue
        # get current class folder
        cls_folder_name = cls
        cls_folder_path = folder_path + '/' + cls_folder_name

        if not os.path.exists(cls_folder_path):
            # If it doesn't exist, create the folder
            os.mkdir(cls_folder_path)

        file_list = os.listdir(dataset_path / cls )

        for file in file_list:
            if file is  not 'train' and  file is not "test" :
                file_name = file.split('.')[0]
                save_path = os.path.join(folder_path + '/' + cls + '/' +  file_name)
                if os.path.exists(save_path + '.npy'):
                    continue

            with open(os.path.join(dataset_path + '/' + cls + '/' +  file), 'rb') as f:
                data = np.loadtxt(f, delimiter=',').astype(np.float32)[:, 0:3]
                points = data[0:npoints,:]
                points = Normalize()(points)
                b, p, xyz = calculate_ffd(np.array(points), n=number_cp_per_axis - 1)
                f.close()
            print(os.path.join(dataset_path + '/' + cls + '/' +  file))
            # get origin 3d img

            origin = np.matmul(b, p)
            origin_tensor = np_to_tensor(np.array(origin)).unsqueeze(0)
            origin_tensor = origin_tensor.transpose(2, 1)

            # # get deformed control points
            if random:
                dp_delta1 = np.random.rand(p.shape[0],p.shape[1])
                dp_delta2 = np.random.rand(p.shape[0],p.shape[1])
                # get the new 3d img
                new1 = np.matmul(b, p + dp_delta1)
                new1 = Normalize()(new1)
                new2 = np.matmul(b, p + dp_delta2)
                new2 = Normalize()(new2)

            else:
                dp1, dp2 = deform_point(origin_tensor, classifier, deform_net1, deform_net2)
                dp1 = Normalize()(dp1)
                dp2 = Normalize()(dp2)
                # get the new 3d img
                new1 = np.matmul(b, p + dp1)
                new1 = Normalize()(new1)
                new2 = np.matmul(b, p + dp2)
                new2 = Normalize()(new2)

            mixup = pointmixup(0.5, new1, new2)
            mixup = Normalize()(mixup[0])
            if np.isnan(b[0][0]):
                print(os.path.join(dataset_path + '/' + cls + '/' + file) + "is NaN")

            np.save(file=save_path + '.npy', arr=xyz)
            np.save(file=save_path + '_ffd1.npy', arr=new1)
            np.save(file=save_path + '_ffd2.npy', arr=new2)
            np.save(file=save_path + '_mixup.npy', arr=mixup)


def main():

    opt = parser.parse_args() 
    dataset_path = Path(opt.dataset)
    folder_path = opt.target_dataset
    model_path  = opt.model
    npoints = opt.npoints
    random = opt.random
    folders = [dir for dir in sorted(os.listdir(dataset_path)) if os.path.isdir(dataset_path/dir)]
    split = opt.split
    deform_txt(dataset_path, folder_path, model_path, npoints,random)
if __name__ == "__main__":
    main()