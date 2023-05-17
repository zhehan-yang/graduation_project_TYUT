import cv2
import joblib
import torch
import torchvision.models as models
import torch.nn as nn
from tqdm import tqdm

import netvlad
import cv2 as cv
import numpy as np
# import res2net
import torchvision.transforms as transforms
import os
import time
import random


from sklearn.decomposition import KernelPCA

import gc
from KPCA import tools

import cupy as cp

# resnet_netvlad
encoder_dim = 1024
number_cluster = 64
# 存放的路径
# model_path = "/home/zhehan-yang/Desktop/Resnet_SEblock_NetVLAD/checkpoint.pth.tar"
# model_path = "/media/zhehan-yang/yzh3/resnet/pytorch-NetVlad-master/runs/Jan06_11-58-03_resnet_netvlad/checkpoints/checkpoint.pth.tar"
# model_path = "/media/zhehan-yang/yzh3/global/pitt_r50l3_netvlad_partial.pth"
# model_path = "/media/zhehan-yang/yzh3/global/t1_msls_r50l3_netvlad.pth"
model_path = "/media/zhehan-yang/yzh3/global/msls_r101l3_netvlad_partial.pth"
# model_path = "/media/zhehan-yang/yzh3/global/pitt_r101l3_netvlad_partial.pth"
# 存放大量图片的地点，用来训练
photo_set_path = "/media/zhehan-yang/yzh3/ubuntu/datasheet/TUM"
# 做前向处理的路径
photo_path = "/media/zhehan-yang/yzh3/ubuntu/datasheet/TUM/Handheld_SLAM/rgbd_dataset_freiburg2_desk/rgb"
# 输出路径，在前向过程中存放全局描述子
# output_path = "/home/zhehan-yang/Desktop/Resnet_SEblock_NetVLAD/output"
output_path = "/media/zhehan-yang/yzh3/ubuntu/datasheet/TUM/Handheld_SLAM/rgbd_dataset_freiburg2_desk/feature_hfnet/way4_on_different_arch/glb_resnet101_msls"
# train_output_path = "/home/zhehan-yang/Desktop/Resnet_SEblock_NetVLAD/train_output"

# 存放kpca中间处理结果的路径
numpy_work_path = '/media/zhehan-yang/yzh3/global/store_kpca/kpca_resnet101vlad_msls.npy'
# kpca_path = "/media/zhehan-yang/yzh3/global/store_kpca/kpca_resnet50vlad_pitts.npy"
# kpca_path = "/media/zhehan-yang/yzh3/global/store_kpca/kpca_resnet50vlad_msls.npy"
# kpca_path = "/media/zhehan-yang/yzh3/global/store_kpca/kpca_resnet101vlad_pitts.npy"
kpca_path = "/media/zhehan-yang/yzh3/global/store_kpca/kpca_resnet101vlad_msls.npy"

# 切分的份数和每一份的长度

mode = "train"
mode = "test"

# 👇👇👇👇👇👇👇这部分做了图片路径读取的任务👇👇👇👇👇👇👇
if mode == "train":
    list_photo = []
    for root, dirs, files in os.walk(photo_set_path):
        for dir in dirs:
            if dir == "rgb":
                list_photo.extend(["/".join([root, "rgb", k]) for k in os.listdir("/".join([root, "rgb"]))])
    print("Load photo directory finish!")
    numArray = set()
    while len(numArray) < 10000:
        numArray.add(random.randint(0, len(list_photo) - 1))
    list_photo = [list_photo[k] for k in numArray]
elif mode == "test":
    list_photo = os.listdir(photo_path)

# 👇👇👇👇👇👇👇这部分做了模型定义和导入的任务👇👇👇👇👇👇👇
encoder = models.resnet101(pretrained=False)
layers = list(encoder.children())[:-3]
encoder = nn.Sequential(*layers)
model = nn.Module()
# model.add_module('encoder', encoder)
model.add_module('backbone', encoder)

net_vlad = netvlad.NetVLAD(num_clusters=number_cluster, dim=encoder_dim, vladv2=False)
# model.add_module("pool", net_vlad)
model.add_module("aggregation", net_vlad)

print("Loading files...")
checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
# model.load_state_dict(checkpoint["state_dict"])  # 代坤的用这个
model.load_state_dict(checkpoint)  # 其他用这个
print("Loading finished")

print("Moving mode to cuda")
model = model.to("cuda")
print("Moving finished")

if mode == "test":
    # kpca = np.load("/".join([numpy_work_path, "kpca", "kpca.npy"]))
    # kpca = np.load(kpca_path)
    kpca = cp.load(kpca_path)

# 👇👇👇👇👇👇👇这部分做了利用模型做前向的过程👇👇👇👇👇👇👇
if mode == "test":

    from cupy.core.dlpack import fromDlpack
    from torch.utils.dlpack import to_dlpack
    with torch.no_grad():
        trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        for photo_name in tqdm(list_photo, desc="photo inference"):
            # begintime = time.time()  # 开始计时

            photo_input = cv.imread("/".join([photo_path, photo_name]))
            photo_input = cv.resize(photo_input, dsize=(224, 224), interpolation=cv.INTER_LINEAR)
            photo_input = trans(photo_input)
            photo_input = photo_input.unsqueeze(0).to("cuda")
            # image_encoding = model.encoder(photo_input)
            # output_numpy = model.pool(image_encoding).cpu().numpy()
            image_encoding = model.backbone(photo_input)
            # output_numpy = model.aggregation(image_encoding).cpu().numpy()


            # output_numpy = model.aggregation(image_encoding).cpu().numpy()
            output_numpy = fromDlpack(to_dlpack(model.aggregation(image_encoding)))

            # 👇👇👇👇👇👇👇这部分做KPCA👇👇👇👇👇👇👇

            # output_numpy_kpca = np.matmul(output_numpy, kpca)
            # output_numpy_kpca = (output_numpy_kpca - np.min(output_numpy_kpca)) \
            #                     / \
            #                     (np.max(output_numpy_kpca) - np.min(output_numpy_kpca))
            output_numpy_kpca = cp.matmul(output_numpy, kpca)
            output_numpy_kpca = (output_numpy_kpca - cp.min(output_numpy_kpca)) \
                                / \
                                (cp.max(output_numpy_kpca) - cp.min(output_numpy_kpca))

            # 👇👇👇👇👇👇👇这部分输出👇👇👇👇👇👇👇
            # np.save("/".join([output_path, photo_name.replace("png", "npy")]), output_numpy_kpca)
            cp.save("/".join([output_path, photo_name.replace("png", "npy")]), output_numpy_kpca)
elif mode == "train":
    model.eval()
    with torch.no_grad():
        trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        first_in = True
        for photo_name in tqdm(list_photo, desc="Training Process:"):
            try:
                photo_input = cv.imread(photo_name)
                photo_input = cv.resize(photo_input, dsize=(224, 224), interpolation=cv2.INTER_LINEAR)
                photo_input = trans(photo_input)
                photo_input = photo_input.unsqueeze(0).to("cuda")
                # image_encoding = model.encoder(photo_input)
                image_encoding = model.backbone(photo_input)
                # output_numpy = model.pool(image_encoding).cpu().numpy()
                output_numpy = model.aggregation(image_encoding).cpu().numpy()
                if first_in:
                    output_numpys = [output_numpy]
                    first_in = False
                else:
                    # output_numpys = numpy.concatenate((output_numpys, output_numpy), axis=0)
                    output_numpys.append(output_numpy)
            except Exception as e:
                print(f"Some error happened in photo path: {photo_name}")
                print(f"Error named: {e}")
            # 👇👇👇👇👇👇👇这部分做KPCA👇👇👇👇👇👇👇
        try:
            print("Concatenating the numpy array.....")
            output_numpys = np.concatenate(tuple(output_numpys), axis=0)

            np.save("output_numpy.npy", output_numpys)
        except Exception as e:
            print("Error happen in numpy save,Error named: ")
            print(e)

        # data_list = []
        print("Start to train KPCA")
        # if not os.path.exists("/".join([numpy_work_path, "kpca"])):
        #     os.makedir("/".join([numpy_work_path,"kpca"]))                                                                                                                                                                                                                os.mkdir("/".join([numpy_work_path, "kpca"]))
        # print("Check kpca file package finish!")
        # for i in tqdm(range(1, copies + 1), desc="Training layer1 numpy"):
        #     data = np.load("/".join([numpy_work_path, "numpy_split", "output_numpy" + str(i) + ".npy"]))
        #     numArray = set()
        #     while len(numArray) < 20000:
        #         numArray.add(random.randint(0, data.shape[0] - 1))
        #     data = data[list(numArray), :]
        #     kpca = KernelPCA(n_components=portion / 8)
        #     # data_list.append(kpca.fit(data))
        #     kpca.fit(data)
        #     joblib.dump(kpca, numpy_work_path + f"/kpca/kpca{str(i)}.m")

        u, lams, mu = tools.pca(output_numpys, num_pcs=4096, subtract_mean=True)
        u = u[:, :4096]
        lams = lams[:4096]
        print('===> Add PCA Whiten')
        u = np.matmul(u, np.diag(np.divide(1., np.sqrt(lams + 1e-9))))
        np.save(numpy_work_path, u)
        a = 1
