from ctypes import *

import tensorflow as tf
# import tensorflow.compat.v1 as tf  # 设置为v1版本
import numpy as np
import os
import cv2
from tensorflow.python.saved_model import tag_constants
import sys
import time

tf.contrib.resampler


class HFNet:
    def __init__(self, model_path, outputs):
        self.session = tf.Session()  # 在1代里，运行都需要在Session里运行
        self.image_ph = tf.placeholder(tf.float32, shape=(None, None, 3))  # 设置输入，以占用的形式存在

        net_input = tf.image.rgb_to_grayscale(self.image_ph[None])  # 将3通道RGB转换成灰度图，这个函数直接转换
        tf.saved_model.loader.load(self.session, [tag_constants.SERVING], str(model_path), clear_devices=True,
                                   input_map={'image:0': net_input})
        # sess: 用于恢复模型的
        # tf.Session()对象
        # tags: 用于标识，这个要和保存的时候一样，所以不能改
        graph = tf.get_default_graph()
        self.outputs = {n: graph.get_tensor_by_name(n + ':0')[0] for n in outputs}
        self.nms_radius_op = graph.get_tensor_by_name('pred/simple_nms/radius:0')
        self.num_keypoints_op = graph.get_tensor_by_name('pred/top_k_keypoints/k:0')

    def inference(self, image, nms_radius=4, num_keypoints=1000):
        inputs = {
            self.image_ph: image[..., ::-1].astype(np.float),
            self.nms_radius_op: nms_radius,
            self.num_keypoints_op: num_keypoints,
        }
        return self.session.run(self.outputs, feed_dict=inputs)  # 这里feed_dict 是以字典的形式赋值


def createModel(modelpath, output1, output2, output3):
    print(modelpath, output1, output2, output3)
    outputs = [output1, output2, output3]
    return HFNet(modelpath, outputs)


def infer(imagePath, model):
    image = cv2.imread(imagePath)
    begintime = time.time()
    output = model.inference(image)
    endtime = time.time()
    print("Time of Processing the frame:", endtime - begintime)
    dem = output['local_descriptors'].shape[0]
    localDes = np.asarray(output['local_descriptors'], dtype=np.float32)
    globalDes = np.asarray(output['global_descriptor'], dtype=np.float32)
    localIndex = np.asarray(output['keypoints'], dtype=np.float32)
    return dem, localDes, globalDes, localIndex

if __name__ == "__main__":
    if (len(sys.argv) != 3):
        print("please input the correct args")
        print("args1: image folder;")
        print("args1: featuer folder;")
        exit(0)

    # tf.compat.v1.disable_eager_execution()  # 新加的
    # tf.disable_v2_behavior()  # 禁用v2版本 新加的
    # folders
    imagePath = sys.argv[1]
    featuerFolder = sys.argv[2]

    # define the net
    # model_path = "./model/hfnet"
    # outputs = ['global_descriptor', 'keypoints', 'local_descriptors']
    # hfnet = HFNet(model_path, outputs)
    hfnet = createModel("./model/hfnet", 'global_descriptor', 'keypoints', 'local_descriptors')

    # input the image
    # imageNames = os.listdir(imageFolder)
    # imageNames.sort()

    # create the output folder 创建文件夹
    localDesFolder = os.path.join(featuerFolder, 'des')
    globalDesFolder = os.path.join(featuerFolder, 'glb')
    keypointFolder = os.path.join(featuerFolder, 'point-txt')
    if not os.path.exists(featuerFolder):
        os.mkdir(featuerFolder)
    if not os.path.exists(localDesFolder):
        os.mkdir(localDesFolder)
    if not os.path.exists(globalDesFolder):
        os.mkdir(globalDesFolder)
    if not os.path.exists(keypointFolder):
        os.mkdir(keypointFolder)

    # inference
    # image = cv2.imread(imagePath)
    # begintime = time.time()
    # query = hfnet.inference(image)
    # endtime = time.time()
    query = infer(imagePath, hfnet)
    localDes = np.asarray(query['local_descriptors'])
    np.save(os.path.join(localDesFolder, imagePath.split("/")[-1].split(".png")[0]), localDes)
    globalDes = np.asarray(query['global_descriptor'])
    np.save(os.path.join(globalDesFolder, imagePath.split("/")[-1].split(".png")[0]), globalDes)
    localIndex = np.asarray(query['keypoints'])
    np.savetxt(os.path.join(keypointFolder, imagePath.split("/")[-1].split(".png")[0] + ".txt"), localIndex)

    print("hello")
