/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*
* DXSLAM，A Robust and Efficient Visual SLAM System with Deep Features，is based on the famous ORB-SLAM2. 
* Copyright (C) 2020, iVip Lab @ EE, THU (https://ivip-tsinghua.github.io/iViP-Homepage/). All rights reserved.
* Licensed under the GPLv3 License;
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
* https://github.com/ivipsourcecode/dxslam/blob/master/License-gpl.txt
*/
#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>
#include <unistd.h>
#include<opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include "cnpy.h"
#include<System.h>

#include <Python.h>
#include<Label.h>

#include <numpy/arrayobject.h>
#include <conio.h>

#ifndef CV_LOAD_IMAGE_UNCHANGED
#define CV_LOAD_IMAGE_UNCHANGED -1
#endif

using namespace std;
using namespace cv;

void LoadImages(const string &strAssociationFilename, vector<string> &vstrImageFilenamesRGB,
                vector<string> &vstrImageFilenamesD, vector<double> &vTimestamps);

//下面这三个是针对ORB-SLAM2新加的
void getdescriptor(string filename, cv::Mat &descriptor, int nkeypoints);

void getGlobaldescriptor(string filename, cv::Mat &descriptor);

void getKeyPoint(string filename, vector<cv::KeyPoint> &keyPoints);

struct feature3 {
    Mat localDes;
    Mat globalDes;
    vector<cv::KeyPoint> keyPoints;
};
PyObject *pModule;
PyObject *pFuncInfer;

//创建HF-Net模型
PyObject *CreateModelAndInitial(wchar_t *pythonPath, const char *featurePyPath) {
    Py_SetPythonHome(pythonPath);     //终于可以了
    Py_Initialize();       //初始化python解释器，告诉编译器要用的python编译器
    if (!Py_IsInitialized()) {
        fprintf(stderr, "Py_Initialize Failed ");
    }
    *pModule;
    PyObject *pFuncCreate;
    // 2、初始化python系统文件路径，保证可以访问到 .py文件
    PyRun_SimpleString("import sys");
    char tempOrder[100] = "sys.path.append('";
    strcat(tempOrder, featurePyPath);
    strcat(tempOrder, "')");
    PyRun_SimpleString(tempOrder);
    PyRun_SimpleString("print(sys.path)");

    pModule = PyImport_ImportModule("getFeature");
    if (pModule == NULL) { cout << "没找到" << endl; }

    //建立模型
    pFuncCreate = PyObject_GetAttrString(pModule, "createModel");
    strcpy(tempOrder, featurePyPath);
    strcat(tempOrder, "/model/hfnet"); //Model path
    PyObject *pParamsCreate = Py_BuildValue("(ssss)",
                                            tempOrder,
                                            "global_descriptor", "keypoints", "local_descriptors");
    PyObject *pRetCreate_hfnet = PyObject_CallObject(pFuncCreate, pParamsCreate);
    if (PyErr_Occurred()) PyErr_Print();
    //顺便初始化前向的函数
    pFuncInfer = PyObject_GetAttrString(pModule, "infer");
    return pRetCreate_hfnet;
}

//HF-Net前向
feature3 Inference(PyObject *model, const char *photoPath) {
    PyObject *pParamsInfer = Py_BuildValue("sO", photoPath, model);
    PyObject *pRetInfer = PyEval_CallObject(pFuncInfer, pParamsInfer);
    int dem;
    if (PyErr_Occurred()) PyErr_Print();
    PyArrayObject *py_array1, *py_array2, *py_array3;
    PyArg_ParseTuple(pRetInfer, "i|O|O|O", &dem, &py_array1, &py_array2, &py_array3);
    Mat local = Mat(dem, 256, CV_32F, PyArray_DATA(py_array1));
    Mat global = Mat(4096, 1, CV_32F, PyArray_DATA(py_array2));
    Mat keypointm = Mat(dem, 2, CV_32F, PyArray_DATA(py_array3));
    vector<cv::KeyPoint> keyPoints;
    for (int i = 0; i < dem; i++) {
        KeyPoint keyPoint(keypointm.at<float>(i, 0), keypointm.at<float>(i, 1), 1);
        //KeyPoint的属性值有angle（角度，表示方向），class_id（对图片惊醒分类的时候，使用的该值做特征点区分）
        //octave表示从金字塔哪一层提取到的数据，pt（二维向量，关键点坐标），reponse（响应程度，好不好？）
        keyPoint.octave = 0;
        keyPoints.push_back(keyPoint);
    }
    feature3 output;
    output.localDes = local;
    output.globalDes = global;
    output.keyPoints = keyPoints;
    return output;
};

int main(int argc, char **argv) {
    PyObject *model = CreateModelAndInitial(L"/home/zhehan-yang/miniconda3/envs/DXSLAM",
                                            "/home/zhehan-yang/Desktop/DX-SLAM/dxslam-master-forreal/hf-net");
    //for counter
    Label ReLocolizationCounter;
    int lastState = -1;
    if (argc != 6) {
        cerr << endl
             << "Usage: ./rgbd_tum path_to_vocabulary path_to_settings path_to_sequence path_to_association path_to_feature"
             << endl;
        return 1;
    }
    //如果输入的参数数目不是6，那么，就报错，格式是./rgbd_tum path_to_vocabulary path_to_settings path_to_sequence path_to_association path_to_feature

    // Retrieve paths to images
    vector<string> vstrImageFilenamesRGB;
    vector<string> vstrImageFilenamesD;
    vector<double> vTimestamps;
    string strSequenceFolder = string(argv[3]);            //path to sequence，这个是存放图片的文件夹的路径，DX-SLAM新加的
    string strAssociationFilename = string(
            argv[4]);    //path to association，存放图片之间 联系的文件，文件里头存放的是图片时间以及图片的文件名，包括RGB图和深度图
    string featureFolder = string(argv[5]);                //path to feature，DX-SLAM新加的
    if (strSequenceFolder != "0") {
        LoadImages(strAssociationFilename, vstrImageFilenamesRGB, vstrImageFilenamesD,
                   vTimestamps);        //他的定义在这个文件下面
        //    这个函数功能是根据第一个参数路径，将数据读入后面三个参数（RGB数值，深度数值和时间数值中，其中，时间数值取前者）

        //     Check consistency in the number of images and depthmaps  这块代码ORB-SLAM也有
        int nImages = vstrImageFilenamesRGB.size();                                //nimage存的是总共有多少帧/多少张图片
        if (vstrImageFilenamesRGB.empty())                                        //一个都没读取到
        {
            cerr << endl << "No images found in provided path." << endl;
            return 1;
        } else if (vstrImageFilenamesD.size() != vstrImageFilenamesRGB.size())        //深度图和RGB图不一样多报错
        {
            cerr << endl << "Different number of images for rgb and depth." << endl;
            return 1;
        }
        // Create SLAM system. It initializes all system threads and gets ready to process frames.
        DXSLAM::System SLAM(argv[1], argv[2], DXSLAM::System::RGBD, true);    //这个对应的是System.cc里面namespace System的定义

        // Vector for tracking time statistics
        vector<float> vTimesTrack;
        vTimesTrack.resize(nImages);        //规划成和图片长度一样

        cout << endl << "-------" << endl;
        cout << "Start processing sequence ..." << endl;                    //开始了
        cout << "Images in the sequence: " << nImages << endl << endl;        //总共有多少张图片

        //    //cat the image path，
        LoadImages(strAssociationFilename, vstrImageFilenamesRGB, vstrImageFilenamesD, vTimestamps);        //再来一次？

        //main loop   正式图像处理
        cv::Mat imRGB, imD;

        for (int ni = 0; ni < vstrImageFilenamesRGB.size(); ni++) {
            // Read image and depthmap from file 读入图片，参数分别是路径，图片标志，1表示"/media/zhehan-yang/yzh3/ubuntu/datasheet/TUM/Handheld_SLAM/rgbd_dataset_freiburg2_desk/rgb/1311868164.363181.png"彩色，0表示灰度，-1表示有alpha的彩色
            imRGB = cv::imread(strSequenceFolder + "/" + vstrImageFilenamesRGB[ni],
                               CV_LOAD_IMAGE_UNCHANGED);        //读入色彩
            imD = cv::imread(strSequenceFolder + "/" + vstrImageFilenamesD[ni],
                             CV_LOAD_IMAGE_UNCHANGED);            //读入深度
            double tframe = vTimestamps[ni];        //时间参数

            if (imRGB.empty()) {                    //没有直接报错
                cerr << endl << "Failed to load image at: "
                     << strSequenceFolder << "/" << vstrImageFilenamesRGB[ni] << endl;
                return 1;
            }


#ifdef COMPILEDWITHC11            //如果说是11标准，那么计时程序就这么写，否则就按照下面那种写法
            std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
#else
            std::chrono::monotonic_clock::time_point t1 = std::chrono::monotonic_clock::now();
#endif

            // Pass the image and hf-net output to the SLAM system
            cv::Mat local_desc;                    //局部描述子
            cv::Mat global_desc;                //全局描述子,注意这是一个矩阵
            vector<cv::KeyPoint> keypoints;        //关键点

            // Get keyPoint,local descriptor and global descriptor
            //这函数是在本文件中
            //        getKeyPoint(featureFolder + "/point-txt/" + to_string(vTimestamps[ni]) + ".txt",
            //                    keypoints);        //为什么要路径，这个是在数据库中读取嘛？
            //        local_desc.create(keypoints.size(), 256, CV_32F);    //创建一个keypoint.size()(特征点个数)维，256行的矩阵，类型是CV-32F（指0~1任意值）
            //        //256的意思是网络输出整个特征点多维度的特征值嘛？
            //        getdescriptor(featureFolder + "/des/" + to_string(vTimestamps[ni]) + ".npy", local_desc, keypoints.size());
            //        //这里.npy是numpy专用的存储二进制文件的格式
            //        global_desc.create(4096, 1, CV_32F);                //同上
            //        getGlobaldescriptor(featureFolder + "/glb/" + to_string(vTimestamps[ni]) + ".npy", global_desc);    //和上面差不多
            string photoPath = string(argv[3]) + "/" + vstrImageFilenamesRGB[ni];
            feature3 output = Inference(model, photoPath.data());

            //源码在System.cc中
            //        SLAM.TrackRGBD(imRGB, imD, tframe, keypoints, local_desc, global_desc);        //跟踪线程，有回环检测，后端优化那些
            SLAM.TrackRGBD(imRGB, imD, tframe, output.keyPoints, output.localDes,
                           output.globalDes);        //跟踪线程，有回环检测，后端优化那些
#ifdef COMPILEDWITHC11
            std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
#else
            std::chrono::monotonic_clock::time_point t2 = std::chrono::monotonic_clock::now();
#endif

            double ttrack = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
            vTimesTrack[ni] = ttrack;        //记录的是每一次处理数据的用时

            // Wait to load the next frame 等待20ms时间,等下一张图片
            usleep(1000 * 20);

            //count all
            if (SLAM.GetTrackingState() != lastState) {
                ReLocolizationCounter.AddChange(SLAM.GetTrackingState(), tframe, true);
                lastState = SLAM.GetTrackingState();
            }
            if (ni == vstrImageFilenamesRGB.size() - 1) {
                //to know the last time
                ReLocolizationCounter.AddChange(1, tframe, true);
            }
        }
        ReLocolizationCounter.printsALLLabels(true, true);
        SLAM.Shutdown();
// Tracking time statistics
        sort(vTimesTrack.begin(), vTimesTrack.end());
        float totaltime = 0;
        for (int ni = 0; ni < nImages; ni++) {
            totaltime += vTimesTrack[ni];        //累加所有的运行时间
        }
        cout << "-------" << endl << endl;
        cout << "median tracking time: " << vTimesTrack[nImages / 2] << endl;
        cout << "mean tracking time: " << totaltime / nImages << endl;
//这个用来输出效率的平均值和总时间
// Save camera trajectory
// 保存所有的路径文件
        SLAM.SaveTrajectoryTUM("CameraTrajectory.txt");
        SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");

        return 0;

    } else {


        DXSLAM::System SLAM(argv[1], argv[2], DXSLAM::System::RGBD, true);    //这个对应的是System.cc里面namespace System的定义

// Vector for tracking time statistics
        vector<float> vTimesTrack;

//main loop   正式图像处理
        cv::Mat imRGB, imD;

        std::chrono::steady_clock::time_point tStart = std::chrono::steady_clock::now();
        while (1) {
// Read image and depthmap from file 读入图片，参数分别是路径，图片标志，1表示"/media/zhehan-yang/yzh3/ubuntu/datasheet/TUM/Handheld_SLAM/rgbd_dataset_freiburg2_desk/rgb/1311868164.363181.png"彩色，0表示灰度，-1表示有alpha的彩色
            cv::VideoCapture cap(0); // 从摄像头读取数据流
            if (!cap.

                    isOpened()

                    ) {
                std::cout << "Failed to open camera!" <<
                          std::endl;
                return -1;
            }
            cv::Mat frame;


            cap >>
                frame; // 读取数据流

            cv::imshow("frame", frame);


//            imRGB = cv::imread(strSequenceFolder + "/" + vstrImageFilenamesRGB[ni],CV_LOAD_IMAGE_UNCHANGED);        //读入色彩
//            imD = cv::imread(strSequenceFolder + "/" + vstrImageFilenamesD[ni],CV_LOAD_IMAGE_UNCHANGED);            //读入深度
            double tframe = std::chrono::duration_cast<std::chrono::duration<double> >(
                    std::chrono::steady_clock::now() - tStart).count();        //时间参数

//            if (imRGB.empty()) {                    //没有直接报错
//                cerr << endl << "Failed to load image at: "
//                     << strSequenceFolder << "/" << vstrImageFilenamesRGB[ni] << endl;
//                return 1;
//            }
#ifdef COMPILEDWITHC11            //如果说是11标准，那么计时程序就这么写，否则就按照下面那种写法
            std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
#else
            std::chrono::monotonic_clock::time_point t1 = std::chrono::monotonic_clock::now();
#endif

// Pass the image and hf-net output to the SLAM system
            cv::Mat local_desc;                    //局部描述子
            cv::Mat global_desc;                //全局描述子,注意这是一个矩阵
            vector<cv::KeyPoint> keypoints;        //关键点

// Get keyPoint,local descriptor and global descriptor
//这函数是在本文件中
//        getKeyPoint(featureFolder + "/point-txt/" + to_string(vTimestamps[ni]) + ".txt",
//                    keypoints);        //为什么要路径，这个是在数据库中读取嘛？
//        local_desc.create(keypoints.size(), 256, CV_32F);    //创建一个keypoint.size()(特征点个数)维，256行的矩阵，类型是CV-32F（指0~1任意值）
//        //256的意思是网络输出整个特征点多维度的特征值嘛？
//        getdescriptor(featureFolder + "/des/" + to_string(vTimestamps[ni]) + ".npy", local_desc, keypoints.size());
//        //这里.npy是numpy专用的存储二进制文件的格式
//        global_desc.create(4096, 1, CV_32F);                //同上
//        getGlobaldescriptor(featureFolder + "/glb/" + to_string(vTimestamps[ni]) + ".npy", global_desc);    //和上面差不多
//            string photoPath = string(argv[3]) + "/" + vstrImageFilenamesRGB[ni];
            string photoPath = "0";
            feature3 output = Inference(model, photoPath.data());

//源码在System.cc中
//        SLAM.TrackRGBD(imRGB, imD, tframe, keypoints, local_desc, global_desc);        //跟踪线程，有回环检测，后端优化那些
            SLAM.
                    TrackRGBD(imRGB, imD, tframe, output
                                      .keyPoints, output.localDes,
                              output.globalDes);        //跟踪线程，有回环检测，后端优化那些
#ifdef COMPILEDWITHC11
            std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
#else
            std::chrono::monotonic_clock::time_point t2 = std::chrono::monotonic_clock::now();
#endif

            double ttrack = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
            vTimesTrack.
                    push_back(ttrack);        //记录的是每一次处理数据的用时

// Wait to load the next frame 等待20ms时间,等下一张图片
            usleep(1000 * 20);

//            //count all
//            if (SLAM.GetTrackingState() != lastState) {
//                ReLocolizationCounter.AddChange(SLAM.GetTrackingState(), tframe, true);
//                lastState = SLAM.GetTrackingState();
//            }
//            if (ni == vstrImageFilenamesRGB.size() - 1) {
//                //to know the last time
//                ReLocolizationCounter.AddChange(1, tframe, true);
//            }
            if (getch()

                != 27) // 按ESC退出
            {
                break;
            }
        }
        SLAM.

                Shutdown();

// Tracking time statistics
        sort(vTimesTrack
                     .

                             begin(), vTimesTrack

                     .

                             end()

        );
        float totaltime = 0;
        for (
                int ni = 0;
                ni < vTimestamps.

                        size();

                ni++) {
            totaltime += vTimesTrack[ni];        //累加所有的运行时间
        }
        cout << "-------" << endl <<
             endl;
        cout << "median tracking time: " << vTimesTrack[vTimestamps.

                size()

                                                        / 2] <<
             endl;
        cout << "mean tracking time: " << totaltime / vTimestamps.

                size()

             <<
             endl;
//这个用来输出效率的平均值和总时间


// Save camera trajectory
// 保存所有的路径文件
        SLAM.SaveTrajectoryTUM("CameraTrajectory.txt");
        SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");

        return 0;
    }
//    ReLocolizationCounter.printsALLLabels(true, true);
    return 0;
}

void LoadImages(const string &strAssociationFilename, vector<string> &vstrImageFilenamesRGB,
                vector<string> &vstrImageFilenamesD, vector<double> &vTimestamps) {
    vstrImageFilenamesD.clear();
    vstrImageFilenamesRGB.clear();
    vTimestamps.clear();            //全部清除
    ifstream fAssociation;            //输入关系
    fAssociation.open(strAssociationFilename.c_str());    //将string转换成C语言中指向字符串的指针
    while (!fAssociation.eof())        //没有到头
    {
        string s;
        getline(fAssociation, s);    //一行一行读取
        if (!s.empty())                //空行不处理，有内容处理
        {
            stringstream ss;
            ss << s;
            double t;
            string sRGB, sD;
            ss >> t;            //第一个位置是（双精度)t，第二个位置是sRGB（字符串类型），第三个位置是t，第四个位置是sD（字符串类型），除了3重复均需要压入到vector数组中
            vTimestamps.push_back(t);        //这个格式是和association.py的输出格式一样的，所以应该先运行association.py，然后运行这个读入
            ss >> sRGB;
            vstrImageFilenamesRGB.push_back(sRGB);
            ss >> t;
            ss >> sD;
            vstrImageFilenamesD.push_back(sD);

        }
    }
}

void getdescriptor(string filename, cv::Mat &descriptor, int nkeypoints) {
    //cnpy库将numpy的数据格式转换为C++用的数据格式
    cnpy::NpyArray arr = cnpy::npy_load(filename);
    for (int i = 0; i < nkeypoints; i++) {        //遍历每一个特征点
        float *pdata = descriptor.ptr<float>(i);    //定义pdata是指向第i+1行的第一个元素的指针
        for (int j = 0; j < 256; j++) {
            float temp = arr.data<float>()[i * 256 + j]; //遍历这一行每一个关键帧的值
            pdata[j] = temp;
        }
    }
    //整个函数将descriptor装满了外部文件中的descriptor的数据
}

void getGlobaldescriptor(string filename, cv::Mat &descriptor) {
    cnpy::NpyArray arr = cnpy::npy_load(filename);
    float *pdata = descriptor.ptr<float>(0);
    for (int j = 0; j < 4096; j++) {
        pdata[j] = arr.data<float>()[j];
    }
}

void getKeyPoint(string filename, vector<cv::KeyPoint> &keyPoints) {
    ifstream getfile(filename);

    for (int i = 0; i < 550 && !getfile.eof(); i++) {
        string s;
        getline(getfile, s);
        if (!s.empty()) {
            stringstream ss;
            ss << s;
            double t_x;
            double t_y;
            ss >> t_x;
            ss >> t_y;
            cv::KeyPoint keyPoint(t_x, t_y, 1);
            //KeyPoint的属性值有angle（角度，表示方向），class_id（对图片惊醒分类的时候，使用的该值做特征点区分）
            //octave表示从金字塔哪一层提取到的数据，pt（二维向量，关键点坐标），reponse（响应程度，好不好？）
            keyPoint.octave = 0;
            keyPoints.push_back(keyPoint);
        }
    }
}
