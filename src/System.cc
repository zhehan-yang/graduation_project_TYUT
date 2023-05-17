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
#include <unistd.h>
#include "System.h"
#include "Converter.h"
#include <thread>
#include <pangolin/pangolin.h>
#include <iomanip>
#include <chrono>

//相关的资料在这里：https://blog.csdn.net/itwaits/article/details/113029296
namespace DXSLAM		//为了避免重复定义而设置的命名空间，里面可以存放函数之类的
{
	//System类下的变量有这些：
#pragma region System类成员变量
	/*
	注意大对象几乎都用的指针
	Vocabulary* mpVocabulary;				词典
	KeyFrameDatabase* mpKeyFrameDatabase;	关键帧数据库
	Map* mpMap;								地图
	Tracking* mpTracker;					Track线程
	LocalMapping* mpLocalMapper;			LocalMapping线程下的对象
	LoopClosing* mpLoopCloser;				LoopCloser线程下的对象
	Viewer* mpViewer;						Viewer线程下的对象
	FrameDrawer* mpFrameDrawer;				绘制用的东西
	MapDrawer* mpMapDrawer;					绘制用的东西
	std::thread* mptLocalMapping;			LocalMapping线程
	std::thread* mptLoopClosing;			LoopClosing线程
	std::thread* mptViewer;					Viewer线程
	std::mutex mMutexReset;					复位锁
	std::mutex mMutexMode;					模式锁
	std::mutex mMutexState;					状态锁
	bool mbReset;							是否需要复位
	bool mbActivateLocalizationMode;		激活定位模式标志
	bool mbDeactivateLocalizationMode;		冻结定位模式标志
	int mTrackingState;						Tracking线程的状态
	std::vector<MapPoint*> mTrackedMapPoints;		？？尚未见过
	std::vector<cv::KeyPoint> mTrackedKeyPointsUn;	？？尚未见过

	*/
#pragma endregion

	//para1:字典文件路径
	//para2:设置文件路径
	//para3:选择的传感器类型，那个文件里头选择的是RGBD
	//para4:是否开启多线程Viewer，即可视化线程，关于多线程涉及的mutex以及unique_lock可以看这篇文章：https://blog.csdn.net/heurobocon/article/details/113542461
System::System(const std::string &strVocFile, const std::string &strSettingsFile, const eSensor sensor,
               const bool bUseViewer):mSensor(sensor), mpViewer(static_cast<Viewer*>(NULL)), mbReset(false),mbActivateLocalizationMode(false),
        mbDeactivateLocalizationMode(false)
{
	//构造函数是不同的
    // Output welcome message	输出欢迎的信息，回车 DXslam 回车 回车
    std::cout << std::endl <<" DXSLAM "<< std::endl << std::endl;


	//下面的传入yml参数和载入vocabulary和那ORB-SLAM2一样的
    //Check settings file		核对设置文件
	//该类将opencv数据类型转化为xml或者yml文件，参数1是存储或者读取的文件名，参数2是操作方式，参数3是编码方式（取默认值即可）
	//这里单纯只是测试能不能打开，具体里面参数怎么定还是要等到后面的初始化变量的过程才会使用到文本内部的内容
    cv::FileStorage fsSettings(strSettingsFile.c_str(), cv::FileStorage::READ);
	//这里配置文件的格式可以看一下ORB-SLAM2里面的变量，关于yml文件可以看一下这个：https://zhuanlan.zhihu.com/p/493137181
    if(!fsSettings.isOpened())		//如果打开失败
    {
       std::cerr << "Failed to open settings file at: " << strSettingsFile << std::endl;
       exit(-1);
    }


    //Load ORB Vocabulary
    std::chrono::steady_clock::time_point  t1 = std::chrono::steady_clock::now();			//和下面的t2形成计时系统
    std::cout << std::endl << "Loading ORB Vocabulary. This could take a while..." << std::endl;	//形成字典

	//下面这两步骤是载入字典的操作，具体参考：https://blog.csdn.net/qq_33236581/article/details/109508345
    mpVocabulary = new Vocabulary();		//fbow::vocabulary类

    mpVocabulary->readFromFile(strVocFile);
	//这个很简单，单纯二进制读入，源码在这里
#pragma region source code
	//void Vocabulary::readFromFile(const std::string &filepath) {
	//	std::ifstream file(filepath, std::ios::binary);		最后那个表示以二进制模式打开
	//	if (!file) throw std::runtime_error("Vocabulary::readFromFile could not open:" + filepath);
	//	fromStream(file);		//这个函数的作用是根据文件里面头上的配置说明分配内存空间，然后读入字典
	//}
	//fromStream:
	//void Vocabulary::fromStream(std::istream &str)
	//{
	//	uint64_t sig;
	//	str.read((char*)&sig, sizeof(sig));	//提取流的长度，并保存？？他怎么知道这么长？？因为刚开始就是这么长
	//	if (sig != 55824124) throw std::runtime_error("Vocabulary::fromStream invalid signature");
	//	//read string
	//	str.read((char*)&_params, sizeof(params));//把他读到结构体里头
	//	_data = std::unique_ptr<char[], decltype(&AlignedFree)>((char*)AlignedAlloc(_params._aligment, _params._total_size), &AlignedFree);
	//	//根据刚才读到的特征进行内存分配，两个参数一个是内存分配，一个是块的数量
	//	if (_data.get() == nullptr) throw std::runtime_error("Vocabulary::fromStream Could not allocate data");
	//	//如果报错，那么抛异常
	//	str.read(_data.get(), _params._total_size);//把后面的读入
	//}
#pragma endregion

    std::chrono::steady_clock::time_point  t2 = std::chrono::steady_clock::now();			//和上面的t1形成计时系统

    std::cout << "Vocabulary loaded!" << std::endl;
    std::cout << "load time:" << std::chrono::duration<double>(t2 - t1).count() * 1000 << std::endl << std::endl;

    //Create KeyFrame Database  创建关键帧的数据库，这个和ORB-SLAM2不同
	//创建过程单纯就是把前面导入的字典存到类的成员变量里头
    mpKeyFrameDatabase = new KeyFrameDatabase(*mpVocabulary);			//这个定义在KeyFrameDatabase.cc里面
	//单纯就这一句：KeyFrameDatabase::KeyFrameDatabase(const Vocabulary &voc) :mpVoc(&voc) {}


    //Create the Map 创建地图 定义在Map.cc中 他的构造任务是初始化最大ID关键帧和最后变动的关键帧
    mpMap = new Map();

    //Create Drawers. These are used by the Viewer  创建绘图工具，这个定义在FramDrawer.cc里面和MapDrawer.cc里面
	//FrameDrawer是用来绘制图像、特征点连线形成的轨迹（初始化时）、框（跟踪时的MapPoint）、圈（跟踪时的特征点）的
    mpFrameDrawer = new FrameDrawer(mpMap);
	//根据yml设置文件，配置Map地图的参数，用的是cv::FileStorage，这个和前头的一样，和ORB_SLAM2一样
    mpMapDrawer = new MapDrawer(mpMap, strSettingsFile);

    //Initialize the Tracking thread		跟踪线程创建，这里主要是读入了配置文件里面的相机参数，深度阈值
    //(it will live in the main thread of execution, the one that called this constructor)
    mpTracker = new Tracking(this, mpVocabulary, mpFrameDrawer, mpMapDrawer,
                             mpMap, mpKeyFrameDatabase, strSettingsFile, mSensor);

    //Initialize the Local Mapping thread and launch	局部地图线程创建，位于LocalMapping.cc中,过偶在函数单纯初始化所有变量，然后传入Map
	//负责对新加入的KeyFrames和MapPoints筛选融合，剔除冗余的KeyFrames和MapPoints，维护稳定的KeyFrame集合，传给后续的LoopClosing线程
    mpLocalMapper = new LocalMapping(mpMap, mSensor==MONOCULAR);
    mptLocalMapping = new std::thread(&DXSLAM::LocalMapping::Run,mpLocalMapper);

    //Initialize the Loop Closing thread and launch		回环检测线程创建，也是把所有的参数读到自己的成员变量里
    mpLoopCloser = new LoopClosing(mpMap, mpKeyFrameDatabase, mpVocabulary, mSensor!=MONOCULAR);
    mptLoopClosing = new std::thread(&DXSLAM::LoopClosing::Run, mpLoopCloser);

    //Initialize the Viewer thread and launch			初始化观察者线程
    if(bUseViewer)		//如果要启用Viewer（可视化）线程，这个默认是打开的
    {
        mpViewer = new Viewer(this, mpFrameDrawer,mpMapDrawer,mpTracker,strSettingsFile);
        mptViewer = new std::thread(&Viewer::Run, mpViewer);
        mpTracker->SetViewer(mpViewer);
    }

    //Set pointers between threads		//把三个连在一起
    mpTracker->SetLocalMapper(mpLocalMapper);
    mpTracker->SetLoopClosing(mpLoopCloser);

    mpLocalMapper->SetTracker(mpTracker);
    mpLocalMapper->SetLoopCloser(mpLoopCloser);

    mpLoopCloser->SetTracker(mpTracker);
    mpLoopCloser->SetLocalMapper(mpLocalMapper);
}

cv::Mat System::TrackRGBD(const cv::Mat &im,
                        const cv::Mat &depthmap,
                        const double &timestamp,
                        const std::vector<cv::KeyPoint> &keypoints,
                        const cv::Mat &local_desc,
                        const cv::Mat &global_desc)
{
		//输入传感器类型错误报错
    if(mSensor!=RGBD)
    {
        std::cerr << "ERROR: you called TrackRGBD but input sensor was not set to RGBD." << std::endl;
        exit(-1);
    }

    // Check mode change
    {
        std::unique_lock<std::mutex> lock(mMutexMode);		//上程序锁
        if(mbActivateLocalizationMode)
        {
			//这句话吧localMap对象中的mbStopRequested停止请求置1，然后把放弃BA置位1
            mpLocalMapper->RequestStop();	//他们几个操作的参数以指针的方式联系在一起

            // Wait until Local Mapping has effectively stopped
            while(!mpLocalMapper->isStopped())		//等待他停下来了，这个需要这个关键帧全部处理完毕
            {
                usleep(1000);
            }

            mpTracker->InformOnlyTracking(true);	//置位的作用，为true就是仅跟踪定位，无重定位，无关键帧
            mbActivateLocalizationMode = false;	//触发器关闭
        }
        if(mbDeactivateLocalizationMode)
        {
            mpTracker->InformOnlyTracking(false);	//反向操作
            mpLocalMapper->Release();				//开启，清空停止请求和停止信号，同时删除所有的关键帧和等待处理的关键帧
            mbDeactivateLocalizationMode = false;
        }
    }

    // Check reset 需要就把他置零，动作完了恢复
    {
    std::unique_lock<std::mutex> lock(mMutexReset);	//复位锁
    if(mbReset)
    {
        mpTracker->Reset();
        mbReset = false;
    }
    }
	//定义在Tracking.cc中
    cv::Mat Tcw = mpTracker->GrabImageRGBD(im, depthmap, timestamp, keypoints, local_desc, global_desc);

    std::unique_lock<std::mutex> lock2(mMutexState);
    mTrackingState = mpTracker->mState;
    mTrackedMapPoints = mpTracker->mCurrentFrame->mvpMapPoints;
    mTrackedKeyPointsUn = mpTracker->mCurrentFrame->mvKeysUn;
    return Tcw;
}

void System::ActivateLocalizationMode()
{
    std::unique_lock<std::mutex> lock(mMutexMode);
    mbActivateLocalizationMode = true;
}

void System::DeactivateLocalizationMode()
{
    std::unique_lock<std::mutex> lock(mMutexMode);
    mbDeactivateLocalizationMode = true;
}

bool System::MapChanged()
{
    static int n=0;
    int curn = mpMap->GetLastBigChangeIdx();
    if(n<curn)
    {
        n=curn;
        return true;
    }
    else
        return false;
}

void System::Reset()
{
    std::unique_lock<std::mutex> lock(mMutexReset);
    mbReset = true;
}

void System::Shutdown()
{
    mpLocalMapper->RequestFinish();
    mpLoopCloser->RequestFinish();
    if(mpViewer)
    {
        mpViewer->RequestFinish();
        while(!mpViewer->isFinished())
            usleep(5000);
    }

    // Wait until all thread have effectively stopped
    while(!mpLocalMapper->isFinished() || !mpLoopCloser->isFinished() || mpLoopCloser->isRunningGBA())
    {
        usleep(5000);
    }

    //if(mpViewer)
      //  pangolin::BindToContext("ORB-SLAM2: Map Viewer");
}

void System::SaveTrajectoryTUM(const std::string &filename)
{
    if(mSensor==MONOCULAR)
    {
        std::cerr << "ERROR: SaveTrajectoryTUM cannot be used for monocular." << std::endl;
        return;
    }

    std::vector<KeyFrame*> vpKFs = mpMap->GetAllKeyFrames();
    sort(vpKFs.begin(),vpKFs.end(),KeyFrame::lId);

    // Transform all keyframes so that the first keyframe is at the origin.
    // After a loop closure the first keyframe might not be at the origin.
    cv::Mat Two = vpKFs[0]->GetPoseInverse();

    std::ofstream f;
    f.open(filename.c_str());
    f << std::fixed;

    // Frame pose is stored relative to its reference keyframe (which is optimized by BA and pose graph).
    // We need to get first the keyframe pose and then concatenate the relative transformation.
    // Frames not localized (tracking failure) are not saved.

    // For each frame we have a reference keyframe (lRit), the timestamp (lT) and a flag
    // which is true when tracking failed (lbL).
    std::list<DXSLAM::KeyFrame*>::iterator lRit = mpTracker->mlpReferences.begin();
    std::list<double>::iterator lT = mpTracker->mlFrameTimes.begin();
    std::list<bool>::iterator lbL = mpTracker->mlbLost.begin();
    for(std::list<cv::Mat>::iterator lit=mpTracker->mlRelativeFramePoses.begin(),
        lend=mpTracker->mlRelativeFramePoses.end();lit!=lend;lit++, lRit++, lT++, lbL++)
    {
        if(*lbL)
            continue;

        KeyFrame* pKF = *lRit;

        cv::Mat Trw = cv::Mat::eye(4,4,CV_32F);

        // If the reference keyframe was culled, traverse the spanning tree to get a suitable keyframe.
        while(pKF->isBad())
        {
            Trw = Trw*pKF->mTcp;
            pKF = pKF->GetParent();
        }

        Trw = Trw*pKF->GetPose()*Two;

        cv::Mat Tcw = (*lit)*Trw;
        cv::Mat Rwc = Tcw.rowRange(0,3).colRange(0,3).t();
        cv::Mat twc = -Rwc*Tcw.rowRange(0,3).col(3);

        std::vector<float> q = Converter::toQuaternion(Rwc);

        f << std::setprecision(6) << *lT << " " <<  std::setprecision(9) << twc.at<float>(0) << " " << twc.at<float>(1) << " " << twc.at<float>(2) << " " << q[0] << " " << q[1] << " " << q[2] << " " << q[3] << std::endl;
    }
    f.close();
}


void System::SaveKeyFrameTrajectoryTUM(const std::string &filename)
{
    std::cout << std::endl << "Saving keyframe trajectory to " << filename << " ..." << std::endl;

    std::vector<KeyFrame*> vpKFs = mpMap->GetAllKeyFrames();
    sort(vpKFs.begin(),vpKFs.end(),KeyFrame::lId);

    // Transform all keyframes so that the first keyframe is at the origin.
    // After a loop closure the first keyframe might not be at the origin.
    //cv::Mat Two = vpKFs[0]->GetPoseInverse();

    std::ofstream f;
    f.open(filename.c_str());
    f << std::fixed;

    for(size_t i=0; i<vpKFs.size(); i++)
    {
        KeyFrame* pKF = vpKFs[i];

       // pKF->SetPose(pKF->GetPose()*Two);

        if(pKF->isBad())
            continue;

        cv::Mat R = pKF->GetRotation().t();
        std::vector<float> q = Converter::toQuaternion(R);
        cv::Mat t = pKF->GetCameraCenter();
        f << std::setprecision(6) << pKF->mTimeStamp << std::setprecision(7) << " " << t.at<float>(0) << " " << t.at<float>(1) << " " << t.at<float>(2)
          << " " << q[0] << " " << q[1] << " " << q[2] << " " << q[3] << std::endl;

    }

    f.close();
    std::cout << std::endl << "trajectory saved!" << std::endl;
}

void System::SaveTrajectoryKITTI(const std::string &filename)
{
    if(mSensor==MONOCULAR)
    {
        std::cerr << "ERROR: SaveTrajectoryKITTI cannot be used for monocular." << std::endl;
        return;
    }

    std::vector<KeyFrame*> vpKFs = mpMap->GetAllKeyFrames();
    sort(vpKFs.begin(),vpKFs.end(),KeyFrame::lId);

    // Transform all keyframes so that the first keyframe is at the origin.
    // After a loop closure the first keyframe might not be at the origin.
    cv::Mat Two = vpKFs[0]->GetPoseInverse();

    std::ofstream f;
    f.open(filename.c_str());
    f << std::fixed;

    // Frame pose is stored relative to its reference keyframe (which is optimized by BA and pose graph).
    // We need to get first the keyframe pose and then concatenate the relative transformation.
    // Frames not localized (tracking failure) are not saved.

    // For each frame we have a reference keyframe (lRit), the timestamp (lT) and a flag
    // which is true when tracking failed (lbL).
    std::list<DXSLAM::KeyFrame*>::iterator lRit = mpTracker->mlpReferences.begin();
    std::list<double>::iterator lT = mpTracker->mlFrameTimes.begin();
    for(std::list<cv::Mat>::iterator lit=mpTracker->mlRelativeFramePoses.begin(), lend=mpTracker->mlRelativeFramePoses.end();lit!=lend;lit++, lRit++, lT++)
    {
        DXSLAM::KeyFrame* pKF = *lRit;

        cv::Mat Trw = cv::Mat::eye(4,4,CV_32F);

        while(pKF->isBad())
        {
            Trw = Trw*pKF->mTcp;
            pKF = pKF->GetParent();
        }

        Trw = Trw*pKF->GetPose()*Two;

        cv::Mat Tcw = (*lit)*Trw;
        cv::Mat Rwc = Tcw.rowRange(0,3).colRange(0,3).t();
        cv::Mat twc = -Rwc*Tcw.rowRange(0,3).col(3);

        f << std::setprecision(9) << Rwc.at<float>(0,0) << " " << Rwc.at<float>(0,1)  << " " << Rwc.at<float>(0,2) << " "  << twc.at<float>(0) << " " <<
             Rwc.at<float>(1,0) << " " << Rwc.at<float>(1,1)  << " " << Rwc.at<float>(1,2) << " "  << twc.at<float>(1) << " " <<
             Rwc.at<float>(2,0) << " " << Rwc.at<float>(2,1)  << " " << Rwc.at<float>(2,2) << " "  << twc.at<float>(2) << std::endl;
    }
    f.close();
    std::cout << std::endl << "trajectory saved!" << std::endl;
}

int System::GetTrackingState()
{
    std::unique_lock<std::mutex> lock(mMutexState);
    return mTrackingState;
}

std::vector<MapPoint*> System::GetTrackedMapPoints()
{
    std::unique_lock<std::mutex> lock(mMutexState);
    return mTrackedMapPoints;
}

std::vector<cv::KeyPoint> System::GetTrackedKeyPointsUn()
{
    std::unique_lock<std::mutex> lock(mMutexState);
    return mTrackedKeyPointsUn;
}

} //namespace ORB_SLAM
