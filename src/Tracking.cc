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
#include "Tracking.h"
#include <unistd.h>
#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>
#include<set>
#include"Matcher.h"
#include"FrameDrawer.h"
#include"Converter.h"
#include"Map.h"
#include"Initializer.h"

#include"Optimizer.h"
#include"PnPsolver.h"

#include<iostream>

#include<mutex>


using namespace std;

namespace DXSLAM
{
#pragma region Track类成员变量
	/*
	bool mbVO;								？？这个干嘛用的，不过初始化时false
	这些都是通过指针的方式传入的
	LocalMapping* mpLocalMapper;			system下LocalMapping线程下的对象，通过SetLocalMapper传入
	LoopClosing* mpLoopClosing;				system下LoopCloser线程下的对象，通过SetLoopClosing传入，原因是Track是先创建的
	Vocabulary* mpORBVocabulary;			system下的词典，但命名不一样，通过构造函数传入
	KeyFrameDatabase* mpKeyFrameDB;			system下的关键帧数据库，命名缩写了，通过构造函数传入
	Initializer* mpInitializer;				尚未见过
	KeyFrame* mpReferenceKF;				参考关键帧？初始化的时候刚刚有
	System* mpSystem;						向上指根节点，传入系统对象
	Viewer* mpViewer;						system下viewer线程下的对象，不通过构造函数传入，因其选择性，由SetViewer方法传入
	FrameDrawer* mpFrameDrawer;				这两个都是画图用的东西，由构造函数传入
	MapDrawer* mpMapDrawer;					
	Map* mpMap;								地图，由构造函数传入
	以上是system系统给的

	std::vector<KeyFrame*> mvpLocalKeyFrames;	所有关键帧
	std::vector<MapPoint*> mvpLocalMapPoints;	所有地图点

	下面这部分都是参数，由yml文件导入
	cv::Mat mK;
	cv::Mat mDistCoef;
	float mbf;			基线*fx
	int mMinFrames;		最大最小插入关键帧频率
	int mMaxFrames;
	float mThDepth;		深度阈值，以基线倍数表示
	float mDepthMapFactor;	相机放缩倍数

	int mnMatchesInliers;
	KeyFrame* mpLastKeyFrame;				上一个关键帧
	std::shared_ptr<Frame> mLastFrame;		上一个帧
	unsigned int mnLastKeyFrameId;			上一个关键帧ID
	unsigned int mnLastRelocFrameId;		
	cv::Mat mVelocity;
	bool mbRGB;
	std::list<MapPoint*> mlpTemporalPoints;
	bool mbOnlyTracking 用于指示是否是-仅跟踪模式，如果是，那么就不需要定位，不需要关键帧处理了

	以下是公有参数：
	eTrackingState mState;					枚举类型，表征tracking状态
	eTrackingState mLastProcessedState;		枚举类型，表征上一次处理完的状态
	int mSensor;							代表传感器类型
	std::shared_ptr<Frame> mCurrentFrame;	表示当前正在处理的帧
	cv::Mat mImGray;						用于存储外部输入的图像
	std::vector<int> mvIniLastMatches;
	std::vector<int> mvIniMatches;
	std::vector<cv::Point2f> mvbPrevMatched;
	std::vector<cv::Point3f> mvIniP3D;
	Frame mInitialFrame;
	std::list<cv::Mat> mlRelativeFramePoses;
	std::list<KeyFrame*> mlpReferences;
	std::list<double> mlFrameTimes;
	std::list<bool> mlbLost;
	bool mbOnlyTracking ;
*/
#pragma endregion

Tracking::Tracking(System *pSys, Vocabulary* pVoc, FrameDrawer *pFrameDrawer, MapDrawer *pMapDrawer, Map *pMap, KeyFrameDatabase* pKFDB, const std::string &strSettingPath, const int sensor):
    mState(NO_IMAGES_YET), mSensor(sensor), mbOnlyTracking(false), mbVO(false), mpORBVocabulary(pVoc),	//mbonlyTracking是仅跟踪的意思，可以自己打开
    mpKeyFrameDB(pKFDB), mpInitializer(static_cast<Initializer*>(NULL)), mpSystem(pSys), mpViewer(NULL),
    mpFrameDrawer(pFrameDrawer), mpMapDrawer(pMapDrawer), mpMap(pMap), mnLastRelocFrameId(0)
{
    // Load camera parameters from settings file
	//录入所有在yml中的相机参数数据
    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];

    cv::Mat K = cv::Mat::eye(3,3,CV_32F);	//单位矩阵，对角矩阵
    K.at<float>(0,0) = fx;
    K.at<float>(1,1) = fy;
    K.at<float>(0,2) = cx;
    K.at<float>(1,2) = cy;
    K.copyTo(mK);		//传递给成员变量
	
	//畸变数据，其中，k3可能是不存在的数据
    cv::Mat DistCoef(4,1,CV_32F);		//Distort Coefficient
    DistCoef.at<float>(0) = fSettings["Camera.k1"];
    DistCoef.at<float>(1) = fSettings["Camera.k2"];
    DistCoef.at<float>(2) = fSettings["Camera.p1"];
    DistCoef.at<float>(3) = fSettings["Camera.p2"];
	//[k1]
	//[k2]
	//[p1]
	//[p2]
	//([k3])
    const float k3 = fSettings["Camera.k3"];
    if(k3!=0)
    {
        DistCoef.resize(5);
        DistCoef.at<float>(4) = k3;
    }
    DistCoef.copyTo(mDistCoef);

    mbf = fSettings["Camera.bf"];		//视差？这个叫基线，是双目相机的概念，是基线的实际距离乘以fx

	//采样频率
    float fps = fSettings["Camera.fps"];
    if(fps==0)
        fps=30;

    // Max/Min Frames to insert keyframes and to check relocalisation
    mMinFrames = 0;		//好像是？？最大和最小的插入关键帧的频率
    mMaxFrames = fps;

	//把采来的数据输出到控制台上
    std::cout << std::endl << "Camera Parameters: " << std::endl;
    std::cout << "- fx: " << fx << std::endl;
    std::cout << "- fy: " << fy << std::endl;
    std::cout << "- cx: " << cx << std::endl;
    std::cout << "- cy: " << cy << std::endl;
    std::cout << "- k1: " << DistCoef.at<float>(0) << std::endl;
    std::cout << "- k2: " << DistCoef.at<float>(1) << std::endl;
    if(DistCoef.rows==5)
        std::cout << "- k3: " << DistCoef.at<float>(4) << std::endl;
    std::cout << "- p1: " << DistCoef.at<float>(2) << std::endl;
    std::cout << "- p2: " << DistCoef.at<float>(3) << std::endl;
    std::cout << "- fps: " << fps << std::endl;


    int nRGB = fSettings["Camera.RGB"];	//可以查yml文件，这里指的是RGB的规定顺序，1表示RGB，0表示BGR，如果是灰度的话，这个参数不重要
    mbRGB = nRGB;

    if(mbRGB)
        std::cout << "- color order: RGB (ignored if grayscale)" << std::endl;
    else
        std::cout << "- color order: BGR (ignored if grayscale)" << std::endl;

    if(sensor==System::STEREO || sensor==System::RGBD)
    {
		//这里ThDepth是深度阈值，用于区分哪一些点是远点，哪一些点是近点，yml取得是40，近点容易做三角化，远点需要使用多点进行三角化，以保证精度
        mThDepth = mbf*(float)fSettings["ThDepth"]/fx;	//公式为基线距离乘以倍数ThDepth
        std::cout << std::endl << "Depth Threshold (Close/Far Points): " << mThDepth << std::endl;
    }

    if(sensor==System::RGBD)
    {
		//这里DepthMapFactor是一个比例系数，指放缩系数，；例如，放缩系数是5000，深度图像中像素值是5000，那么对应理摄像机1m远
        mDepthMapFactor = fSettings["DepthMapFactor"];
        if(fabs(mDepthMapFactor)<1e-5)
            mDepthMapFactor=1;
        else
            mDepthMapFactor = 1.0f/mDepthMapFactor;		//方便求解
    }

}

void Tracking::SetLocalMapper(LocalMapping *pLocalMapper)
{
    mpLocalMapper=pLocalMapper;
}

void Tracking::SetLoopClosing(LoopClosing *pLoopClosing)
{
    mpLoopClosing=pLoopClosing;
}

void Tracking::SetViewer(Viewer *pViewer)
{
    mpViewer=pViewer;
}


cv::Mat Tracking::GrabImageRGBD(const cv::Mat &imRGB,	//色彩图像
                            const cv::Mat &imD,			//深度图像
                            const double &timestamp,	//实际时间
                            const std::vector<cv::KeyPoint> &keypoints,	//特征点
                            const cv::Mat &local_desc,	//描述子
                            const cv::Mat &global_desc)	//全局描述子
{
	//cv::Mat格式
    mImGray = imRGB;
    cv::Mat imDepth = imD;

    if(mImGray.channels()==3)			//三通道
    {
		//这个函数用来转换图片的格式将RGB图转换为灰度图像
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGB2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGR2GRAY);
    }
    else if(mImGray.channels()==4)		//RGB加上alpha
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGBA2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGRA2GRAY);
    }

    if((fabs(mDepthMapFactor-1.0f)>1e-5) || imDepth.type()!=CV_32F)		//放缩系数离1有一定距离，或者他的类型不是0~1的这种
        imDepth.convertTo(imDepth,CV_32F,mDepthMapFactor);				//将深度坐标进行放缩，同时转换格式为32位深度浮点的形式
	//放缩，参数依次是图像，需要转换的数据类型，放缩系数，（偏移系数）

	//make_shared用于返回指定类型的指针
    mCurrentFrame = std::make_shared<Frame>(mImGray,
                                            imDepth,
                                            timestamp,
                                            mpORBVocabulary,
                                            mK, mDistCoef,
                                            mbf,
                                            mThDepth,
                                            keypoints,
                                            local_desc,
                                            global_desc);	//是帧的类
	//创建一个帧，这里做了将相机等各种参数的导入帧类里，并编号，然后把里面的关键帧全部放到网格里
	//你可以理解为上面全在初始化，下面才是正片
    Track();

    return mCurrentFrame->mTcw.clone();	//返回当前的位置
}


void Tracking::Track()
{
	//初始化完成，mState的值是这个
    if(mState==NO_IMAGES_YET)		//定义在Tracking中，是枚举类型，为0对应尚未定义
    {
        mState = NOT_INITIALIZED;
    }

    mLastProcessedState=mState;	//上一次运行完后的状态

    // Get Map Mutex -> Map cannot be changed
    std::unique_lock<std::mutex> lock(mpMap->mMutexMapUpdate);	//地图更新锁

    if(mState==NOT_INITIALIZED)	//对应1，尚未被初始化，枚举类型eTrackingState
    {
        if(mSensor==System::STEREO || mSensor==System::RGBD)
            StereoInitialization();	//这个需要有200个关键点才会开始动

        mpFrameDrawer->Update(this);	//要求Tracking指针

        if(mState!=OK)
            return;		//如果上面那个：Stero啥的没有200个特征点走这条路
    }
    else
    {
        // System is initialized. Track Frame. 如果已经完成了初始化任务
        bool bOK;

        // Initial camera pose estimation using motion model or relocalization (if tracking is lost)
        if(!mbOnlyTracking)		//两个都会打开
        {
            // Local Mapping is activated. This is the normal behaviour, unless
            // you explicitly activate the "only tracking" mode.

            if(mState==OK)		//状态正常
            {
                // Local Mapping might have changed some MapPoints tracked in last frame
                CheckReplacedInLastFrame();		//替代地图点

                if(mVelocity.empty() || mCurrentFrame->mnId<mnLastRelocFrameId+2)
                {
                    bOK = TrackReferenceKeyFrame();
                }
                else
                {
                    bOK = TrackWithMotionModel();
                    if(!bOK)
                        bOK = TrackReferenceKeyFrame();
                }
            }
            else
            {
                bOK = Relocalization();
            }
        }
        else
        {
            // Localization Mode: Local Mapping is deactivated
			//这种模式下，首先会进行重定位，然后根据mbVO 参数判断是进行正常的跟踪定位操作还是要结合重定位信息。mbVO为真表示当前图像和上一帧地图点匹配数目小于10，有可能是运动过快的原因。
			//	假如上次定位显示，mbVO为0，则进行正常的定位跟踪，但是发现跟踪后mbVO在TrackWithMotionModel里被设置为1了，且定位是成功的，则进行正常的更新速度以及显示操作，如果mbVO仍然为0，则要先跟局部地图进行匹配跟踪，优化位姿，再进行后续操作。因为如果mbVO为1时，也就是跟上一帧地图点匹配较少时，可能得不到有效的局部地图信息。
			//	在下次定位时，如果mbVO为1，则先进行TrackWithMotionModel跟踪，再进行重定位，为的是保证定位不会轻易丢失。但是如果运动速度仍然过快，mbVO 仍然为1，则下次任然重复步骤2。直到TrackWithMotionModel里设置mbVO为0，或者重定位成功把mbVO设置为0。
			//也就是期望值是1
            if(mState==LOST)
            {
                bOK = Relocalization();
            }
            else
            {
                if(!mbVO)		//正常
                {
                    // In last frame we tracked enough MapPoints in the map
                    if(!mVelocity.empty())	//速度非空
                    {
                        bOK = TrackWithMotionModel();
                    }
                    else  //否则
                    {
                        bOK = TrackReferenceKeyFrame();
                    }
                }
                else
                {
                    // In last frame we tracked mainly "visual odometry" points.

                    // We compute two camera poses, one from motion model and one doing relocalization.
                    // If relocalization is sucessfull we choose that solution, otherwise we retain
                    // the "visual odometry" solution.

                    bool bOKMM = false;
                    bool bOKReloc = false;
                    std::vector<MapPoint*> vpMPsMM;
                    std::vector<bool> vbOutMM;
                    cv::Mat TcwMM;
                    if(!mVelocity.empty())
                    {
                        bOKMM = TrackWithMotionModel();
                        vpMPsMM = mCurrentFrame->mvpMapPoints;
                        vbOutMM = mCurrentFrame->mvbOutlier;
                        TcwMM = mCurrentFrame->mTcw.clone();
                    }
                    bOKReloc = Relocalization();

                    if(bOKMM && !bOKReloc)
                    {
                        mCurrentFrame->SetPose(TcwMM);
                        mCurrentFrame->mvpMapPoints = vpMPsMM;
                        mCurrentFrame->mvbOutlier = vbOutMM;

                        if(mbVO)
                        {
                            for(int i =0; i<mCurrentFrame->N; i++)
                            {
                                if(mCurrentFrame->mvpMapPoints[i] && !mCurrentFrame->mvbOutlier[i])
                                {
                                    mCurrentFrame->mvpMapPoints[i]->IncreaseFound();
                                }
                            }
                        }
                    }
                    else if(bOKReloc)
                    {
                        mbVO = false;
                    }

                    bOK = bOKReloc || bOKMM;
                }
            }
        }

        mCurrentFrame->mpReferenceKF = mpReferenceKF;

        // If we have an initial estimation of the camera pose and matching. Track the local map.
        if(!mbOnlyTracking)
        {
            if(bOK) {
                bOK = TrackLocalMap();
            }
        }
        else
        {
            // mbVO true means that there are few matches to MapPoints in the map. We cannot retrieve
            // a local map and therefore we do not perform TrackLocalMap(). Once the system relocalizes
            // the camera we will use the local map again.
            if(bOK && !mbVO)
                bOK = TrackLocalMap();
        }

        if(bOK)
            mState = OK;
        else
            mState=LOST;

        // Update drawer
        mpFrameDrawer->Update(this);

        // If tracking were good, check if we insert a keyframe  跟踪好，那么就插入这个关键帧
        if(bOK)
        {
            // Update motion model
            if(!mLastFrame->mTcw.empty())
            {
                cv::Mat LastTwc = cv::Mat::eye(4,4,CV_32F);
                mLastFrame->GetRotationInverse().copyTo(LastTwc.rowRange(0,3).colRange(0,3));
                mLastFrame->GetCameraCenter().copyTo(LastTwc.rowRange(0,3).col(3));
                mVelocity = mCurrentFrame->mTcw*LastTwc;
            }
            else
                mVelocity = cv::Mat();

            mpMapDrawer->SetCurrentCameraPose(mCurrentFrame->mTcw);

            // Clean VO matches
            for(int i=0; i<mCurrentFrame->N; i++)
            {
                MapPoint* pMP = mCurrentFrame->mvpMapPoints[i];
                if(pMP)
                    if(pMP->Observations()<1)
                    {
                        mCurrentFrame->mvbOutlier[i] = false;
                        mCurrentFrame->mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                    }
            }

            // Delete temporal MapPoints
            for(std::list<MapPoint*>::iterator lit = mlpTemporalPoints.begin(), lend =  mlpTemporalPoints.end(); lit!=lend; lit++)
            {
                MapPoint* pMP = *lit;
                delete pMP;
            }
            mlpTemporalPoints.clear();

            // Check if we need to insert a new keyframe
            if(NeedNewKeyFrame())
                CreateNewKeyFrame();

            // We allow points with high innovation (considererd outliers by the Huber Function)
            // pass to the new keyframe, so that bundle adjustment will finally decide
            // if they are outliers or not. We don't want next frame to estimate its position
            // with those points so we discard them in the frame.
            for(int i=0; i<mCurrentFrame->N;i++)
            {
                if(mCurrentFrame->mvpMapPoints[i] && mCurrentFrame->mvbOutlier[i])
                    mCurrentFrame->mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
            }
        }

        // Reset if the camera get lost soon after initialization
        if(mState==LOST)
        {
            if(mpMap->KeyFramesInMap()<=5)
            {
                std::cout << "Track lost soon after initialisation, reseting..." << std::endl;
                mpSystem->Reset();
                return;
            }
        }

        if(!mCurrentFrame->mpReferenceKF)
            mCurrentFrame->mpReferenceKF = mpReferenceKF;

        mLastFrame = mCurrentFrame;
    }

    // Store frame pose information to retrieve the complete camera trajectory afterwards.
    if(!mCurrentFrame->mTcw.empty())
    {
		//能跟上
        cv::Mat Tcr = mCurrentFrame->mTcw*mCurrentFrame->mpReferenceKF->GetPoseInverse(); //相对变化
        mlRelativeFramePoses.push_back(Tcr);		//相对于参考关键帧的位置变化
        mlpReferences.push_back(mpReferenceKF);		//？少了？压入参考关键帧
        mlFrameTimes.push_back(mCurrentFrame->mTimeStamp);	//压入时间戳
        mlbLost.push_back(mState==LOST);				//？？啥玩意，是没用的参数吧？
    }
    else
    {
		//跟丢了
        // This can happen if tracking is lost
        mlRelativeFramePoses.push_back(mlRelativeFramePoses.back());	//全部用上一个代替
        mlpReferences.push_back(mlpReferences.back());
        mlFrameTimes.push_back(mlFrameTimes.back());
        mlbLost.push_back(mState==LOST);
    }

}

//双目初始化
void Tracking::StereoInitialization()
{	
    if(mCurrentFrame->N>200)	//如果所在的帧数关键点数目大于200，那么开始初始化，认为是第一个关键帧
    {
		//需要注意的是，直到第一个关键帧，他才会开始建图工作，第一个关键帧位姿设置为1
        // Set Frame pose to the origin
        mCurrentFrame->SetPose(cv::Mat::eye(4,4,CV_32F));	//设置他的位姿是对角的这种

        // Create KeyFrame
        KeyFrame* pKFini = new KeyFrame(*mCurrentFrame,mpMap,mpKeyFrameDB);//将当前帧设置为关键帧

        // Insert KeyFrame in the map
        mpMap->AddKeyFrame(pKFini);		//在地图里加入第一个关键帧，构造函数基本上完成了拷贝工作

        // Create MapPoints and asscoiate to KeyFrame
        for(int i=0; i<mCurrentFrame->N;i++)	//遍历所有的关键点
        {
            float z = mCurrentFrame->mvDepth[i];	//这个关键点的深度信息
            if(z>0)
            {
                cv::Mat x3D = mCurrentFrame->UnprojectStereo(i);	//获得地面坐标系的坐标
                MapPoint* pNewMP = new MapPoint(x3D,pKFini,mpMap);	//新增一个地图点，位姿，关键帧，地图
                pNewMP->AddObservation(pKFini,i);					//地图点能观测到的点
                pKFini->AddMapPoint(pNewMP,i);						//互相
                pNewMP->ComputeDistinctiveDescriptors();			//更新最佳描述子
                pNewMP->UpdateNormalAndDepth();						//更新地图点的平均观测方向和距离
                mpMap->AddMapPoint(pNewMP);							//把地图点增加到地图中

                mCurrentFrame->mvpMapPoints[i]=pNewMP;
            }
        }

		//输出新地图中有多少个地图点
        std::cout << "New map created with " << mpMap->MapPointsInMap() << " points" << std::endl;

        mpLocalMapper->InsertKeyFrame(pKFini);		//将需要处理的新的帧插入到等待序列中

        mLastFrame = mCurrentFrame;					//上一个处理的帧
        mnLastKeyFrameId=mCurrentFrame->mnId;		//帧ID
        mpLastKeyFrame = pKFini;					//上一个关键帧

        mvpLocalKeyFrames.push_back(pKFini);		//保存关键帧
        mvpLocalMapPoints=mpMap->GetAllMapPoints();	//获得所有的地图点
        mpReferenceKF = pKFini;						//参考关键帧设置为第一个关键帧
        mCurrentFrame->mpReferenceKF = pKFini;		//当前帧的参考关键帧设置为第一个关键帧

        mpMap->SetReferenceMapPoints(mvpLocalMapPoints);	//把刚刚初始化的地图点全部设置成参考地图点

        mpMap->mvpKeyFrameOrigins.push_back(pKFini);		//最初关键帧

        mpMapDrawer->SetCurrentCameraPose(mCurrentFrame->mTcw);//设置  就是单位矩阵嘛？

        mState=OK;		//初始化完成
    }
}

//这个函数作用是吧所有的上一个帧的地图点尽可能全部替换成替代地图点，这种地图但在呼唤的时候可能有用
void Tracking::CheckReplacedInLastFrame()
{
    for(int i =0; i<mLastFrame->N; i++)
    {
        MapPoint* pMP = mLastFrame->mvpMapPoints[i]; //遍历他的每一个地图点，也就是关键点

        if(pMP)		//没被删？
        {
            MapPoint* pRep = pMP->GetReplaced();		//返回的是地图点？替代地图点
            if(pRep)
            {
                mLastFrame->mvpMapPoints[i] = pRep;		//如果有，那么就用替代地图点
            }
        }
    }
}


bool Tracking::TrackReferenceKeyFrame()
{
    // Compute Bag of Words std::vector
    mCurrentFrame->ComputeBoW();

    // We perform first an ORB matching with the reference keyframe
    // If enough matches are found we setup a PnP solver
    Matcher matcher(0.7, true);
    std::vector<MapPoint*> vpMapPointMatches;

    int nmatches = matcher.SearchByBoW(mpReferenceKF,*mCurrentFrame,vpMapPointMatches);
    std::cout << "match numbers: " << nmatches << std::endl;
    if(nmatches<15)
        return false;

    mCurrentFrame->mvpMapPoints = vpMapPointMatches;
    mCurrentFrame->SetPose(mLastFrame->mTcw);

    Optimizer::PoseOptimization(&*mCurrentFrame);

    // Discard outliers
    int nmatchesMap = 0;
    for(int i =0; i<mCurrentFrame->N; i++)
    {
        if(mCurrentFrame->mvpMapPoints[i])
        {
            if(mCurrentFrame->mvbOutlier[i])
            {
                MapPoint* pMP = mCurrentFrame->mvpMapPoints[i];

                mCurrentFrame->mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                mCurrentFrame->mvbOutlier[i]=false;
                pMP->mbTrackInView = false;
                pMP->mnLastFrameSeen = mCurrentFrame->mnId;
                nmatches--;
            }
            else if(mCurrentFrame->mvpMapPoints[i]->Observations()>0)
                nmatchesMap++;
        }
    }
    std::cout << "nmatchesMap: " << nmatchesMap << std::endl;
    return nmatchesMap>=10;
}

void Tracking::UpdateLastFrame()
{
    // Update pose according to reference keyframe
    KeyFrame* pRef = mLastFrame->mpReferenceKF;
    cv::Mat Tlr = mlRelativeFramePoses.back();

    mLastFrame->SetPose(Tlr*pRef->GetPose());

    if(mnLastKeyFrameId==mLastFrame->mnId || mSensor==System::MONOCULAR || !mbOnlyTracking)
        return;

    // Create "visual odometry" MapPoints
    // We sort points according to their measured depth by the stereo/RGB-D sensor
    std::vector<std::pair<float,int> > vDepthIdx;
    vDepthIdx.reserve(mLastFrame->N);
    for(int i=0; i<mLastFrame->N;i++)
    {
        float z = mLastFrame->mvDepth[i];
        if(z>0)
        {
            vDepthIdx.push_back(std::make_pair(z,i));
        }
    }

    if(vDepthIdx.empty())
        return;

    sort(vDepthIdx.begin(),vDepthIdx.end());

    // We insert all close points (depth<mThDepth)
    // If less than 100 close points, we insert the 100 closest ones.
    int nPoints = 0;
    for(size_t j=0; j<vDepthIdx.size();j++)
    {
        int i = vDepthIdx[j].second;

        bool bCreateNew = false;

        MapPoint* pMP = mLastFrame->mvpMapPoints[i];
        if(!pMP)
            bCreateNew = true;
        else if(pMP->Observations()<1)
        {
            bCreateNew = true;
        }

        if(bCreateNew)
        {
            cv::Mat x3D = mLastFrame->UnprojectStereo(i);
            MapPoint* pNewMP = new MapPoint(x3D,mpMap,&*mLastFrame,i);

            mLastFrame->mvpMapPoints[i]=pNewMP;

            mlpTemporalPoints.push_back(pNewMP);
            nPoints++;
        }
        else
        {
            nPoints++;
        }

        if(vDepthIdx[j].first>mThDepth && nPoints>100)
            break;
    }
}

bool Tracking::TrackWithMotionModel()
{
    Matcher matcher(0.9, true);

    // Update last frame pose according to its reference keyframe
    // Create "visual odometry" points if in Localization Mode
    UpdateLastFrame();

    mCurrentFrame->SetPose(mVelocity*mLastFrame->mTcw);

    fill(mCurrentFrame->mvpMapPoints.begin(),mCurrentFrame->mvpMapPoints.end(),static_cast<MapPoint*>(NULL));

    // Project points seen in previous frame
    int th;
    if(mSensor!=System::STEREO)
        th=15;
    else
        th=7;
    int nmatches = matcher.SearchByProjection(*mCurrentFrame,*mLastFrame,th,mSensor==System::MONOCULAR);

    // If few matches, uses a wider window search
    if(nmatches<20)
    {
        fill(mCurrentFrame->mvpMapPoints.begin(),mCurrentFrame->mvpMapPoints.end(),static_cast<MapPoint*>(NULL));
        nmatches = matcher.SearchByProjection(*mCurrentFrame,*mLastFrame,2*th,mSensor==System::MONOCULAR);
    }

    if(nmatches<20)
        return false;

    // Optimize frame pose with all matches
    Optimizer::PoseOptimization(&*mCurrentFrame);

    // Discard outliers
    int nmatchesMap = 0;
    for(int i =0; i<mCurrentFrame->N; i++)
    {
        if(mCurrentFrame->mvpMapPoints[i])
        {
            if(mCurrentFrame->mvbOutlier[i])
            {
                MapPoint* pMP = mCurrentFrame->mvpMapPoints[i];

                mCurrentFrame->mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                mCurrentFrame->mvbOutlier[i]=false;
                pMP->mbTrackInView = false;
                pMP->mnLastFrameSeen = mCurrentFrame->mnId;
                nmatches--;
            }
            else if(mCurrentFrame->mvpMapPoints[i]->Observations()>0)
                nmatchesMap++;
        }
    }

    if(mbOnlyTracking)
    {
        mbVO = nmatchesMap<10;
        return nmatches>20;
    }

    return nmatchesMap>=10;
}

bool Tracking::TrackLocalMap()
{
    // We have an estimation of the camera pose and some map points tracked in the frame.
    // We retrieve the local map and try to find matches to points in the local map.

    UpdateLocalMap();

    SearchLocalPoints();

    // Optimize Pose
    Optimizer::PoseOptimization(&*mCurrentFrame);
    mnMatchesInliers = 0;

    // Update MapPoints Statistics
    for(int i=0; i<mCurrentFrame->N; i++)
    {
        if(mCurrentFrame->mvpMapPoints[i])
        {
            if(!mCurrentFrame->mvbOutlier[i])
            {
                mCurrentFrame->mvpMapPoints[i]->IncreaseFound();
                if(!mbOnlyTracking)
                {
                    if(mCurrentFrame->mvpMapPoints[i]->Observations()>0)
                        mnMatchesInliers++;
                }
                else
                    mnMatchesInliers++;
            }
            else if(mSensor==System::STEREO)
                mCurrentFrame->mvpMapPoints[i] = static_cast<MapPoint*>(NULL);

        }
    }

    // Decide if the tracking was succesful
    // More restrictive if there was a relocalization recently
    if(mCurrentFrame->mnId<mnLastRelocFrameId+mMaxFrames && mnMatchesInliers<50)
        return false;

    if(mnMatchesInliers<30)
        return false;
    else
        return true;
}

bool Tracking::NeedNewKeyFrame()
{
    if(mbOnlyTracking)
        return false;

    // If Local Mapping is freezed by a Loop Closure do not insert keyframes
    if(mpLocalMapper->isStopped() || mpLocalMapper->stopRequested())
        return false;

    const int nKFs = mpMap->KeyFramesInMap();

    // Do not insert keyframes if not enough frames have passed from last relocalisation
    if(mCurrentFrame->mnId<mnLastRelocFrameId+mMaxFrames && nKFs>mMaxFrames)
        return false;

    // Tracked MapPoints in the reference keyframe
    int nMinObs = 3;
    if(nKFs<=2)
        nMinObs=2;
    int nRefMatches = mpReferenceKF->TrackedMapPoints(nMinObs);

    // Local Mapping accept keyframes?
    bool bLocalMappingIdle = mpLocalMapper->AcceptKeyFrames();

    // Check how many "close" points are being tracked and how many could be potentially created.
    int nNonTrackedClose = 0;
    int nTrackedClose= 0;
    if(mSensor!=System::MONOCULAR)
    {
        for(int i =0; i<mCurrentFrame->N; i++)
        {
            if(mCurrentFrame->mvDepth[i]>0 && mCurrentFrame->mvDepth[i]<mThDepth)
            {
                if(mCurrentFrame->mvpMapPoints[i] && !mCurrentFrame->mvbOutlier[i])
                    nTrackedClose++;
                else
                    nNonTrackedClose++;
            }
        }
    }

    bool bNeedToInsertClose = (nTrackedClose<100) && (nNonTrackedClose>70);

    // Thresholds
    float thRefRatio = 0.75f;
    if(nKFs<2)
        thRefRatio = 0.4f;

    if(mSensor==System::MONOCULAR)
        thRefRatio = 0.9f;

    // Condition 1a: More than "MaxFrames" have passed from last keyframe insertion
    const bool c1a = mCurrentFrame->mnId>=mnLastKeyFrameId+mMaxFrames;
    // Condition 1b: More than "MinFrames" have passed and Local Mapping is idle
    const bool c1b = (mCurrentFrame->mnId>=mnLastKeyFrameId+mMinFrames && bLocalMappingIdle);
    //Condition 1c: tracking is weak
    const bool c1c =  mSensor!=System::MONOCULAR && (mnMatchesInliers<nRefMatches*0.25 || bNeedToInsertClose) ;
    // Condition 2: Few tracked points compared to reference keyframe. Lots of visual odometry compared to map matches.
    const bool c2 = ((mnMatchesInliers<nRefMatches*thRefRatio|| bNeedToInsertClose) && mnMatchesInliers>15);

    if((c1a||c1b||c1c)&&c2)
    {
        // If the mapping accepts keyframes, insert keyframe.
        // Otherwise send a signal to interrupt BA
        if(bLocalMappingIdle)
        {
            return true;
        }
        else
        {
            mpLocalMapper->InterruptBA();
            if(mSensor!=System::MONOCULAR)
            {
                if(mpLocalMapper->KeyframesInQueue()<3)
                    return true;
                else
                    return false;
            }
            else
                return false;
        }
    }
    else
        return false;
}

void Tracking::CreateNewKeyFrame()
{
    if(!mpLocalMapper->SetNotStop(true))
        return;

    KeyFrame* pKF = new KeyFrame(*mCurrentFrame,mpMap,mpKeyFrameDB);

    mpReferenceKF = pKF;
    mCurrentFrame->mpReferenceKF = pKF;

    if(mSensor!=System::MONOCULAR)
    {
        mCurrentFrame->UpdatePoseMatrices();

        // We sort points by the measured depth by the stereo/RGBD sensor.
        // We create all those MapPoints whose depth < mThDepth.
        // If there are less than 100 close points we create the 100 closest.
        std::vector<std::pair<float,int> > vDepthIdx;
        vDepthIdx.reserve(mCurrentFrame->N);
        for(int i=0; i<mCurrentFrame->N; i++)
        {
            float z = mCurrentFrame->mvDepth[i];
            if(z>0)
            {
                vDepthIdx.push_back(std::make_pair(z,i));
            }
        }

        if(!vDepthIdx.empty())
        {
            sort(vDepthIdx.begin(),vDepthIdx.end());

            int nPoints = 0;
            for(size_t j=0; j<vDepthIdx.size();j++)
            {
                int i = vDepthIdx[j].second;

                bool bCreateNew = false;

                MapPoint* pMP = mCurrentFrame->mvpMapPoints[i];
                if(!pMP)
                    bCreateNew = true;
                else if(pMP->Observations()<1)
                {
                    bCreateNew = true;
                    mCurrentFrame->mvpMapPoints[i] = static_cast<MapPoint*>(NULL);
                }

                if(bCreateNew)
                {
                    cv::Mat x3D = mCurrentFrame->UnprojectStereo(i);
                    MapPoint* pNewMP = new MapPoint(x3D,pKF,mpMap);
                    pNewMP->AddObservation(pKF,i);
                    pKF->AddMapPoint(pNewMP,i);
                    pNewMP->ComputeDistinctiveDescriptors();
                    pNewMP->UpdateNormalAndDepth();
                    mpMap->AddMapPoint(pNewMP);

                    mCurrentFrame->mvpMapPoints[i]=pNewMP;
                    nPoints++;
                }
                else
                {
                    nPoints++;
                }

                if(vDepthIdx[j].first>mThDepth && nPoints>100)
                    break;
            }
        }
    }

    mpLocalMapper->InsertKeyFrame(pKF);

    mpLocalMapper->SetNotStop(false);

    mnLastKeyFrameId = mCurrentFrame->mnId;
    mpLastKeyFrame = pKF;
}

void Tracking::SearchLocalPoints()
{
    // Do not search map points already matched
    for(std::vector<MapPoint*>::iterator vit=mCurrentFrame->mvpMapPoints.begin(), vend=mCurrentFrame->mvpMapPoints.end(); vit!=vend; vit++)
    {
        MapPoint* pMP = *vit;
        if(pMP)
        {
            if(pMP->isBad())
            {
                *vit = static_cast<MapPoint*>(NULL);
            }
            else
            {
                pMP->IncreaseVisible();
                pMP->mnLastFrameSeen = mCurrentFrame->mnId;
                pMP->mbTrackInView = false;
            }
        }
    }

    int nToMatch=0;

    // Project points in frame and check its visibility
    for(std::vector<MapPoint*>::iterator vit=mvpLocalMapPoints.begin(), vend=mvpLocalMapPoints.end(); vit!=vend; vit++)
    {
        MapPoint* pMP = *vit;
        if(pMP->mnLastFrameSeen == mCurrentFrame->mnId)
            continue;
        if(pMP->isBad())
            continue;
        // Project (this fills MapPoint variables for matching)
        if(mCurrentFrame->isInFrustum(pMP,0.5))
        {
            pMP->IncreaseVisible();
            nToMatch++;
        }
    }

    if(nToMatch>0)
    {
        Matcher matcher(0.8);
        int th = 1;
        if(mSensor==System::RGBD)
            th=3;
        // If the camera has been relocalised recently, perform a coarser search
        if(mCurrentFrame->mnId<mnLastRelocFrameId+2)
            th=5;
        matcher.SearchByProjection(*mCurrentFrame,mvpLocalMapPoints,th);
    }
}

void Tracking::UpdateLocalMap()
{
    // This is for visualization
    mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

    // Update
    UpdateLocalKeyFrames();
    UpdateLocalPoints();
}

void Tracking::UpdateLocalPoints()
{
    mvpLocalMapPoints.clear();

    for(std::vector<KeyFrame*>::const_iterator itKF=mvpLocalKeyFrames.begin(), itEndKF=mvpLocalKeyFrames.end(); itKF!=itEndKF; itKF++)
    {
        KeyFrame* pKF = *itKF;
        const std::vector<MapPoint*> vpMPs = pKF->GetMapPointMatches();

        for(std::vector<MapPoint*>::const_iterator itMP=vpMPs.begin(), itEndMP=vpMPs.end(); itMP!=itEndMP; itMP++)
        {
            MapPoint* pMP = *itMP;
            if(!pMP)
                continue;
            if(pMP->mnTrackReferenceForFrame==mCurrentFrame->mnId)
                continue;
            if(!pMP->isBad())
            {
                mvpLocalMapPoints.push_back(pMP);
                pMP->mnTrackReferenceForFrame=mCurrentFrame->mnId;
            }
        }
    }
}


void Tracking::UpdateLocalKeyFrames()
{
    // Each map point vote for the keyframes in which it has been observed
    std::map<KeyFrame*,int> keyframeCounter;
    for(int i=0; i<mCurrentFrame->N; i++)
    {
        if(mCurrentFrame->mvpMapPoints[i])
        {
            MapPoint* pMP = mCurrentFrame->mvpMapPoints[i];
            if(!pMP->isBad())
            {
                const std::map<KeyFrame*,size_t> observations = pMP->GetObservations();
                for(std::map<KeyFrame*,size_t>::const_iterator it=observations.begin(), itend=observations.end(); it!=itend; it++)
                    keyframeCounter[it->first]++;
            }
            else
            {
                mCurrentFrame->mvpMapPoints[i]=NULL;
            }
        }
    }

    if(keyframeCounter.empty())
        return;

    int max=0;
    KeyFrame* pKFmax= static_cast<KeyFrame*>(NULL);

    mvpLocalKeyFrames.clear();
    mvpLocalKeyFrames.reserve(3*keyframeCounter.size());

    // All keyframes that observe a map point are included in the local map. Also check which keyframe shares most points
    for(std::map<KeyFrame*,int>::const_iterator it=keyframeCounter.begin(), itEnd=keyframeCounter.end(); it!=itEnd; it++)
    {
        KeyFrame* pKF = it->first;

        if(pKF->isBad())
            continue;

        if(it->second>max)
        {
            max=it->second;
            pKFmax=pKF;
        }

        mvpLocalKeyFrames.push_back(it->first);
        pKF->mnTrackReferenceForFrame = mCurrentFrame->mnId;
    }


    // Include also some not-already-included keyframes that are neighbors to already-included keyframes
    for(std::vector<KeyFrame*>::const_iterator itKF=mvpLocalKeyFrames.begin(), itEndKF=mvpLocalKeyFrames.end(); itKF!=itEndKF; itKF++)
    {
        // Limit the number of keyframes
        if(mvpLocalKeyFrames.size()>80)
            break;

        KeyFrame* pKF = *itKF;

        const std::vector<KeyFrame*> vNeighs = pKF->GetBestCovisibilityKeyFrames(10);

        for(std::vector<KeyFrame*>::const_iterator itNeighKF=vNeighs.begin(), itEndNeighKF=vNeighs.end(); itNeighKF!=itEndNeighKF; itNeighKF++)
        {
            KeyFrame* pNeighKF = *itNeighKF;
            if(!pNeighKF->isBad())
            {
                if(pNeighKF->mnTrackReferenceForFrame!=mCurrentFrame->mnId)
                {
                    mvpLocalKeyFrames.push_back(pNeighKF);
                    pNeighKF->mnTrackReferenceForFrame=mCurrentFrame->mnId;
                    break;
                }
            }
        }

        const std::set<KeyFrame*> spChilds = pKF->GetChilds();
        for(std::set<KeyFrame*>::const_iterator sit=spChilds.begin(), send=spChilds.end(); sit!=send; sit++)
        {
            KeyFrame* pChildKF = *sit;
            if(!pChildKF->isBad())
            {
                if(pChildKF->mnTrackReferenceForFrame!=mCurrentFrame->mnId)
                {
                    mvpLocalKeyFrames.push_back(pChildKF);
                    pChildKF->mnTrackReferenceForFrame=mCurrentFrame->mnId;
                    break;
                }
            }
        }

        KeyFrame* pParent = pKF->GetParent();
        if(pParent)
        {
            if(pParent->mnTrackReferenceForFrame!=mCurrentFrame->mnId)
            {
                mvpLocalKeyFrames.push_back(pParent);
                pParent->mnTrackReferenceForFrame=mCurrentFrame->mnId;
                break;
            }
        }

    }

    if(pKFmax)
    {
        mpReferenceKF = pKFmax;
        mCurrentFrame->mpReferenceKF = mpReferenceKF;
    }
}

bool Tracking::Relocalization()
{
    // Compute Bag of Words Vector
    mCurrentFrame->ComputeBoW();
	std::vector<KeyFrame*> vpCandidateKFs = mpKeyFrameDB->DetectRelocalizationByglb(&*mCurrentFrame);

    if(vpCandidateKFs.empty())
        return false;

    const int nKFs = vpCandidateKFs.size();

    // We perform first an ORB matching with each candidate
    // If enough matches are found we setup a PnP solver
    Matcher matcher(0.75, true);

    std::vector<PnPsolver*> vpPnPsolvers;
    vpPnPsolvers.resize(nKFs);

    std::vector<std::vector<MapPoint*> > vvpMapPointMatches;
    vvpMapPointMatches.resize(nKFs);

    std::vector<bool> vbDiscarded;
    vbDiscarded.resize(nKFs);

    int nCandidates=0;

    // For test
    std::vector<MapPoint*> vpMapPointMatches;

    std::vector<cv::Mat> querys;
    querys.resize(vpCandidateKFs.size());
    for(size_t i = 0; i < vpCandidateKFs.size(); i++){
        cv::Mat sss(vpCandidateKFs[i]->mDescriptors.rows, 256, CV_32F);
        sss = vpCandidateKFs[i]->mDescriptors;
        querys[i].push_back(sss);
    }

    cv::Mat query(querys.front());

    int count[querys.size()]={querys.front().rows};

    std::vector<cv::Mat>::iterator it;

    for(it=querys.begin()+1; it!=querys.end(); it++)
    {
        count[it-querys.begin()]=(*it).rows;
        cv::vconcat(query, (*it), query);
    }

    cv::BFMatcher match(cv::NORM_L2);

    std::vector< std::vector<cv::DMatch> > knnMatches;

    match.knnMatch(mCurrentFrame->mDescriptors, query, knnMatches, 2);

    std::vector<cv::DMatch> results;
    float thresh = 0.98;
    std::vector< std::vector<cv::DMatch> >::iterator ite;

    for(ite=knnMatches.begin(); ite!=knnMatches.end(); ite++)
    {
        if (((*ite).front().distance/(*ite).back().distance) < thresh)
        {
            size_t img_count=0;
            for (img_count=0; img_count<sizeof(count); img_count++)
            {
                (*ite).front().trainIdx -= count[img_count];
                if ((*ite).front().trainIdx < 0)
                {
                    (*ite).front().trainIdx += count[img_count];
                    break;
                }
            }
            results.push_back(cv::DMatch ((*ite).front().queryIdx, (*ite).front().trainIdx, img_count, (*ite).front().distance));
        }

    }

    vpMapPointMatches = std::vector<MapPoint*>(mCurrentFrame->N,static_cast<MapPoint*>(NULL));
    int aaa = 0;
    for(size_t i = 0; i < results.size(); ++i) {
        const int dist = Matcher::DescriptorDistance(vpCandidateKFs[results[i].imgIdx]->mDescriptors.row(results[i].trainIdx), mCurrentFrame->mDescriptors.row(results[i].queryIdx));
        if(dist<=50){
            const std::vector<MapPoint*> vpMapPointsKF = vpCandidateKFs[results[i].imgIdx]->GetMapPointMatches();
            vpMapPointMatches[results[i].queryIdx]=vpMapPointsKF[results[i].trainIdx];

            aaa++;
        }
    }
    PnPsolver* pSolver = new PnPsolver(*mCurrentFrame,vpMapPointMatches);
    pSolver->SetRansacParameters(0.99,4,800,4,0.5,10);
    vpPnPsolvers[0] = pSolver;
    std::vector<bool> vbInliers;
    int nInliers;
    bool bNoMore;
    cv::Mat Tccw = vpPnPsolvers[0]->iterate(5,bNoMore,vbInliers,nInliers);


    // No change following contest
    for(int i=0; i<nKFs; i++)
    {
        KeyFrame* pKF = vpCandidateKFs[i];
        if(pKF->isBad())
            vbDiscarded[i] = true;
        else
        {
            int nmatches = matcher.SearchByBoW(pKF,*mCurrentFrame,vvpMapPointMatches[i]);
            if(nmatches<15)
            {
                vbDiscarded[i] = true;
                continue;
            }
            else
            {
                PnPsolver* pSolver = new PnPsolver(*mCurrentFrame,vvpMapPointMatches[i]);
                pSolver->SetRansacParameters(0.99,10,300,4,0.5,5.991);
                vpPnPsolvers[i] = pSolver;
                nCandidates++;
            }
        }
    }

    // Alternatively perform some iterations of P4P RANSAC
    // Until we found a camera pose supported by enough inliers
    bool bMatch = false;
    Matcher matcher2(0.9, true);

    while(nCandidates>0 && !bMatch)
    {
        for(int i=0; i<nKFs; i++)
        {
            if(vbDiscarded[i])
                continue;

            // Perform 5 Ransac Iterations
            std::vector<bool> vbInliers;
            int nInliers;
            bool bNoMore;

            PnPsolver* pSolver = vpPnPsolvers[i];
            cv::Mat Tcw = pSolver->iterate(5,bNoMore,vbInliers,nInliers);

            // If Ransac reachs max. iterations discard keyframe
            if(bNoMore)
            {
                vbDiscarded[i]=true;
                nCandidates--;
            }

            // If a Camera Pose is computed, optimize
            if(!Tcw.empty())
            {
                Tcw.copyTo(mCurrentFrame->mTcw);

                std::set<MapPoint*> sFound;

                const int np = vbInliers.size();

                for(int j=0; j<np; j++)
                {
                    if(vbInliers[j])
                    {
                        mCurrentFrame->mvpMapPoints[j]=vvpMapPointMatches[i][j];
                        sFound.insert(vvpMapPointMatches[i][j]);
                    }
                    else
                        mCurrentFrame->mvpMapPoints[j]=NULL;
                }

                int nGood = Optimizer::PoseOptimization(&*mCurrentFrame);

                if(nGood<10)
                    continue;

                for(int io =0; io<mCurrentFrame->N; io++)
                    if(mCurrentFrame->mvbOutlier[io])
                        mCurrentFrame->mvpMapPoints[io]=static_cast<MapPoint*>(NULL);

                // If few inliers, search by projection in a coarse window and optimize again
                if(nGood<50)
                {
                    int nadditional =matcher2.SearchByProjection(*mCurrentFrame,vpCandidateKFs[i],sFound,10,100);

                    if(nadditional+nGood>=50)
                    {
                        nGood = Optimizer::PoseOptimization(&*mCurrentFrame);

                        // If many inliers but still not enough, search by projection again in a narrower window
                        // the camera has been already optimized with many points
                        if(nGood>30 && nGood<50)
                        {
                            sFound.clear();
                            for(int ip =0; ip<mCurrentFrame->N; ip++)
                                if(mCurrentFrame->mvpMapPoints[ip])
                                    sFound.insert(mCurrentFrame->mvpMapPoints[ip]);
                            nadditional =matcher2.SearchByProjection(*mCurrentFrame,vpCandidateKFs[i],sFound,3,64);

                            // Final optimization
                            if(nGood+nadditional>=50)
                            {
                                nGood = Optimizer::PoseOptimization(&*mCurrentFrame);

                                for(int io =0; io<mCurrentFrame->N; io++)
                                    if(mCurrentFrame->mvbOutlier[io])
                                        mCurrentFrame->mvpMapPoints[io]=NULL;
                            }
                        }
                    }
                }


                // If the pose is supported by enough inliers stop ransacs and continue
                if(nGood>=50)
                {
                    bMatch = true;
                    break;
                }
            }
        }
    }

    if(!bMatch)
    {
        return false;
    }
    else
    {
        mnLastRelocFrameId = mCurrentFrame->mnId;
        return true;
    }

}

void Tracking::Reset()
{

    std::cout << "System Reseting" << std::endl;
    if(mpViewer)
    {
        mpViewer->RequestStop();	//单纯将viewer里面置位停止的请求
        while(!mpViewer->isStopped())
            usleep(3000);	//然后等他结束	后续工作估计已经融入到线程当中了
    }

    // Reset Local Mapping
    std::cout << "Reseting Local Mapper...";
    mpLocalMapper->RequestReset();
    std::cout << " done" << std::endl;

    // Reset Loop Closing
    std::cout << "Reseting Loop Closing...";
    mpLoopClosing->RequestReset();		//估计和上面差不多，不用看了
    std::cout << " done" << std::endl;

    // Clear BoW Database
    std::cout << "Reseting Database...";
    mpKeyFrameDB->clear();		//清空，就是清除所有的有用参数
    std::cout << " done" << std::endl;

    // Clear Map (this erase MapPoints and KeyFrames)
    mpMap->clear();

    KeyFrame::nNextId = 0;
    Frame::nNextId = 0;
    mState = NO_IMAGES_YET;		//恢复初始状态

    if(mpInitializer)		//初始化用的工具
    {
        delete mpInitializer;
        mpInitializer = static_cast<Initializer*>(NULL);
    }

    mlRelativeFramePoses.clear();		//全是list
    mlpReferences.clear();
    mlFrameTimes.clear();
    mlbLost.clear();

    if(mpViewer)
        mpViewer->Release();		//stop置为false
}

void Tracking::ChangeCalibration(const std::string &strSettingPath)
{
    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];

    cv::Mat K = cv::Mat::eye(3,3,CV_32F);
    K.at<float>(0,0) = fx;
    K.at<float>(1,1) = fy;
    K.at<float>(0,2) = cx;
    K.at<float>(1,2) = cy;
    K.copyTo(mK);

    cv::Mat DistCoef(4,1,CV_32F);
    DistCoef.at<float>(0) = fSettings["Camera.k1"];
    DistCoef.at<float>(1) = fSettings["Camera.k2"];
    DistCoef.at<float>(2) = fSettings["Camera.p1"];
    DistCoef.at<float>(3) = fSettings["Camera.p2"];
    const float k3 = fSettings["Camera.k3"];
    if(k3!=0)
    {
        DistCoef.resize(5);
        DistCoef.at<float>(4) = k3;
    }
    DistCoef.copyTo(mDistCoef);

    mbf = fSettings["Camera.bf"];

    Frame::mbInitialComputations = true;
}

void Tracking::InformOnlyTracking(const bool &flag)
{
    mbOnlyTracking = flag;
}

} //namespace ORB_SLAM
