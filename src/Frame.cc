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
#include "Frame.h"
#include <thread>

namespace DXSLAM
{
	//？？？为什么要重写一遍？
#pragma region MyRegion
	/*
	Vocabulary* mpORBvocabulary;	字典指针，通过构造函数传入
	double mTimeStamp;
	上面传下来的相机参数
	cv::Mat mK;
	static float fx;
	static float fy;
	static float cx;
	static float cy;
	static float invfx;
	static float invfy;
	cv::Mat mDistCoef;
	float mbf;		mb*fx
	float mb;		基线长度
	float mThDepth;	深度阈值
	
	int N;				//关键点个数
	std::vector<cv::KeyPoint> mvKeys, mvKeysRight;	关键点
	std::vector<cv::KeyPoint> mvKeysUn;				关键点经过纠正畸变
	std::vector<float> mvuRight;			右目中的相机x坐标
	std::vector<float> mvDepth;				深度坐标
	fbow::fBow mBowVec;
	fbow::fBow2 mFeatVec;
	cv::Mat mDescriptors, mDescriptorsRight;			局部描述子
	std::vector<MapPoint*> mvpMapPoints;				地图点，构造的时候为空，在地图点配置完成后加入到地图中
	std::vector<bool> mvbOutlier;
	static float mfGridElementWidthInv;					每一个格子宽度的倒数
	static float mfGridElementHeightInv;				每一个格子高度的倒数
	std::vector<std::size_t> mGrid[FRAME_GRID_COLS][FRAME_GRID_ROWS];
	cv::Mat mTcw;
	static long unsigned int nNextId;
	long unsigned int mnId;
	KeyFrame* mpReferenceKF;				参考关键帧

	四个边界值：表示畸变完了之后实际需要处理的像素点范围
	static float mnMinX;
	static float mnMaxX;
	static float mnMinY;
	static float mnMaxY;

	static bool mbInitialComputations;		//静态变量，刚开始是真，作用完第一次就为假，用于标记是否未处理第一帧变量
	cv::Mat globaldescriptors;				全局描述子
	cv::Mat glbDistance;
	下面是私有的变量：
	cv::Mat mRcw;
	cv::Mat mtcw;
	cv::Mat mRwc;
	cv::Mat mOw; //==mtwc
	*/
#pragma endregion

long unsigned int Frame::nNextId=0;
bool Frame::mbInitialComputations=true;
float Frame::cx, Frame::cy, Frame::fx, Frame::fy, Frame::invfx, Frame::invfy;
float Frame::mnMinX, Frame::mnMinY, Frame::mnMaxX, Frame::mnMaxY;
float Frame::mfGridElementWidthInv, Frame::mfGridElementHeightInv;

Frame::Frame()
{}

//Copy Constructor 拷贝构造函数
Frame::Frame(const Frame &frame)
    :mpORBvocabulary(frame.mpORBvocabulary),
     mTimeStamp(frame.mTimeStamp), mK(frame.mK.clone()), mDistCoef(frame.mDistCoef.clone()),
     mbf(frame.mbf), mb(frame.mb), mThDepth(frame.mThDepth), N(frame.N), mvKeys(frame.mvKeys),
     mvKeysRight(frame.mvKeysRight), mvKeysUn(frame.mvKeysUn),  mvuRight(frame.mvuRight),
     mvDepth(frame.mvDepth), mBowVec(frame.mBowVec),
     mDescriptors(frame.mDescriptors.clone()), mDescriptorsRight(frame.mDescriptorsRight.clone()),
     mvpMapPoints(frame.mvpMapPoints), mvbOutlier(frame.mvbOutlier), mnId(frame.mnId),
     mpReferenceKF(frame.mpReferenceKF)
{
	//确实是源文件注释掉的
    // mBowVec = frame.mBowVec;
    for(int i=0;i<FRAME_GRID_COLS;i++)
        for(int j=0; j<FRAME_GRID_ROWS; j++)
            mGrid[i][j]=frame.mGrid[i][j];

    if(!frame.mTcw.empty())
        SetPose(frame.mTcw);
}

Frame::Frame(const cv::Mat &imGray,		//RGB图
        const cv::Mat &imDepth,			//深度图
        const double &timeStamp,		//时间戳
        Vocabulary* voc,				//字典指针
        cv::Mat &K,						//变换矩阵
        cv::Mat &distCoef,				//畸变参数
        const float &bf,				//基线*fx
        const float &thDepth,			//深度阈值
        const std::vector<cv::KeyPoint> &keypoints,		//这一帧里关键点
        const cv::Mat &local_desc,		//局部描述子
        const cv::Mat &global_desc		//全局描述子
        )
    :mpORBvocabulary(voc),
     mTimeStamp(timeStamp), mK(K.clone()),mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth),
     mvKeys(keypoints), mDescriptors(local_desc), globaldescriptors(global_desc)
{
    // Frame ID
    mnId=nNextId++;	//静态变量，创建一个就加一，用于编号

    N = mvKeys.size();		//关键点个数

    if(mvKeys.empty())		//没关键点？
        return;

    UndistortKeyPoints();		//抗扭曲处理

    ComputeStereoFromRGBD(imDepth);		//构造右目的图像
    mvpMapPoints = std::vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));	//地图点为空
    mvbOutlier = std::vector<bool>(N,false);		//标记用的？

    // This is done only for the first Frame (or after a change in the calibration)
    if(mbInitialComputations)
    {
        ComputeImageBounds(imGray);		//计算边界值

        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/static_cast<float>(mnMaxX-mnMinX);
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/static_cast<float>(mnMaxY-mnMinY);

		//导入相机参数
        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

        mbInitialComputations=false;
    }

    mb = mbf/fx;	//基线长度

    AssignFeaturesToGrid();
}

//将提取的ORB特征点分配到图像网格中方便特征点匹配的时候用到
void Frame::AssignFeaturesToGrid()
{
	//#define FRAME_GRID_ROWS 48  他是这么定义的
	//#define FRAME_GRID_COLS 64
    int nReserve = 0.5f*N/(FRAME_GRID_COLS*FRAME_GRID_ROWS);	//每一个网格中平均特征点的数目
    for(unsigned int i=0; i<FRAME_GRID_COLS;i++)
        for (unsigned int j=0; j<FRAME_GRID_ROWS;j++)
            mGrid[i][j].reserve(nReserve);	//预留空间		每一个点预留空间

    for(int i=0;i<N;i++)
    {
        const cv::KeyPoint &kp = mvKeysUn[i];	//解扭曲的点

        int nGridPosX, nGridPosY;
        if(PosInGrid(kp,nGridPosX,nGridPosY))	//如果在网格的范围内，同时计算网格编号
            mGrid[nGridPosX][nGridPosY].push_back(i);	//放入网格中
    }
}

//置入坐标，同时更新矩阵分解开来的值
void Frame::SetPose(const cv::Mat &Tcw)
{
    mTcw = Tcw.clone();	//相当于赋值
    UpdatePoseMatrices();	//更新里面的数据
}

//计算T的四个内部小矩阵
void Frame::UpdatePoseMatrices()
{
    mRcw = mTcw.rowRange(0,3).colRange(0,3);	//左上角的旋转矩阵
    mRwc = mRcw.t();		//R转置
    mtcw = mTcw.rowRange(0,3).col(3);		//右边的位移矩阵
    mOw = -mRcw.t()*mtcw;		//转置后的位移矩阵
}

bool Frame::isInFrustum(MapPoint *pMP, float viewingCosLimit)
{
    pMP->mbTrackInView = false;

    // 3D in absolute coordinates
    cv::Mat P = pMP->GetWorldPos();

    // 3D in camera coordinates
    const cv::Mat Pc = mRcw*P+mtcw;
    const float &PcX = Pc.at<float>(0);
    const float &PcY= Pc.at<float>(1);
    const float &PcZ = Pc.at<float>(2);

    // Check positive depth
    if(PcZ<0.0f)
        return false;

    // Project in image and check it is not outside
    const float invz = 1.0f/PcZ;
    const float u=fx*PcX*invz+cx;
    const float v=fy*PcY*invz+cy;

    if(u<mnMinX || u>mnMaxX)
        return false;
    if(v<mnMinY || v>mnMaxY)
        return false;

    // Check distance is in the scale invariance region of the MapPoint
    const float maxDistance = pMP->GetMaxDistanceInvariance();
    const float minDistance = pMP->GetMinDistanceInvariance();
    const cv::Mat PO = P-mOw;
    const float dist = cv::norm(PO);

    if(dist<minDistance || dist>maxDistance)
        return false;

   // Check viewing angle
    cv::Mat Pn = pMP->GetNormal();

    const float viewCos = PO.dot(Pn)/dist;

    if(viewCos<viewingCosLimit)
        return false;


    // Data used by the tracking
    pMP->mbTrackInView = true;
    pMP->mTrackProjX = u;
    pMP->mTrackProjXR = u - mbf*invz;
    pMP->mTrackProjY = v;
    pMP->mTrackViewCos = viewCos;

    return true;
}

std::vector<size_t> Frame::GetFeaturesInArea(const float &x, const float  &y, const float  &r, const int minLevel, const int maxLevel) const
{
    std::vector<size_t> vIndices;
    vIndices.reserve(N);

    const int nMinCellX = std::max(0,(int)floor((x-mnMinX-r)*mfGridElementWidthInv));
    if(nMinCellX>=FRAME_GRID_COLS)
        return vIndices;

    const int nMaxCellX = std::min((int)FRAME_GRID_COLS-1,(int)ceil((x-mnMinX+r)*mfGridElementWidthInv));
    if(nMaxCellX<0)
        return vIndices;

    const int nMinCellY = std::max(0,(int)floor((y-mnMinY-r)*mfGridElementHeightInv));
    if(nMinCellY>=FRAME_GRID_ROWS)
        return vIndices;

    const int nMaxCellY = std::min((int)FRAME_GRID_ROWS-1,(int)ceil((y-mnMinY+r)*mfGridElementHeightInv));
    if(nMaxCellY<0)
        return vIndices;

    const bool bCheckLevels = (minLevel>0) || (maxLevel>=0);

    for(int ix = nMinCellX; ix<=nMaxCellX; ix++)
    {
        for(int iy = nMinCellY; iy<=nMaxCellY; iy++)
        {
            const std::vector<size_t> vCell = mGrid[ix][iy];
            if(vCell.empty())
                continue;

            for(size_t j=0, jend=vCell.size(); j<jend; j++)
            {
                const cv::KeyPoint &kpUn = mvKeysUn[vCell[j]];
                if(bCheckLevels)
                {
                    if(kpUn.octave<minLevel)
                        continue;
                    if(maxLevel>=0)
                        if(kpUn.octave>maxLevel)
                            continue;
                }

                const float distx = kpUn.pt.x-x;
                const float disty = kpUn.pt.y-y;

                if(fabs(distx)<r && fabs(disty)<r)
                    vIndices.push_back(vCell[j]);
            }
        }
    }

    return vIndices;
}

//用来判断点是否在栅格内，然后返回具体在哪个格子
bool Frame::PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY)
{
    posX = round((kp.pt.x-mnMinX)*mfGridElementWidthInv);		//横坐标除以格子宽，算出来是第几个格子
    posY = round((kp.pt.y-mnMinY)*mfGridElementHeightInv);

    //Keypoint's coordinates are undistorted, which could cause to go out of the image
    if(posX<0 || posX>=FRAME_GRID_COLS || posY<0 || posY>=FRAME_GRID_ROWS)		//如果说在格子外头
        return false;

    return true;
}


void Frame::ComputeBoW()
{
    if(mBowVec.empty()|| mFeatVec.empty())
    {
        mpORBvocabulary->transform(mDescriptors,4,mBowVec,mFeatVec);
    }
}

void Frame::UndistortKeyPoints()
{
    if(mDistCoef.at<float>(0)==0.0)
    {
        mvKeysUn=mvKeys;		//不需要纠正畸变，直接输出
        return;
    }

    // Fill matrix with points
    cv::Mat mat(N,2,CV_32F);	//关键点个数*2矩阵，32位深度
    for(int i=0; i<N; i++)
    {
        mat.at<float>(i,0)=mvKeys[i].pt.x;
        mat.at<float>(i,1)=mvKeys[i].pt.y;		//存起来
    }

    // Undistort points
    mat=mat.reshape(2);		//加一个通道,估计用来放完成的参数
    cv::undistortPoints(mat,mat,mK,mDistCoef,cv::Mat(),mK);
	//参数依次是：输入坐标矩阵/输出坐标矩阵/相机内参系数/畸变矩阵/相机坐标系校正矩阵/新的相机矩阵
    //我们的校正不用相机偏移，所以看着相机坐标系校正是不需要的，但是这里要就给他
	mat=mat.reshape(1);	//变回1维

    // Fill undistorted keypoint std::vector
    mvKeysUn.resize(N);	//vector，开到N维
    for(int i=0; i<N; i++)
    {
        cv::KeyPoint kp = mvKeys[i];	//一个一个开，设置，然后存储
        kp.pt.x=mat.at<float>(i,0);
        kp.pt.y=mat.at<float>(i,1);
        mvKeysUn[i]=kp;
    }
}

void Frame::ComputeImageBounds(const cv::Mat &imLeft)
{
	//这个函数用于计算边界值的大小
    if(mDistCoef.at<float>(0)!=0.0)	//有畸变
    {
		//要做的就是吧四个角上的参数做一次畸变处理即可
        cv::Mat mat(4,2,CV_32F);
        mat.at<float>(0,0)=0.0; mat.at<float>(0,1)=0.0;
        mat.at<float>(1,0)=imLeft.cols; mat.at<float>(1,1)=0.0;
        mat.at<float>(2,0)=0.0; mat.at<float>(2,1)=imLeft.rows;
        mat.at<float>(3,0)=imLeft.cols; mat.at<float>(3,1)=imLeft.rows;

        // Undistort corners
        mat=mat.reshape(2);
        cv::undistortPoints(mat,mat,mK,mDistCoef,cv::Mat(),mK);	//参考下面的UndistortKeyPoints函数
        mat=mat.reshape(1);

        mnMinX = std::min(mat.at<float>(0,0),mat.at<float>(2,0));
        mnMaxX = std::max(mat.at<float>(1,0),mat.at<float>(3,0));
        mnMinY = std::min(mat.at<float>(0,1),mat.at<float>(1,1));
        mnMaxY = std::max(mat.at<float>(2,1),mat.at<float>(3,1));

    }
    else
    {
        mnMinX = 0.0f;
        mnMaxX = imLeft.cols;
        mnMinY = 0.0f;
        mnMaxY = imLeft.rows;
    }
}


void Frame::ComputeStereoFromRGBD(const cv::Mat &imDepth)
{
	//根据RGBD图像，构造虚拟的右目图像
    mvuRight = std::vector<float>(N,-1);	//N个值是-1的函数
    mvDepth = std::vector<float>(N,-1);		//

    for(int i=0; i<N; i++)
    {
        const cv::KeyPoint &kp = mvKeys[i];		//没处理过得
        const cv::KeyPoint &kpU = mvKeysUn[i];	//抗扭曲处理过得

        const float &v = kp.pt.y;
        const float &u = kp.pt.x;

        const float d = imDepth.at<float>(v,u);	//找到这个关键点的深度，应该在原图上找，因为深度是基于原图的

        if(d>0)
        {
            mvDepth[i] = d;		//深度就是所谓的深度
            mvuRight[i] = kpU.pt.x-mbf/d;	//构造右目，画一个图就懂了，注意这里用的是mbf，就不用坐标转换了
        }
    }
}

//这个函数返回第i个关键帧在地面坐标系下的坐标值
cv::Mat Frame::UnprojectStereo(const int &i)
{
    const float z = mvDepth[i];
    if(z>0)
    {
        const float u = mvKeysUn[i].pt.x;
        const float v = mvKeysUn[i].pt.y;
        const float x = (u-cx)*z*invfx;		//在相机坐标系下实际点的坐标
        const float y = (v-cy)*z*invfy;
        cv::Mat x3Dc = (cv::Mat_<float>(3,1) << x, y, z);	//在相机坐标系下他的坐标值
        return mRwc*x3Dc+mOw;			//计算在地面坐标系下的坐标 ，注意这里用的是转回去
    }
    else
        return cv::Mat();
}

} //namespace ORB_SLAM
