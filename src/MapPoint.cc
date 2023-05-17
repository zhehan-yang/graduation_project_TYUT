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
#include "MapPoint.h"
#include "Matcher.h"

#include<mutex>

namespace DXSLAM
{
#pragma region Mappoint类成员函数
	/*
	long unsigned int mnId;		这个地图点的ID，每创建一个就加一
	static long unsigned int nNextId;
	long int mnFirstKFid;	第一个关键帧的ID号 构造函数导入
	long int mnFirstFrame;	这个关键帧的帧ID 构造函数导入
	int nObs;
	float mTrackProjX;
	float mTrackProjY;
	float mTrackProjXR;
	bool mbTrackInView;
	float mTrackViewCos;
	long unsigned int mnTrackReferenceForFrame;
	long unsigned int mnLastFrameSeen;
	long unsigned int mnBALocalForKF;
	long unsigned int mnFuseCandidateForKF;
	long unsigned int mnLoopPointForKF;
	long unsigned int mnCorrectedByKF;
	long unsigned int mnCorrectedReference;
	cv::Mat mPosGBA;
	long unsigned int mnBAGlobalForKF;
	static std::mutex mGlobalMutex;
protected:
	 cv::Mat mWorldPos;			这个地图点的世界坐标系的坐标
	 std::map<KeyFrame*,size_t> mObservations;  字典类型，这里存放的是他的观测点，左边是关键帧，右边是关键帧的第几个关键点
	 cv::Mat mNormalVector;		表示平均观测方向
	 cv::Mat mDescriptor;		最佳的描述子，通过ComputeDistinctiveDescriptors函数不断更新
	 KeyFrame* mpRefKF;		刚开始是创建他的关键帧，构造函数，后面会抑制更新参考的关键帧
	 int mnVisible;
	 int mnFound;
	 bool mbBad;			是否为坏的地图点，初始为false
	 MapPoint* mpReplaced;
	 float mfMinDistance;		这里认为是同一个就行了，表示代表关键帧的距离
	 float mfMaxDistance;
	 Map* mpMap;
	 std::mutex mMutexPos;
	 std::mutex mMutexFeatures;
	*/
#pragma endregion

long unsigned int MapPoint::nNextId=0;
std::mutex MapPoint::mGlobalMutex;

//位姿，关键帧，地图
MapPoint::MapPoint(const cv::Mat &Pos, KeyFrame *pRefKF, Map* pMap):
    mnFirstKFid(pRefKF->mnId), mnFirstFrame(pRefKF->mnFrameId), nObs(0), mnTrackReferenceForFrame(0),
    mnLastFrameSeen(0), mnBALocalForKF(0), mnFuseCandidateForKF(0), mnLoopPointForKF(0), mnCorrectedByKF(0),
    mnCorrectedReference(0), mnBAGlobalForKF(0), mpRefKF(pRefKF), mnVisible(1), mnFound(1), mbBad(false),
    mpReplaced(static_cast<MapPoint*>(NULL)), mfMinDistance(0), mfMaxDistance(0), mpMap(pMap)
{
    Pos.copyTo(mWorldPos);
    mNormalVector = cv::Mat::zeros(3,1,CV_32F);

    // MapPoints can be created from Tracking and Local Mapping. This std::mutex avoid conflicts with id.
    std::unique_lock<std::mutex> lock(mpMap->mMutexPointCreation);
    mnId=nNextId++;
}

MapPoint::MapPoint(const cv::Mat &Pos, Map* pMap, Frame* pFrame, const int &idxF):
    mnFirstKFid(-1), mnFirstFrame(pFrame->mnId), nObs(0), mnTrackReferenceForFrame(0), mnLastFrameSeen(0),
    mnBALocalForKF(0), mnFuseCandidateForKF(0),mnLoopPointForKF(0), mnCorrectedByKF(0),
    mnCorrectedReference(0), mnBAGlobalForKF(0), mpRefKF(static_cast<KeyFrame*>(NULL)), mnVisible(1),
    mnFound(1), mbBad(false), mpReplaced(NULL), mpMap(pMap)
{
    Pos.copyTo(mWorldPos);
    cv::Mat Ow = pFrame->GetCameraCenter();
    mNormalVector = mWorldPos - Ow;
    mNormalVector = mNormalVector/cv::norm(mNormalVector);

    cv::Mat PC = Pos - Ow;
    const float dist = cv::norm(PC);
    const int level = pFrame->mvKeysUn[idxF].octave;
    const float levelScaleFactor =  1;
    const int nLevels = 1;

    mfMaxDistance = dist*levelScaleFactor;
    mfMinDistance = mfMaxDistance/1;

    pFrame->mDescriptors.row(idxF).copyTo(mDescriptor);

    // MapPoints can be created from Tracking and Local Mapping. This std::mutex avoid conflicts with id.
    std::unique_lock<std::mutex> lock(mpMap->mMutexPointCreation);
    mnId=nNextId++;
}

void MapPoint::SetWorldPos(const cv::Mat &Pos)
{
    std::unique_lock<std::mutex> lock2(mGlobalMutex);
    std::unique_lock<std::mutex> lock(mMutexPos);
    Pos.copyTo(mWorldPos);
}

cv::Mat MapPoint::GetWorldPos()
{
    std::unique_lock<std::mutex> lock(mMutexPos);
    return mWorldPos.clone();
}

cv::Mat MapPoint::GetNormal()
{
    std::unique_lock<std::mutex> lock(mMutexPos);
    return mNormalVector.clone();
}

KeyFrame* MapPoint::GetReferenceKeyFrame()
{
    std::unique_lock<std::mutex> lock(mMutexFeatures);
    return mpRefKF;
}

//给地图点添加能看到这个地图点的关键帧以及这个关键帧里关键点编号
void MapPoint::AddObservation(KeyFrame* pKF, size_t idx)
{
    std::unique_lock<std::mutex> lock(mMutexFeatures);	//特征锁锁上
    if(mObservations.count(pKF))	//如果已经有
        return;
    mObservations[pKF]=idx;

    if(pKF->mvuRight[idx]>=0)		//如果右目也可以看到他，那么就加上2，否则加1
        nObs+=2;
    else
        nObs++;
}

void MapPoint::EraseObservation(KeyFrame* pKF)
{
    bool bBad=false;
    {
        std::unique_lock<std::mutex> lock(mMutexFeatures);
        if(mObservations.count(pKF))
        {
            int idx = mObservations[pKF];
            if(pKF->mvuRight[idx]>=0)
                nObs-=2;
            else
                nObs--;

            mObservations.erase(pKF);

            if(mpRefKF==pKF)
                mpRefKF=mObservations.begin()->first;

            // If only 2 observations or less, discard point
            if(nObs<=2)
                bBad=true;
        }
    }

    if(bBad)
        SetBadFlag();
}

std::map<KeyFrame*, size_t> MapPoint::GetObservations()
{
    std::unique_lock<std::mutex> lock(mMutexFeatures);
    return mObservations;
}

int MapPoint::Observations()
{
    std::unique_lock<std::mutex> lock(mMutexFeatures);
    return nObs;
}

void MapPoint::SetBadFlag()
{
    std::map<KeyFrame*,size_t> obs;
    {
        std::unique_lock<std::mutex> lock1(mMutexFeatures);
        std::unique_lock<std::mutex> lock2(mMutexPos);
        mbBad=true;
        obs = mObservations;
        mObservations.clear();
    }
    for(std::map<KeyFrame*,size_t>::iterator mit=obs.begin(), mend=obs.end(); mit!=mend; mit++)
    {
        KeyFrame* pKF = mit->first;
        pKF->EraseMapPointMatch(mit->second);
    }

    mpMap->EraseMapPoint(this);
}

//得到替代地图点
MapPoint* MapPoint::GetReplaced()
{
    std::unique_lock<std::mutex> lock1(mMutexFeatures);
    std::unique_lock<std::mutex> lock2(mMutexPos);
    return mpReplaced;
}

void MapPoint::Replace(MapPoint* pMP)
{
    if(pMP->mnId==this->mnId)
        return;

    int nvisible, nfound;
    std::map<KeyFrame*,size_t> obs;
    {
        std::unique_lock<std::mutex> lock1(mMutexFeatures);
        std::unique_lock<std::mutex> lock2(mMutexPos);
        obs=mObservations;
        mObservations.clear();
        mbBad=true;
        nvisible = mnVisible;
        nfound = mnFound;
        mpReplaced = pMP;
    }

    for(std::map<KeyFrame*,size_t>::iterator mit=obs.begin(), mend=obs.end(); mit!=mend; mit++)
    {
        // Replace measurement in keyframe
        KeyFrame* pKF = mit->first;

        if(!pMP->IsInKeyFrame(pKF))
        {
            pKF->ReplaceMapPointMatch(mit->second, pMP);
            pMP->AddObservation(pKF,mit->second);
        }
        else
        {
            pKF->EraseMapPointMatch(mit->second);
        }
    }
    pMP->IncreaseFound(nfound);
    pMP->IncreaseVisible(nvisible);
    pMP->ComputeDistinctiveDescriptors();

    mpMap->EraseMapPoint(this);
}

bool MapPoint::isBad()
{
    std::unique_lock<std::mutex> lock(mMutexFeatures);
    std::unique_lock<std::mutex> lock2(mMutexPos);
    return mbBad;
}

void MapPoint::IncreaseVisible(int n)
{
    std::unique_lock<std::mutex> lock(mMutexFeatures);
    mnVisible+=n;
}

void MapPoint::IncreaseFound(int n)
{
    std::unique_lock<std::mutex> lock(mMutexFeatures);
    mnFound+=n;
}

float MapPoint::GetFoundRatio()
{
    std::unique_lock<std::mutex> lock(mMutexFeatures);
    return static_cast<float>(mnFound)/mnVisible;
}

//这个函数根据能看到地图点的关键帧中的描述子，更新最佳的描述子
void MapPoint::ComputeDistinctiveDescriptors()
{
    // Retrieve all observed descriptors
    std::vector<cv::Mat> vDescriptors;

    std::map<KeyFrame*,size_t> observations;

    {
        std::unique_lock<std::mutex> lock1(mMutexFeatures);	//特征锁
        if(mbBad) //是坏点，不用处理，初始化为false
            return;
        observations=mObservations;	//拿到函数里来
    }

    if(observations.empty())	//如果没有，退出
        return;

    vDescriptors.reserve(observations.size());	//保留能看到他的关键帧的数量

    for(std::map<KeyFrame*,size_t>::iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
    {
        KeyFrame* pKF = mit->first;	//获得他的元素值，即关键帧的值

        if(!pKF->isBad())	//不是坏点
            vDescriptors.push_back(pKF->mDescriptors.row(mit->second));	//压入这个关键帧的描述子
    }

    if(vDescriptors.empty()) //如果全都是坏关键帧
        return;

    // Compute distances between them
    const size_t N = vDescriptors.size();	//相当于好关键帧的数量

    float Distances[N][N];
    for(size_t i=0;i<N;i++)
    {
        Distances[i][i]=0;
        for(size_t j=i+1;j<N;j++)
        {
			//这个函数用来计算两个矩阵之间的曼哈顿距离 最大支持256维，描述子貌似就是256维
            int distij = Matcher::DescriptorDistance(vDescriptors[i], vDescriptors[j]);
            Distances[i][j]=distij;
            Distances[j][i]=distij;
        }
    }

    // Take the descriptor with least median distance to the rest
    int BestMedian = INT_MAX;		//最大有符号整数
    int BestIdx = 0;
    for(size_t i=0;i<N;i++)
    {
        std::vector<int> vDists(Distances[i],Distances[i]+N);	//注意这里是描述子，
        sort(vDists.begin(),vDists.end());		//排序
        int median = vDists[0.5*(N-1)];	//取中位数

        if(median<BestMedian)		//取中位数最小的那个，认为是最佳的描述子
        {
            BestMedian = median;
            BestIdx = i;
        }
    }

    {
        std::unique_lock<std::mutex> lock(mMutexFeatures);
        mDescriptor = vDescriptors[BestIdx].clone();	//更新最佳描述子
    }
}

cv::Mat MapPoint::GetDescriptor()
{
    std::unique_lock<std::mutex> lock(mMutexFeatures);
    return mDescriptor.clone();
}

int MapPoint::GetIndexInKeyFrame(KeyFrame *pKF)
{
    std::unique_lock<std::mutex> lock(mMutexFeatures);
    if(mObservations.count(pKF))
        return mObservations[pKF];
    else
        return -1;
}

bool MapPoint::IsInKeyFrame(KeyFrame *pKF)
{
    std::unique_lock<std::mutex> lock(mMutexFeatures);
    return (mObservations.count(pKF));
}

//更新平均观测方向和距离
void MapPoint::UpdateNormalAndDepth()
{
    std::map<KeyFrame*,size_t> observations;
    KeyFrame* pRefKF;
    cv::Mat Pos;
    {
        std::unique_lock<std::mutex> lock1(mMutexFeatures);
        std::unique_lock<std::mutex> lock2(mMutexPos);
        if(mbBad)		//如果是坏的地图点就不用做了
            return;
        observations=mObservations;		//拿进能看到地图点的关键帧
        pRefKF=mpRefKF;					//创建他的关键帧
        Pos = mWorldPos.clone();		//世界坐标系
    }

    if(observations.empty())			//没有能观察到他的地图点
        return;

    cv::Mat normal = cv::Mat::zeros(3,1,CV_32F);
    int n=0;
	//遍历能观察到他的关键帧
    for(std::map<KeyFrame*,size_t>::iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
    {
        KeyFrame* pKF = mit->first;	 
        cv::Mat Owi = pKF->GetCameraCenter();	//即返回ow
        cv::Mat normali = mWorldPos - Owi;		//向量？
        normal = normal + normali/cv::norm(normali);	//加上单位向量 累加
        n++;		//关键帧数量计数
    }

    cv::Mat PC = Pos - pRefKF->GetCameraCenter();
    const float dist = cv::norm(PC);	//范数，相当于欧氏距离
    const int level = pRefKF->mvKeysUn[observations[pRefKF]].octave;	//金字塔哪一层提取的
    const float levelScaleFactor =  1;
//    const int nLevels = pRefKF->mnScaleLevels;

    {
        std::unique_lock<std::mutex> lock3(mMutexPos);
        mfMaxDistance = dist*levelScaleFactor;
        mfMinDistance = mfMaxDistance/1;
        mNormalVector = normal/n;	//平均一下的向量
    }
}

float MapPoint::GetMinDistanceInvariance()
{
    std::unique_lock<std::mutex> lock(mMutexPos);
    return 0.8f*mfMinDistance;
}

float MapPoint::GetMaxDistanceInvariance()
{
    std::unique_lock<std::mutex> lock(mMutexPos);
    return 1.2f*mfMaxDistance;
}


} //namespace ORB_SLAM
