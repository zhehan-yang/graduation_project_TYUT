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
#include "KeyFrame.h"
#include "Converter.h"
#include "Matcher.h"
#include<mutex>

namespace DXSLAM
{

	//参考资料：https://zhuanlan.zhihu.com/p/84293190 涉及Keyframe优化理论与该类下函数分类
#pragma region MyRegion
	/*
	static long unsigned int nNextId;
	long unsigned int mnId;
	const long unsigned int mnFrameId;	//用于记录关键帧的帧ID号
	const double mTimeStamp;			//时间戳
	const int mnGridCols;				//帧的格子几行几列
	const int mnGridRows;
	const float mfGridElementWidthInv;	//每一格长宽的倒数
	const float mfGridElementHeightInv;
	long unsigned int mnTrackReferenceForFrame;
	long unsigned int mnFuseTargetForKF;
	long unsigned int mnBALocalForKF;
	long unsigned int mnBAFixedForKF;
	long unsigned int mnLoopQuery;
	int mnLoopWords;
	float mLoopScore;
	long unsigned int mnRelocQuery;
	int mnRelocWords;
	float mRelocScore;
	cv::Mat mTcwGBA;
	cv::Mat mTcwBefGBA;
	long unsigned int mnBAGlobalForKF;
	const float fx, fy, cx, cy, invfx, invfy, mbf, mb, mThDepth;		老参数了
	const int N;					还是frame的N，表示特征点的个数
	const std::vector<cv::KeyPoint> mvKeys;		关键点
	const std::vector<cv::KeyPoint> mvKeysUn;	解扭曲的关键点坐标
	const std::vector<float> mvuRight; // negative value for monocular points 创建双目的过程
	const std::vector<float> mvDepth; // negative value for monocular points	深度的过程
	const cv::Mat mDescriptors;	帧的特征点，通过构造函数获得
	fbow::fBow mBowVec;		？？，通过构造函数传入
	fbow::fBow2 mFeatVec;
	cv::Mat mTcp;
	四个边界值，通过构造函数从Frame导入
	const int mnMinX;
	const int mnMinY;
	const int mnMaxX;
	const int mnMaxY;
	const cv::Mat mK;		相机参数
	cv::Mat _globaldescriptors;	全局描述子，通过构造函数传入
	double glbDistance;
	std::vector<MapPoint*> mvpMapPoints;	地图点，通过构造函数传入，后续会通过增加地图点创建帧和地图点关系
protected:
	cv::Mat Tcw;
	cv::Mat Twc;
	cv::Mat Ow;
	cv::Mat Cw; // Stereo middel point. Only for visualization
	KeyFrameDatabase* mpKeyFrameDB;				关键帧数据库，通过构造函数传入
	Vocabulary* mpORBvocabulary;
	std::vector< std::vector <std::vector<size_t> > > mGrid;	用于存放网格中的关键点编号的网格
	std::map<KeyFrame*,int> mConnectedKeyFrameWeights;
	std::vector<KeyFrame*> mvpOrderedConnectedKeyFrames;
	std::vector<int> mvOrderedWeights;
	bool mbFirstConnection;
	KeyFrame* mpParent;
	std::set<KeyFrame*> mspChildrens;
	std::set<KeyFrame*> mspLoopEdges;
	bool mbNotErase;
	bool mbToBeErased;
	bool mbBad;
	float mHalfBaseline; // Only for visualization 一半基线长度，通过构造函数传入
	Map* mpMap;			滴入
	std::mutex mMutexPose;
	std::mutex mMutexConnections;
	std::mutex mMutexFeatures;
	*/
#pragma endregion

long unsigned int KeyFrame::nNextId=0;

/*所有的函数
可以分成以下六类：
1，构造函数
2，位姿相关：设置位姿，获取位姿，获取位姿的逆，获取传感器中心坐标，获取双目中心坐标，获取旋转矩阵，获取平移矩阵
3，共视图相关：增加删除更新连接，跟心最好的共视，获取连接的关键帧，获取共视关键帧，获取最好的共视关键帧，根据权重获得共视，获取权重，设置不要移除，设置要移除，删除关键帧，判断是否需要移除，增加回环边，获取回环边
4，Spanning Tree有关：增加移除子树，更改父节点，获取子节点，判断是否有子节点
5，MapPoint相关：添加地图点，移除地图点的匹配关系，替换地图点匹配，跟踪地图点，
6，其他：利用词袋计算特征，获取特点区域内特征点，获取双目特征点3D坐标，判断是否在图像内
*/

KeyFrame::KeyFrame(Frame &F, Map *pMap, KeyFrameDatabase *pKFDB):	//第一类，构造函数
    mnFrameId(F.mnId),  mTimeStamp(F.mTimeStamp), mnGridCols(FRAME_GRID_COLS), mnGridRows(FRAME_GRID_ROWS),
    mfGridElementWidthInv(F.mfGridElementWidthInv), mfGridElementHeightInv(F.mfGridElementHeightInv),
    mnTrackReferenceForFrame(0), mnFuseTargetForKF(0), mnBALocalForKF(0), mnBAFixedForKF(0),
    mnLoopQuery(0), mnLoopWords(0), mnRelocQuery(0), mnRelocWords(0), mnBAGlobalForKF(0),
    fx(F.fx), fy(F.fy), cx(F.cx), cy(F.cy), invfx(F.invfx), invfy(F.invfy),
    mbf(F.mbf), mb(F.mb), mThDepth(F.mThDepth), N(F.N), mvKeys(F.mvKeys), mvKeysUn(F.mvKeysUn),
    mvuRight(F.mvuRight), mvDepth(F.mvDepth), mDescriptors(F.mDescriptors.clone()),
    mBowVec(F.mBowVec),mFeatVec(F.mFeatVec),
    mnMinX(F.mnMinX), mnMinY(F.mnMinY), mnMaxX(F.mnMaxX),
    mnMaxY(F.mnMaxY), mK(F.mK), mvpMapPoints(F.mvpMapPoints), mpKeyFrameDB(pKFDB),
    mpORBvocabulary(F.mpORBvocabulary), mbFirstConnection(true), mpParent(NULL), mbNotErase(false),
    mbToBeErased(false), mbBad(false), mHalfBaseline(F.mb/2), mpMap(pMap),_globaldescriptors(F.globaldescriptors)
{
    mnId=nNextId++;	//他的ID+1 用来标号

    mGrid.resize(mnGridCols);	//重置栅格的列数
    for(int i=0; i<mnGridCols;i++)
    {
        mGrid[i].resize(mnGridRows);	//里面的每一个都重置为栅格的行数
        for(int j=0; j<mnGridRows; j++)
            mGrid[i][j] = F.mGrid[i][j];	//把栅格里头的信息给到Keyframe的栅格里头
    }

    SetPose(F.mTcw);		//把当前的姿态给该关键帧
}

void KeyFrame::ComputeBoW()
{
    if(mBowVec.empty()|| mFeatVec.empty())
    {
        mpORBvocabulary->transform(mDescriptors,4,mBowVec,mFeatVec);
    }
}

//第二类，位姿有关，设置位姿
void KeyFrame::SetPose(const cv::Mat &Tcw_) 
{
    std::unique_lock<std::mutex> lock(mMutexPose);
    Tcw_.copyTo(Tcw);
    cv::Mat Rcw = Tcw.rowRange(0,3).colRange(0,3);	//左上角3*3
    cv::Mat tcw = Tcw.rowRange(0,3).col(3);			//右边哪一个
    cv::Mat Rwc = Rcw.t();							//转置
    Ow = -Rwc*tcw;									//方便处理用

    Twc = cv::Mat::eye(4,4,Tcw.type());
    Rwc.copyTo(Twc.rowRange(0,3).colRange(0,3));
    Ow.copyTo(Twc.rowRange(0,3).col(3));
    cv::Mat center = (cv::Mat_<float>(4,1) << mHalfBaseline, 0 , 0, 1);
    Cw = Twc*center;
}

//第二类，返回他的位姿矩阵
cv::Mat KeyFrame::GetPose()
{
    std::unique_lock<std::mutex> lock(mMutexPose);
    return Tcw.clone();
}

//第二类，返回位姿矩阵的逆
cv::Mat KeyFrame::GetPoseInverse()
{
    std::unique_lock<std::mutex> lock(mMutexPose);
    return Twc.clone();
}

//第二类，获取相机中心，返回OW
cv::Mat KeyFrame::GetCameraCenter()
{
    std::unique_lock<std::mutex> lock(mMutexPose);
    return Ow.clone();
}

cv::Mat KeyFrame::GetStereoCenter()
{
    std::unique_lock<std::mutex> lock(mMutexPose);
    return Cw.clone();
}


cv::Mat KeyFrame::GetRotation()
{
    std::unique_lock<std::mutex> lock(mMutexPose);
    return Tcw.rowRange(0,3).colRange(0,3).clone();
}

cv::Mat KeyFrame::GetTranslation()
{
    std::unique_lock<std::mutex> lock(mMutexPose);
    return Tcw.rowRange(0,3).col(3).clone();
}

void KeyFrame::AddConnection(KeyFrame *pKF, const int &weight)
{
    {
        std::unique_lock<std::mutex> lock(mMutexConnections);
        if(!mConnectedKeyFrameWeights.count(pKF))
            mConnectedKeyFrameWeights[pKF]=weight;
        else if(mConnectedKeyFrameWeights[pKF]!=weight)
            mConnectedKeyFrameWeights[pKF]=weight;
        else
            return;
    }

    UpdateBestCovisibles();
}

void KeyFrame::UpdateBestCovisibles()
{
    std::unique_lock<std::mutex> lock(mMutexConnections);
    std::vector<std::pair<int,KeyFrame*> > vPairs;
    vPairs.reserve(mConnectedKeyFrameWeights.size());
    for(std::map<KeyFrame*,int>::iterator mit=mConnectedKeyFrameWeights.begin(), mend=mConnectedKeyFrameWeights.end(); mit!=mend; mit++)
       vPairs.push_back(std::make_pair(mit->second,mit->first));

    sort(vPairs.begin(),vPairs.end());
    std::list<KeyFrame*> lKFs;
    std::list<int> lWs;
    for(size_t i=0, iend=vPairs.size(); i<iend;i++)
    {
        lKFs.push_front(vPairs[i].second);
        lWs.push_front(vPairs[i].first);
    }

    mvpOrderedConnectedKeyFrames = std::vector<KeyFrame*>(lKFs.begin(),lKFs.end());
    mvOrderedWeights = std::vector<int>(lWs.begin(), lWs.end());
}

std::set<KeyFrame*> KeyFrame::GetConnectedKeyFrames()
{
    std::unique_lock<std::mutex> lock(mMutexConnections);
    std::set<KeyFrame*> s;
    for(std::map<KeyFrame*,int>::iterator mit=mConnectedKeyFrameWeights.begin();mit!=mConnectedKeyFrameWeights.end();mit++)
        s.insert(mit->first);
    return s;
}

std::vector<KeyFrame*> KeyFrame::GetVectorCovisibleKeyFrames()
{
    std::unique_lock<std::mutex> lock(mMutexConnections);
    return mvpOrderedConnectedKeyFrames;
}

std::vector<KeyFrame*> KeyFrame::GetBestCovisibilityKeyFrames(const int &N)
{
    std::unique_lock<std::mutex> lock(mMutexConnections);
    if((int)mvpOrderedConnectedKeyFrames.size()<N)
        return mvpOrderedConnectedKeyFrames;
    else
        return std::vector<KeyFrame*>(mvpOrderedConnectedKeyFrames.begin(),mvpOrderedConnectedKeyFrames.begin()+N);

}

std::vector<KeyFrame*> KeyFrame::GetCovisiblesByWeight(const int &w)
{
    std::unique_lock<std::mutex> lock(mMutexConnections);

    if(mvpOrderedConnectedKeyFrames.empty())
        return std::vector<KeyFrame*>();

    std::vector<int>::iterator it = upper_bound(mvOrderedWeights.begin(),mvOrderedWeights.end(),w,KeyFrame::weightComp);
    if(it==mvOrderedWeights.end())
        return std::vector<KeyFrame*>();
    else
    {
        int n = it-mvOrderedWeights.begin();
        return std::vector<KeyFrame*>(mvpOrderedConnectedKeyFrames.begin(), mvpOrderedConnectedKeyFrames.begin()+n);
    }
}

int KeyFrame::GetWeight(KeyFrame *pKF)
{
    std::unique_lock<std::mutex> lock(mMutexConnections);
    if(mConnectedKeyFrameWeights.count(pKF))
        return mConnectedKeyFrameWeights[pKF];
    else
        return 0;
}

//第五类，新增地图点
void KeyFrame::AddMapPoint(MapPoint *pMP, const size_t &idx)
{
    std::unique_lock<std::mutex> lock(mMutexFeatures);
    mvpMapPoints[idx]=pMP;
}

void KeyFrame::EraseMapPointMatch(const size_t &idx)
{
    std::unique_lock<std::mutex> lock(mMutexFeatures);
    mvpMapPoints[idx]=static_cast<MapPoint*>(NULL);
}

void KeyFrame::EraseMapPointMatch(MapPoint* pMP)
{
    int idx = pMP->GetIndexInKeyFrame(this);
    if(idx>=0)
        mvpMapPoints[idx]=static_cast<MapPoint*>(NULL);
}


void KeyFrame::ReplaceMapPointMatch(const size_t &idx, MapPoint* pMP)
{
    mvpMapPoints[idx]=pMP;
}

std::set<MapPoint*> KeyFrame::GetMapPoints()
{
    std::unique_lock<std::mutex> lock(mMutexFeatures);
    std::set<MapPoint*> s;
    for(size_t i=0, iend=mvpMapPoints.size(); i<iend; i++)
    {
        if(!mvpMapPoints[i])
            continue;
        MapPoint* pMP = mvpMapPoints[i];
        if(!pMP->isBad())
            s.insert(pMP);
    }
    return s;
}

int KeyFrame::TrackedMapPoints(const int &minObs)
{
    std::unique_lock<std::mutex> lock(mMutexFeatures);

    int nPoints=0;
    const bool bCheckObs = minObs>0;
    for(int i=0; i<N; i++)
    {
        MapPoint* pMP = mvpMapPoints[i];
        if(pMP)
        {
            if(!pMP->isBad())
            {
                if(bCheckObs)
                {
                    if(mvpMapPoints[i]->Observations()>=minObs)
                        nPoints++;
                }
                else
                    nPoints++;
            }
        }
    }

    return nPoints;
}

std::vector<MapPoint*> KeyFrame::GetMapPointMatches()
{
    std::unique_lock<std::mutex> lock(mMutexFeatures);
    return mvpMapPoints;
}

MapPoint* KeyFrame::GetMapPoint(const size_t &idx)
{
    std::unique_lock<std::mutex> lock(mMutexFeatures);
    return mvpMapPoints[idx];
}

void KeyFrame::UpdateConnections()
{
    std::map<KeyFrame*,int> KFcounter;

    std::vector<MapPoint*> vpMP;

    {
        std::unique_lock<std::mutex> lockMPs(mMutexFeatures);
        vpMP = mvpMapPoints;
    }

    //For all map points in keyframe check in which other keyframes are they seen
    //Increase counter for those keyframes
    for(std::vector<MapPoint*>::iterator vit=vpMP.begin(), vend=vpMP.end(); vit!=vend; vit++)
    {
        MapPoint* pMP = *vit;

        if(!pMP)
            continue;

        if(pMP->isBad())
            continue;

        std::map<KeyFrame*,size_t> observations = pMP->GetObservations();

        for(std::map<KeyFrame*,size_t>::iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
        {
            if(mit->first->mnId==mnId)
                continue;
            KFcounter[mit->first]++;
        }
    }

    // This should not happen
    if(KFcounter.empty())
        return;

    //If the counter is greater than threshold add connection
    //In case no keyframe counter is over threshold add the one with maximum counter
    int nmax=0;
    KeyFrame* pKFmax=NULL;
    int th = 15;

    std::vector<std::pair<int,KeyFrame*> > vPairs;
    vPairs.reserve(KFcounter.size());
    for(std::map<KeyFrame*,int>::iterator mit=KFcounter.begin(), mend=KFcounter.end(); mit!=mend; mit++)
    {
        if(mit->second>nmax)
        {
            nmax=mit->second;
            pKFmax=mit->first;
        }
        if(mit->second>=th)
        {
            vPairs.push_back(std::make_pair(mit->second,mit->first));
            (mit->first)->AddConnection(this,mit->second);
        }
    }

    if(vPairs.empty())
    {
        vPairs.push_back(std::make_pair(nmax,pKFmax));
        pKFmax->AddConnection(this,nmax);
    }

    sort(vPairs.begin(),vPairs.end());
    std::list<KeyFrame*> lKFs;
    std::list<int> lWs;
    for(size_t i=0; i<vPairs.size();i++)
    {
        lKFs.push_front(vPairs[i].second);
        lWs.push_front(vPairs[i].first);
    }

    {
        std::unique_lock<std::mutex> lockCon(mMutexConnections);

        // mspConnectedKeyFrames = spConnectedKeyFrames;
        mConnectedKeyFrameWeights = KFcounter;
        mvpOrderedConnectedKeyFrames = std::vector<KeyFrame*>(lKFs.begin(),lKFs.end());
        mvOrderedWeights = std::vector<int>(lWs.begin(), lWs.end());

        if(mbFirstConnection && mnId!=0)
        {
            mpParent = mvpOrderedConnectedKeyFrames.front();
            mpParent->AddChild(this);
            mbFirstConnection = false;
        }

    }
}

void KeyFrame::AddChild(KeyFrame *pKF)
{
    std::unique_lock<std::mutex> lockCon(mMutexConnections);
    mspChildrens.insert(pKF);
}

void KeyFrame::EraseChild(KeyFrame *pKF)
{
    std::unique_lock<std::mutex> lockCon(mMutexConnections);
    mspChildrens.erase(pKF);
}

void KeyFrame::ChangeParent(KeyFrame *pKF)
{
    std::unique_lock<std::mutex> lockCon(mMutexConnections);
    mpParent = pKF;
    pKF->AddChild(this);
}

std::set<KeyFrame*> KeyFrame::GetChilds()
{
    std::unique_lock<std::mutex> lockCon(mMutexConnections);
    return mspChildrens;
}

KeyFrame* KeyFrame::GetParent()
{
    std::unique_lock<std::mutex> lockCon(mMutexConnections);
    return mpParent;
}

bool KeyFrame::hasChild(KeyFrame *pKF)
{
    std::unique_lock<std::mutex> lockCon(mMutexConnections);
    return mspChildrens.count(pKF);
}

void KeyFrame::AddLoopEdge(KeyFrame *pKF)
{
    std::unique_lock<std::mutex> lockCon(mMutexConnections);
    mbNotErase = true;
    mspLoopEdges.insert(pKF);
}

std::set<KeyFrame*> KeyFrame::GetLoopEdges()
{
    std::unique_lock<std::mutex> lockCon(mMutexConnections);
    return mspLoopEdges;
}

void KeyFrame::SetNotErase()
{
    std::unique_lock<std::mutex> lock(mMutexConnections);
    mbNotErase = true;
}

void KeyFrame::SetErase()
{
    {
        std::unique_lock<std::mutex> lock(mMutexConnections);
        if(mspLoopEdges.empty())
        {
            mbNotErase = false;
        }
    }

    if(mbToBeErased)
    {
        SetBadFlag();
    }
}

void KeyFrame::SetBadFlag()
{
    {
        std::unique_lock<std::mutex> lock(mMutexConnections);
        if(mnId==0)
            return;
        else if(mbNotErase)
        {
            mbToBeErased = true;
            return;
        }
    }

    for(std::map<KeyFrame*,int>::iterator mit = mConnectedKeyFrameWeights.begin(), mend=mConnectedKeyFrameWeights.end(); mit!=mend; mit++)
        mit->first->EraseConnection(this);

    for(size_t i=0; i<mvpMapPoints.size(); i++)
        if(mvpMapPoints[i])
            mvpMapPoints[i]->EraseObservation(this);
    {
        std::unique_lock<std::mutex> lock(mMutexConnections);
        std::unique_lock<std::mutex> lock1(mMutexFeatures);

        mConnectedKeyFrameWeights.clear();
        mvpOrderedConnectedKeyFrames.clear();

        // Update Spanning Tree
        std::set<KeyFrame*> sParentCandidates;
        sParentCandidates.insert(mpParent);

        // Assign at each iteration one children with a parent (the pair with highest covisibility weight)
        // Include that children as new parent candidate for the rest
        while(!mspChildrens.empty())
        {
            bool bContinue = false;

            int max = -1;
            KeyFrame* pC;
            KeyFrame* pP;

            for(std::set<KeyFrame*>::iterator sit=mspChildrens.begin(), send=mspChildrens.end(); sit!=send; sit++)
            {
                KeyFrame* pKF = *sit;
                if(pKF->isBad())
                    continue;

                // Check if a parent candidate is connected to the keyframe
                std::vector<KeyFrame*> vpConnected = pKF->GetVectorCovisibleKeyFrames();
                for(size_t i=0, iend=vpConnected.size(); i<iend; i++)
                {
                    for(std::set<KeyFrame*>::iterator spcit=sParentCandidates.begin(), spcend=sParentCandidates.end(); spcit!=spcend; spcit++)
                    {
                        if(vpConnected[i]->mnId == (*spcit)->mnId)
                        {
                            int w = pKF->GetWeight(vpConnected[i]);
                            if(w>max)
                            {
                                pC = pKF;
                                pP = vpConnected[i];
                                max = w;
                                bContinue = true;
                            }
                        }
                    }
                }
            }

            if(bContinue)
            {
                pC->ChangeParent(pP);
                sParentCandidates.insert(pC);
                mspChildrens.erase(pC);
            }
            else
                break;
        }

        // If a children has no covisibility links with any parent candidate, assign to the original parent of this KF
        if(!mspChildrens.empty())
            for(std::set<KeyFrame*>::iterator sit=mspChildrens.begin(); sit!=mspChildrens.end(); sit++)
            {
                (*sit)->ChangeParent(mpParent);
            }

        mpParent->EraseChild(this);
        mTcp = Tcw*mpParent->GetPoseInverse();
        mbBad = true;
    }


    mpMap->EraseKeyFrame(this);
    mpKeyFrameDB->erase(this);
}

//返回mbbad变量
bool KeyFrame::isBad()
{
    std::unique_lock<std::mutex> lock(mMutexConnections);
    return mbBad;
}

void KeyFrame::EraseConnection(KeyFrame* pKF)
{
    bool bUpdate = false;
    {
        std::unique_lock<std::mutex> lock(mMutexConnections);
        if(mConnectedKeyFrameWeights.count(pKF))
        {
            mConnectedKeyFrameWeights.erase(pKF);
            bUpdate=true;
        }
    }

    if(bUpdate)
        UpdateBestCovisibles();
}

std::vector<size_t> KeyFrame::GetFeaturesInArea(const float &x, const float &y, const float &r) const
{
    std::vector<size_t> vIndices;
    vIndices.reserve(N);

    const int nMinCellX = std::max(0,(int)floor((x-mnMinX-r)*mfGridElementWidthInv));
    if(nMinCellX>=mnGridCols)
        return vIndices;

    const int nMaxCellX = std::min((int)mnGridCols-1,(int)ceil((x-mnMinX+r)*mfGridElementWidthInv));
    if(nMaxCellX<0)
        return vIndices;

    const int nMinCellY = std::max(0,(int)floor((y-mnMinY-r)*mfGridElementHeightInv));
    if(nMinCellY>=mnGridRows)
        return vIndices;

    const int nMaxCellY = std::min((int)mnGridRows-1,(int)ceil((y-mnMinY+r)*mfGridElementHeightInv));
    if(nMaxCellY<0)
        return vIndices;

    for(int ix = nMinCellX; ix<=nMaxCellX; ix++)
    {
        for(int iy = nMinCellY; iy<=nMaxCellY; iy++)
        {
            const std::vector<size_t> vCell = mGrid[ix][iy];
            for(size_t j=0, jend=vCell.size(); j<jend; j++)
            {
                const cv::KeyPoint &kpUn = mvKeysUn[vCell[j]];
                const float distx = kpUn.pt.x-x;
                const float disty = kpUn.pt.y-y;

                if(fabs(distx)<r && fabs(disty)<r)
                    vIndices.push_back(vCell[j]);
            }
        }
    }

    return vIndices;
}

bool KeyFrame::IsInImage(const float &x, const float &y) const
{
    return (x>=mnMinX && x<mnMaxX && y>=mnMinY && y<mnMaxY);
}

cv::Mat KeyFrame::UnprojectStereo(int i)
{
    const float z = mvDepth[i];
    if(z>0)
    {
        const float u = mvKeys[i].pt.x;
        const float v = mvKeys[i].pt.y;
        const float x = (u-cx)*z*invfx;
        const float y = (v-cy)*z*invfy;
        cv::Mat x3Dc = (cv::Mat_<float>(3,1) << x, y, z);

        std::unique_lock<std::mutex> lock(mMutexPose);
        return Twc.rowRange(0,3).colRange(0,3)*x3Dc+Twc.rowRange(0,3).col(3);
    }
    else
        return cv::Mat();
}

float KeyFrame::ComputeSceneMedianDepth(const int q)
{
    std::vector<MapPoint*> vpMapPoints;
    cv::Mat Tcw_;
    {
        std::unique_lock<std::mutex> lock(mMutexFeatures);
        std::unique_lock<std::mutex> lock2(mMutexPose);
        vpMapPoints = mvpMapPoints;
        Tcw_ = Tcw.clone();
    }

    std::vector<float> vDepths;
    vDepths.reserve(N);
    cv::Mat Rcw2 = Tcw_.row(2).colRange(0,3);
    Rcw2 = Rcw2.t();
    float zcw = Tcw_.at<float>(2,3);
    for(int i=0; i<N; i++)
    {
        if(mvpMapPoints[i])
        {
            MapPoint* pMP = mvpMapPoints[i];
            cv::Mat x3Dw = pMP->GetWorldPos();
            float z = Rcw2.dot(x3Dw)+zcw;
            vDepths.push_back(z);
        }
    }

    sort(vDepths.begin(),vDepths.end());

    return vDepths[(vDepths.size()-1)/q];
}

} //namespace ORB_SLAM
