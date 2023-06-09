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
#include "Map.h"

#include<mutex>

namespace DXSLAM
{
#pragma region Map类成员函数
	/*
	std::vector<KeyFrame*> mvpKeyFrameOrigins;	最初关键帧	
	std::mutex mMutexMapUpdate;
	std::mutex mMutexPointCreation;
protected:
	std::set<MapPoint*> mspMapPoints;  地图下的地图点集合
	std::set<KeyFrame*> mspKeyFrames;
	std::vector<MapPoint*> mvpReferenceMapPoints;
	long unsigned int mnMaxKFid; 最大的ID关键帧
	int mnBigChangeIdx;		最后变动的关键帧
	std::mutex mMutexMap;
	*/
#pragma endregion

//构造函数
Map::Map():mnMaxKFid(0),mnBigChangeIdx(0)	//初始化最大ID关键帧和最后变动的关键帧
{
}

void Map::AddKeyFrame(KeyFrame *pKF)
{
    std::unique_lock<std::mutex> lock(mMutexMap);	//写的不一样，但是作用一样
	//unique_lock<mutex> lock(mMutexMap); 原写法
	mspKeyFrames.insert(pKF);		//这个是vector类型的
    if(pKF->mnId>mnMaxKFid)		//如果大于最大的允许ID，那么最大ID就是他
        mnMaxKFid=pKF->mnId;
}

//增加一个地图点，这个地图点一般是处理好的
void Map::AddMapPoint(MapPoint *pMP)
{
    std::unique_lock<std::mutex> lock(mMutexMap);
    mspMapPoints.insert(pMP);
}

void Map::EraseMapPoint(MapPoint *pMP)
{
    std::unique_lock<std::mutex> lock(mMutexMap);
    mspMapPoints.erase(pMP);

    // TODO: This only erase the pointer.
    // Delete the MapPoint
}

void Map::EraseKeyFrame(KeyFrame *pKF)
{
    std::unique_lock<std::mutex> lock(mMutexMap);
    mspKeyFrames.erase(pKF);

    // TODO: This only erase the pointer.
    // Delete the MapPoint
}

//设置参考地图点
void Map::SetReferenceMapPoints(const std::vector<MapPoint *> &vpMPs)
{
    std::unique_lock<std::mutex> lock(mMutexMap);
    mvpReferenceMapPoints = vpMPs;
}

void Map::InformNewBigChange()
{
    std::unique_lock<std::mutex> lock(mMutexMap);
    mnBigChangeIdx++;
}

int Map::GetLastBigChangeIdx()
{
    std::unique_lock<std::mutex> lock(mMutexMap);
    return mnBigChangeIdx;
}

std::vector<KeyFrame*> Map::GetAllKeyFrames()
{
    std::unique_lock<std::mutex> lock(mMutexMap);
    return std::vector<KeyFrame*>(mspKeyFrames.begin(),mspKeyFrames.end());
}

//获得所有的地图点
std::vector<MapPoint*> Map::GetAllMapPoints()
{
    std::unique_lock<std::mutex> lock(mMutexMap);
    return std::vector<MapPoint*>(mspMapPoints.begin(),mspMapPoints.end());
}

//返回地图里有几个地图点
long unsigned int Map::MapPointsInMap()
{
    std::unique_lock<std::mutex> lock(mMutexMap);
    return mspMapPoints.size();
}

long unsigned int Map::KeyFramesInMap()
{
    std::unique_lock<std::mutex> lock(mMutexMap);
    return mspKeyFrames.size();
}

std::vector<MapPoint*> Map::GetReferenceMapPoints()
{
    std::unique_lock<std::mutex> lock(mMutexMap);
    return mvpReferenceMapPoints;
}

long unsigned int Map::GetMaxKFid()
{
    std::unique_lock<std::mutex> lock(mMutexMap);
    return mnMaxKFid;
}

void Map::clear()
{
    for(std::set<MapPoint*>::iterator sit=mspMapPoints.begin(), send=mspMapPoints.end(); sit!=send; sit++)
        delete *sit;

    for(std::set<KeyFrame*>::iterator sit=mspKeyFrames.begin(), send=mspKeyFrames.end(); sit!=send; sit++)
        delete *sit;

    mspMapPoints.clear();
    mspKeyFrames.clear();
    mnMaxKFid = 0;
    mvpReferenceMapPoints.clear();
    mvpKeyFrameOrigins.clear();
}

} //namespace ORB_SLAM
