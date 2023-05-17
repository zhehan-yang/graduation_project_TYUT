//
// Created by zhehan-yang on 07/03/23.
//

#ifndef DXSLAM_MASTER_LABEL_H
#define DXSLAM_MASTER_LABEL_H

#endif //DXSLAM_MASTER_LABEL_H

#include<iostream>
#include<map>
#include<queue>
#include<string>

using namespace std;


class Label {
    enum label {
        Initialing = 0,
        NormalRunning = 1,
        ReLocalization = 2
    };

    string enum2str(label lab) {
        switch (lab) {
            case Initialing:
                return "Initialing";
            case NormalRunning:
                return "NormalRunning";
            case ReLocalization:
                return "ReLocalization";

        }
    }

public:
    queue<pair<label, double>> labelGroups;
    map<label, double> mapForTotal;

    void printsALLLabels(bool easyMode = true, bool needAnalysis = true) {
        //empty
        if (labelGroups.empty()) {
            cout << "The container is empty!!" << endl;
            return;
        }else{
            cout<<endl<<"Start to print All Change time!!"<<endl<<"----------------------"<<endl;
        }
        //output easily
        if (easyMode) {
//            for (auto it = labelGroups.begin(); it != labelGroups.end(); it++) {
//                cout << enum_name(it->first) << " ==> " << it->second << endl;
//            }
            double startTime=labelGroups.front().second;
            while (!labelGroups.empty()) {
                cout << enum2str(labelGroups.front().first) << " ==> " << labelGroups.front().second-startTime << endl;
                labelGroups.pop();
            }
            //output concisely
        } else {
//            for (auto it = labelGroups.begin(), itLast = it; it != labelGroups.end(); it++) {
//                cout << enum_name(itLast->first) << " ==> " << enum_name(it->first) << " at " << it->second << " seconds" << endl;
//            }
            while (!labelGroups.empty()) {
                cout << "Programme Change to Mode " << enum2str(labelGroups.front().first) << " at "
                     << labelGroups.front().second << " Seconds." << endl;
                labelGroups.pop();
            }
        }

        //Analysis
        if (needAnalysis) {
            double total = mapForTotal[Initialing] + mapForTotal[NormalRunning] + mapForTotal[ReLocalization];
            cout.precision(4);
            cout << "Analysis finished,Total time of datasheet series is " << total << endl;
            cout << "Initialize ratio:" << mapForTotal[Initialing] / total * 100 << "%" << endl;
            cout << "Normal Running ratio:" << mapForTotal[NormalRunning] / total * 100 << "%" << endl;
            cout << "Relocalization ratio:" << mapForTotal[ReLocalization] / total * 100 << "%" << endl;
        }
    }

    void AddChange(int input, double time, bool switchFromState = false) {
        label lab;
//        enum eTrackingState{
//            SYSTEM_NOT_READY=-1,
//            NO_IMAGES_YET=0,
//            NOT_INITIALIZED=1,
//            OK=2,
//            LOST=3
//        };
        if (switchFromState) {
            switch (input) {
                case -1:
                case 0:
                    return;
                case 1:
                    lab=Initialing;
                    break;
                case 2:
                    lab=NormalRunning;
                    break;
                case 3:
                    lab=ReLocalization;
                    break;
                default:
                    return;
            }
        }else{
            switch (input) {
                case 0:
                    lab=Initialing;
                    break;
                case 1:
                    lab=NormalRunning;
                case 2:
                    lab=ReLocalization;
                default:
                    return;
            }
        }
        if(!labelGroups.empty()){
            mapForTotal[labelGroups.back().first] += time - labelGroups.back().second;
        }
//        auto temp= make_pair(lab,time);
        labelGroups.push(make_pair(lab, time));
    }

    void clear() {
        while (!labelGroups.empty()) labelGroups.pop();
    }
};
