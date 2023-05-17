# graduation_project_TYUT
这是我的毕业设计的一部分
# 代码使用规则
以上代码替换DXSLAM框架中的代码；
DXSLAM代码链接：https://github.com/ivipsourcecode/dxslam；
将该文件夹中的代码替换DXSLAM框架中的对应位置的代码或文件夹，后运行指令
> run_tum_cc /home/zhehan-yang/Desktop/DX-SLAM/dxslam-master-forreal/Vocabulary/DXSLAM.fbow /media/zhehan-yang/yzh3/ubuntu/datasheet/TUM/parameter/TUM1.yaml /media/zhehan-yang/yzh3/ubuntu/datasheet/TUM/Handheld_SLAM/rgbd_dataset_freiburg1_floor_deblur /media/zhehan-yang/yzh3/ubuntu/datasheet/TUM/Handheld_SLAM/rgbd_dataset_freiburg1_floor_deblur/associations.txt /media/zhehan-yang/yzh3/ubuntu/datasheet/TUM/Handheld_SLAM/rgbd_dataset_freiburg1_floor_deblur/feature_hfnet


# 需要修改的位置：
1，run_tum_cc，129行：
CreateModelAndInitial函数，参数为运行的虚拟环境路径与本本文件中hf-net文件夹路径
2，CMAKE文件最后含Python的文件夹对应虚拟环境中的包和动态链接库路径

3，target_link_libraries(rgbd_tum -lpython3.6m -lpthread -ldl  -lutil -lrt -lm)
仅需要修改-lpython3.6m即可


