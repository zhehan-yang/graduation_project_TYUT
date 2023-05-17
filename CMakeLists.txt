cmake_minimum_required(VERSION 2.8)
#cmake版本最低2.8
project(DXSLAM)

add_definitions(-w)
#网上给的都是-D，对应在主代码里头的#ifdef USE..

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()
#如果没有确认，那么就是Release


MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})
#输出cmake build type是什么

set( Cnpy_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/Thirdparty/cnpy )
set( Cnpy_LIBS ${PROJECT_SOURCE_DIR}/Thirdparty/cnpy/build/libcnpy.so )

set( Fbow_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}/Thirdparty/fbow/src/" )
set( Fbow_LIBS "${PROJECT_SOURCE_DIR}/Thirdparty/fbow/build/src/libfbow.so" )
#上面四行定义了四个变量

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall  -O3 -march=native ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall   -O3 -march=native")
#分别针对C编译器和C++编译器 -O3表示优化等级 -march=native,它告诉编译器调整生成的代码以用于主机CPU的微架构和ISA扩展.即使它没有使用相同的名称

# Check C++11 or C++0x support  在检查当前编译器是否支持c++11依次向下取
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
   add_definitions(-DCOMPILEDWITHC11)
   message(STATUS "Using flag -std=c++11.")
elseif(COMPILER_SUPPORTS_CXX0X)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
   add_definitions(-DCOMPILEDWITHC0X)
   message(STATUS "Using flag -std=c++0x.")
else()
   message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)
#list( APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake_modules) 这个cmake_modules理解为那些cmake文件的捷径，例如opencv之类的
#include_directories( ${PROJECT_SOURCE_DIR}/include)这个是用来添加头文件的
#add_library(test_install STATIC src/test_install.cpp)这个是用来添加源文件的


find_package(OpenCV 4 QUIET)
if(NOT OpenCV_FOUND)
find_package(OpenCV 3.0 QUIET)
endif()
if(NOT OpenCV_FOUND)
   find_package(OpenCV 2.4.3 QUIET)
   if(NOT OpenCV_FOUND)
      message(FATAL_ERROR "OpenCV > 2.4.3 not found.")
   endif()
endif()
#一直找opencv，最低2.4.3，可以向上找到4的opencv

find_package(Eigen3 3.1.0 REQUIRED)  
find_package(Pangolin REQUIRED)
find_package (Threads)
#这个Thread是一个线程类

include_directories(
${PROJECT_SOURCE_DIR}
${PROJECT_SOURCE_DIR}/include
${EIGEN3_INCLUDE_DIR}
${Pangolin_INCLUDE_DIRS}
${Cnpy_INCLUDE_DIRS}
${Fbow_INCLUDE_DIRS}
)
#上面是include头文件的，然后cnpy是把传感器的数据格式化的方法，其他的库都知道

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
#设置编译好的库的防止位置

add_library(${PROJECT_NAME} SHARED
src/System.cc
src/Tracking.cc
src/LocalMapping.cc
src/LoopClosing.cc
        src/Matcher.cc
src/FrameDrawer.cc
src/Converter.cc
src/MapPoint.cc
src/KeyFrame.cc
src/Map.cc
src/MapDrawer.cc
src/Optimizer.cc
src/PnPsolver.cc
src/Frame.cc
src/KeyFrameDatabase.cc
src/Sim3Solver.cc
src/Initializer.cc
src/Viewer.cc
        include/Label.h)
#这些全部都要编译成库，名称是项目的名称


target_link_libraries(${PROJECT_NAME}
${OpenCV_LIBS}
${EIGEN3_LIBS}
${Pangolin_LIBRARIES}
${PROJECT_SOURCE_DIR}/Thirdparty/g2o/lib/libg2o.so
${PROJECT_SOURCE_DIR}/Thirdparty/DBoW2/lib/libDBoW2.so
${Cnpy_LIBS}
${Fbow_LIBS}
)
#把这些全部都链接上去

# Build examples

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/RGB-D)
#存放可执行软件的目录

include_directories(/home/zhehan-yang/miniconda3/envs/DXSLAM/include/python3.6m)
include_directories(/home/zhehan-yang/miniconda3/envs/DXSLAM/lib/python3.6/site-packages/numpy/core/include)
link_directories(/home/zhehan-yang/miniconda3/envs/DXSLAM/lib/)
link_directories(/home/zhehan-yang/miniconda3/envs/DXSLAM/lib/python3.6/config-3.6m-x86_64-linux-gnu)
link_directories(/home/zhehan-yang/miniconda3/envs/DXSLAM/lib/python3.6/site-packages/numpy/core/lib)

add_executable(rgbd_tum Examples/RGB-D/rgbd_tum.cc)
target_link_libraries(rgbd_tum ${PROJECT_NAME})
target_link_libraries(rgbd_tum ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(rgbd_tum -lpython3.6m -lpthread -ldl  -lutil -lrt -lm)
