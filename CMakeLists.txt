cmake_minimum_required(VERSION 3.6)
project(faster_rcnn_cplusplus)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(PYTHON_INCLUDE_DIR "/usr/include/python2.7")
set(CAFFE_INCLUDE_DIR "py-faster-rcnn/caffe-fast-rcnn/include")
set(CAFFE_LIBRARY_DIR "py-faster-rcnn/caffe-fast-rcnn/build/lib")

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

include_directories(${PYTHON_INCLUDE_DIR} ${CAFFE_INCLUDE_DIR})
link_directories(${CAFFE_LIBRARY_DIR})

set(SOURCE_FILES main.cpp faster_rcnn.cpp py-faster-rcnn/lib/nms/nms_kernel.cu)
cuda_add_executable(faster_rcnn_cplusplus ${SOURCE_FILES})
target_link_libraries(faster_rcnn_cplusplus caffe python2.7 glog boost_system opencv_highgui opencv_core opencv_imgproc)
