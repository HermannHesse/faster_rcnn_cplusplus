cmake_minimum_required(VERSION 3.6)
project(fast_rcnn_c__)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(PYTHON_INCLUDE_DIR "/usr/include/python2.7")
set(CAFFE_INCLUDE_DIR "caffe-fast-rcnn/include")
set(CAFFE_LIBRARY_DIR "caffe-fast-rcnn/build/lib")
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)
#include_directories(/home/droneplan/Brais/py-faster-rcnn/caffe-fast-rcnn/include)
#include_directories(/home/droneplan/Brais/py-faster-rcnn/caffe-fast-rcnn/python)
#include_directories(/home/droneplan/Brais/py-faster-rcnn/lib/nms)
#include_directories(/usr/local/cuda-7.5/targets/x86_64-linux/include/)
include_directories(${PYTHON_INCLUDE_DIR} ${CAFFE_INCLUDE_DIR})
link_directories(${CAFFE_LIBRARY_DIR})
#include_directories(/usr/include)
#set(CUDA_USE_STATIC_CUDA_RUNTIME "OFF")
set(SOURCE_FILES main.cpp faster_rcnn.cpp lib/nms/nms_kernel.cu)
cuda_add_executable(fast_rcnn_c__ ${SOURCE_FILES})
target_link_libraries(fast_rcnn_c__ caffe python2.7 glog boost_system opencv_highgui opencv_core opencv_imgproc)

# lpython2.7        ->  /usr/lib/python2.7/config-x86_64-linux-gnu/libpython2.7.so
# lglog             ->  /usr/lib/x86_64-linux-gnu/libglog.so
# lboost_system     ->  /usr/lib/x86_64-linux-gnu/libboost_system.so
# lboost_python     ->  /usr/lib/x86_64-linux-gnu/libboost_python.so NO NEED
# lopencv_highgui   ->  /usr/lib/x86_64-linux-gnu/libopencv_highgui.so


