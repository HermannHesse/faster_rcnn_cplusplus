#ifndef FASTER_RCNN_HPP
#define FASTER_RCNN_HPP

#include <string>
#include <vector>
#include <math.h>
#include <fstream>
#include <boost/python.hpp>
#include "caffe/caffe.hpp"
#include "py-faster-rcnn/lib/nms/gpu_nms.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace caffe;
using namespace std;

#define CONF_THRESH 0.8
#define NMS_THRESH 0.3
#define MAX_INPUT_SIDE 1000;
#define MIN_INPUT_SIDE 600;

/*
 * ===  Struct  ======================================================================
 *         Name:  Detection
 *  Description:  Struct to return each detection
 * =====================================================================================
 */
struct Detection {
    float x;
    float y;
    float width;
    float height;
    float score;
    int categoryIndex;
    string category;
};

/*
 * ===  Struct  ======================================================================
 *         Name:  Info
 *  Description:  Used for bbox sort
 * =====================================================================================
 */
struct Info {
    float score;
    const float* head;

    bool operator <(const Info& info) {
        return (info.score < score);
    }
};

/*
 * ===  Class  ======================================================================
 *         Name:  Faster_RCNN
 *  Description:  FasterRCNN C++ Detector
 * =====================================================================================
 */
class Faster_RCNN {
public:
    Faster_RCNN(const string& model_file, const string& weights_file,
                const string& labels_file, const int GPUID);
    void detect(cv::Mat cv_image, vector<Detection>& detections);
    void vis_detections(cv::Mat& cv_image, vector<Detection> detections);

private:
    void preprocess(const cv::Mat cv_image, float *im_info);
    void bbox_sort(int num, const float* pred, float *sorted_pred);
    void bbox_transform_inv(const int num_rois, const float* box_deltas, const float* pred_cls,
                            float* boxes, float* pred, int img_height, int img_width);

private:
    Faster_RCNN(){}
    std::shared_ptr<Net<float> > net_;
    std::vector<string> labels_;
    int num_channels_;
    int num_clases_;
};


#endif
