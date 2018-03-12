#include "faster_rcnn.hpp"

using namespace caffe;
using namespace std;

/*
 * ===  FUNCTION  ======================================================================
 *         Name:  Faster_RCNN
 *  Description:  Load the model file and weights file
 *       Output:  Constructor
 * =====================================================================================
 */
Faster_RCNN::Faster_RCNN(const string& model_file, const string& weights_file, const string& labels_file, const int GPUID) {
    Caffe::SetDevice(GPUID);
    Caffe::set_mode(Caffe::GPU);
    net_ = std::shared_ptr<Net<float> >(new Net<float>(model_file, caffe::TEST));
    net_->CopyTrainedLayersFrom(weights_file);

    CHECK_EQ(net_->num_inputs(), 2) << "Network should have exactly two inputs.";
    CHECK_EQ(net_->num_outputs(), 2) << "Network should have exactly two outputs.";

    Blob<float>* input_layer = net_->input_blobs()[0];
    num_channels_ = input_layer->channels();
    CHECK_EQ(num_channels_, 3) << "Input layer should have 3 channels.";

    /* Load labels. */
    std::ifstream labels(labels_file.c_str());
    CHECK(labels) << "Unable to open labels file " << labels_file;
    string line;
    while (std::getline(labels, line))
        labels_.push_back(string(line));

    Blob<float>* cls_layer = net_->output_blobs()[1];
    Blob<float>* bbox_layer = net_->output_blobs()[0];
    CHECK_EQ(labels_.size(), cls_layer->channels()) << "Number of labels is different from the output layer dimension.";
    CHECK_EQ(labels_.size()*4, bbox_layer->channels()) << "Number of labels is different from the output layer dimension.";
    num_clases_ = labels_.size();

}

/*
 * ===  FUNCTION  ======================================================================
 *         Name:  Detect
 *  Description:  Perform detection operation
 *                Warning the max input size should less than 1000*600
 *       Output:  vector<Detection> detections
 * =====================================================================================
 */
void Faster_RCNN::detect(cv::Mat cv_image, vector<Detection>& detections) {

    if(cv_image.empty()) {
        std::cout<<"Can not reach the image"<<std::endl;
        return;
    }
    
    int height = cv_image.rows;
    int width = cv_image.cols;

    /* It is necessary to pass as pointers in order to keep them in the net */
    float *im_info = new float[3];

    preprocess(cv_image, im_info);

    int height_resized = int(im_info[0]);
    int width_resized = int(im_info[1]);
    float img_scale = im_info[2];

    net_->ForwardFrom(0);
    const float* bbox_delt = net_->blob_by_name("bbox_pred")->cpu_data();
    const float* pred_cls = net_->blob_by_name("cls_prob")->cpu_data(); // Array n*5 con las probabilidades de las cinco clases para cada roi
    const float* rois = net_->blob_by_name("rois")->cpu_data(); // Array n*5 donde en for 0<n:[n*5+0] == 0 y de [1-4] son las cordenadas de la roi
    const int num_rois = net_->blob_by_name("rois")->num();

    float *boxes = new float[num_rois*4];
    float *pred = new float[num_rois*5*num_clases_];
    int *num_keep = new int[num_clases_];

    float **pred_per_class = new float*[num_clases_];
    float **sorted_pred_cls = new float*[num_clases_];
    int **keep = new int*[num_clases_];
    for (int i = 0; i < num_clases_; i++) {
        pred_per_class[i] = new float[num_rois*5];
        sorted_pred_cls[i] = new float[num_rois*5];
        keep[i] = new int[num_rois];
    }


    for (int n = 0; n < num_rois; n++)
        for (int c = 0; c < 4; c++)
            boxes[n*4+c] = rois[n*5+c+1] / img_scale; //rois[n*5] == 0 SIEMPRE

    bbox_transform_inv(num_rois, bbox_delt, pred_cls, boxes, pred, height, width);

    /* Background class is ignored hereafter */
    for (int i = 1; i < num_clases_; i++) {
        for (int j = 0; j < num_rois; j++)
            for (int k = 0; k < 5; k++)
                pred_per_class[i][j*5+k] = pred[(i*num_rois+j)*5+k];

        bbox_sort(num_rois, pred_per_class[i], sorted_pred_cls[i]);
        _nms(keep[i], &num_keep[i], sorted_pred_cls[i], num_rois, 5, NMS_THRESH, 0);

    }

    int k = 0;
    for (int i = 1; i < num_clases_; i ++) {
        while (sorted_pred_cls[i][keep[i][k] * 5 + 4] > CONF_THRESH && k < num_keep[i]) {

            Detection aux;
            aux.x = sorted_pred_cls[i][keep[i][k] * 5 + 0];
            aux.y = sorted_pred_cls[i][keep[i][k] * 5 + 1];
            aux.width = sorted_pred_cls[i][keep[i][k] * 5 + 2] - aux.x;
            aux.height = sorted_pred_cls[i][keep[i][k] * 5 + 3] - aux.y;
            aux.score = sorted_pred_cls[i][keep[i][k] * 5 + 4];
            aux.categoryIndex = i;
            aux.category = labels_[i];
            detections.push_back(aux);
            k++;
        }

        k = 0;
    }

    delete []im_info;
    delete []boxes;
    delete []pred;
    for (int i = 0; i < num_clases_; i++) {
        delete []pred_per_class[i];
        delete []sorted_pred_cls[i];
        delete []keep[i];
    }
    delete []pred_per_class;
    delete []sorted_pred_cls;
    delete []keep;

}

/*
 * ===  FUNCTION  ======================================================================
 *         Name:  Preprocess
 *  Description:  Compute mean substract
 *                Copy input image into the net
 *       Output:  float im_info[height, width, img_scale]
 * 	              It is need to pass "im_info" through the functions to keep the net
 *                blob("im_info") filled. 
 *                "set_cpu_data" makes just a pointer to the memory
 * =====================================================================================
 */
void Faster_RCNN::preprocess(const cv::Mat cv_image, float* im_info) {

    cv::Mat cv_new(cv_image.rows, cv_image.cols, CV_32FC3, cv::Scalar(0,0,0));

    int height = cv_image.rows;
    int width = cv_image.cols;

    /* Mean normalization (in this case it may not be the average of the training) */
    for (int h = 0; h < height; ++h ) {
        for (int w = 0; w < width; ++w) {
            cv_new.at<cv::Vec3f>(cv::Point(w, h))[0] = float(cv_image.at<cv::Vec3b>(cv::Point(w, h))[0])-float(102.9801);
            cv_new.at<cv::Vec3f>(cv::Point(w, h))[1] = float(cv_image.at<cv::Vec3b>(cv::Point(w, h))[1])-float(115.9465);
            cv_new.at<cv::Vec3f>(cv::Point(w, h))[2] = float(cv_image.at<cv::Vec3b>(cv::Point(w, h))[2])-float(122.7717);
        }
    }

    /* Max image size comparation to know if resize is needed */
    int max_side = MAX(height, width);
    int min_side = MIN(height, width);

    float max_side_scale = float(max_side) / MAX_INPUT_SIDE;
    float min_side_scale = float(min_side) / MIN_INPUT_SIDE;
    float max_scale = MAX(max_side_scale, min_side_scale);
    float img_scale = 1;

    if(max_scale > 1)
        img_scale = float(1) / max_scale;

    int height_resized = int(height * img_scale);
    int width_resized = int(width * img_scale);

    cv::Mat cv_resized;
    cv::resize(cv_new, cv_resized, cv::Size(width_resized, height_resized));

    float data_buf[height_resized*width_resized*3];

    for (int h = 0; h < height_resized; ++h )
    {
        for (int w = 0; w < width_resized; ++w)
        {
            data_buf[(0*height_resized+h)*width_resized+w] = float(cv_resized.at<cv::Vec3f>(cv::Point(w, h))[0]);
            data_buf[(1*height_resized+h)*width_resized+w] = float(cv_resized.at<cv::Vec3f>(cv::Point(w, h))[1]);
            data_buf[(2*height_resized+h)*width_resized+w] = float(cv_resized.at<cv::Vec3f>(cv::Point(w, h))[2]);
        }
    }

    net_->blob_by_name("data")->Reshape(1, num_channels_, height_resized, width_resized);
    //net_->blob_by_name("data")->set_cpu_data(data_buf);
    Blob<float> * input_blobs= net_->input_blobs()[0];
    switch(Caffe::mode()){
        case Caffe::CPU:
            memcpy(input_blobs->mutable_cpu_data(), data_buf, sizeof(float) * input_blobs->count());
            break;
        case Caffe::GPU:
            caffe_gpu_memcpy(sizeof(float)* input_blobs->count(), data_buf, input_blobs->mutable_gpu_data());
            break;
        default:
            LOG(FATAL)<<"Unknow Caffe mode";
    }

    im_info[0] = height_resized;
    im_info[1] = width_resized;
    im_info[2] = img_scale;

    net_->blob_by_name("im_info")->set_cpu_data(im_info);
}

/*
 * ===  FUNCTION  ======================================================================
 *         Name:  vis_detections
 *  Description:  Visuallize the detection result
 *       Output:  None
 * =====================================================================================
 */
void Faster_RCNN::vis_detections(cv::Mat& cv_image, vector<Detection> detections)
{
    int fontFace = cv::FONT_HERSHEY_PLAIN;
    double fontScale = 2;
    double thickness = 1.5;
    int baseline = 0;

    for(int i = 0; i < detections.size(); i++) {
        cv::rectangle(cv_image, cv::Point(detections[i].x,detections[i].y),
                      cv::Point(detections[i].x + detections[i].width,detections[i].y + detections[i].height),
                      cv::Scalar(0, 0, 255), 2, 8, 0);
        string text = detections[i].category + " " + std::to_string(detections[i].score);
        cv::Size textSize = cv::getTextSize(text, fontFace, fontScale, thickness, &baseline);
        cv::rectangle(cv_image, cv::Point(detections[i].x,detections[i].y - 2),
                      cv::Point(detections[i].x + textSize.width/1.3, detections[i].y - 2 - textSize.height),
                      cv::Scalar::all(180), CV_FILLED);
        cv::putText(cv_image, text, cv::Point(detections[i].x,detections[i].y - 2),
                    fontFace, thickness, cv::Scalar::all(0), fontScale, 8);
    }

    cv::imshow("Detections", cv_image);
    cv::waitKey(0);
}

/*
 * ===  FUNCTION  ======================================================================
 *         Name:  boxes_sort
 *  Description:  Sort the bounding box according score
 *       Output:  float* sorted_pred
 *                An ordered pointer derived from "pred" by its score 
 * =====================================================================================
 */
void Faster_RCNN::bbox_sort(const int num_rois, const float* pred, float* sorted_pred)
{
    vector<Info> my;
    Info tmp;
    for (int i = 0; i< num_rois; i++) {
        tmp.score = pred[i*5 + 4];
        tmp.head = pred + i*5;
        my.push_back(tmp);
    }

    std::sort(my.begin(), my.end());

    for (int i=0; i < num_rois; i++)
        for (int j=0; j<5; j++)
            sorted_pred[i*5+j] = my[i].head[j];
}

/*
 * ===  FUNCTION  ======================================================================
 *         Name:  bbox_transform_inv
 *  Description:  Compute bounding box regression value
 *       Output:  float* pred
 *                A pointer "pred" is formed with predictions [x, y, width, height, 
 *                pred_category] ordered in such a way that all predictions of category
 *                '0' go first, those of category '1' after...
 *                So that the predictions derived from roi[0] appear in the pred[0 to 4] 
 *                for category '0' and in the pred[(1*class_num+0)*5 to 
 *                (1*class_num+0)*5+4] for category '1'
 * =====================================================================================
 */
void Faster_RCNN::bbox_transform_inv(int num_rois, const float* box_deltas, const float* pred_cls, float* boxes, float* pred, int img_height, int img_width)  {
    
    float width, height, ctr_x, ctr_y, dx, dy, dw, dh, pred_ctr_x, pred_ctr_y, pred_w, pred_h;

    for(int i = 0; i < num_rois; i++) {

        width = boxes[i*4+2] - boxes[i*4+0] + 1.0;
        height = boxes[i*4+3] - boxes[i*4+1] + 1.0;
        ctr_x = boxes[i*4+0] + 0.5 * width;
        ctr_y = boxes[i*4+1] + 0.5 * height;

        for (int j=0; j< num_clases_; j++) {

            dx = box_deltas[(i*num_clases_+j)*4+0];
            dy = box_deltas[(i*num_clases_+j)*4+1];
            dw = box_deltas[(i*num_clases_+j)*4+2];
            dh = box_deltas[(i*num_clases_+j)*4+3];
            pred_ctr_x = ctr_x + width*dx;
            pred_ctr_y = ctr_y + height*dy;
            pred_w = width * exp(dw);
            pred_h = height * exp(dh);
            pred[(j*num_rois+i)*5+0] = MAX(MIN(pred_ctr_x - 0.5* pred_w, img_width -1), 0);
            pred[(j*num_rois+i)*5+1] = MAX(MIN(pred_ctr_y - 0.5* pred_h, img_height -1), 0);
            pred[(j*num_rois+i)*5+2] = MAX(MIN(pred_ctr_x + 0.5* pred_w, img_width -1), 0);
            pred[(j*num_rois+i)*5+3] = MAX(MIN(pred_ctr_y + 0.5* pred_h, img_height -1), 0);
            pred[(j*num_rois+i)*5+4] = pred_cls[i*num_clases_+j];
        }
    }

}
