#include "faster_rcnn.hpp"


int main()
{
    string model_file = "data/pascal_voc/faster_rcnn_test.prototxt";
    string weights_file = "data/pascal_voc/ZF_faster_rcnn_final.caffemodel";
    string labels_file = "data/pascal_voc/synset_words.txt";
    int GPUID = 0;

    Faster_RCNN detector(model_file, weights_file, labels_file, GPUID);

    vector<Detection> detections;
    cv::Mat image = cv::imread("data/pascal_voc/2007_000847.jpg");
    detector.detect(image, detections);

    std::cout<<"x\ty\twidth\theight\tcategory\tscore"<<std::endl;
    for(int i = 0; i < detections.size(); i++) {
        std::cout<<detections[i].x<<"\t"<<detections[i].y<<"\t"
                 <<detections[i].width<<"\t"<<detections[i].height<<"\t"
                 <<detections[i].category<<"\t"<<detections[i].score<<std::endl;
    }

    detector.vis_detections(image, detections);
    return 0;
}
