//
// Created by  on 28/11/2017.
//

#ifndef SWIFTPR_SEGMENTATIONFREERECOGNIZER_H
#define SWIFTPR_SEGMENTATIONFREERECOGNIZER_H
#include "PlateInfo.h"
#include "opencv2/dnn.hpp"

namespace pr{
    std::pair<std::vector<int>,float> decodeResults(cv::Mat code_table);

    class segmentationFreeRecognizer{
    public:
        const int CHAR_INPUT_W = 14;
        const int CHAR_INPUT_H = 30;
        const int CHAR_LEN = 84;

        segmentationFreeRecognizer(std::string prototxt,std::string caffemodel);
        std::pair<std::string,float> SegmentationFreeForSinglePlate(cv::Mat plate,std::vector<std::string> mapping_table);
        std::pair<std::vector<int>,float> SegmentationFreeForSinglePlate(const cv::Mat &plate);


    private:
        cv::dnn::Net net;

    };

}
#endif //SWIFTPR_SEGMENTATIONFREERECOGNIZER_H
