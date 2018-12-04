#ifndef SWIFTPR_PLATEDETECTION_H
#define SWIFTPR_PLATEDETECTION_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include "IObjZoneDetect.h"

using namespace std;

namespace Vehicle{
    enum PlateColorType
    {
        COLOR_PLATE_BLUE=1,
        COLOR_PLATE_NEW=2,
        COLOR_PLATE_BLACK=3,
        COLOR_PLATE_YELLOW2=4,
        COLOR_PLATE_YELLOW1=5,
        COLOR_PLATE_WHITE1=6,
        COLOR_PLATE_WHITE2=7
    };

    class PlateDetector{
    public:
        PlateDetector(const std::string& cascade_file,const string& deploy_file, const string& weight_file,const int gpu_id=0);
        void detectPlate(const cv::Mat& im,std::vector<ObjZoneDetect::Object>& plate_zones,const float confidence_threshold=0.5);

        ~PlateDetector(){delete(detector_);};
    private:
        void detect(const cv::Mat& im,std::vector<ObjZoneDetect::Object>& plate_zones,const int min_w=36,const int max_w=800);
        void detect(const cv::Mat& im,std::vector<ObjZoneDetect::Object>& plate_zones,const float confidence_threshold);

    private:
        cv::CascadeClassifier cascade;
        ObjZoneDetect::IObjZoneDetect *detector_= nullptr;
    };
}

#endif
