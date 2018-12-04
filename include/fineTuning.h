#ifndef SWIFTPR_FINETUNING_H
#define SWIFTPR_FINETUNING_H

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <string>

namespace Vehicle{
    class FineTuning
    {
    public:
        FineTuning(const std::string& deploy_file,const std::string& weight_file);
        cv::Mat fineTuningVertical(const cv::Mat& im, const int plate_class, const int slice_num=5,
        		const int upper=0,const int lower=-50,const int windows_size=17);//15->5
		cv::Mat fineTuningVertical2(const cv::Mat& im, const int slice_num = 15, const int upper = 0,
				const int lower = -50, const int window_size = 17);
		cv::Mat fineTuningHorizon(const cv::Mat& im,int pad_l,int pad_r);

    private:
        cv::dnn::Net net_;

    };
}
#endif //SWIFTPR_FINEMAPPING_H
