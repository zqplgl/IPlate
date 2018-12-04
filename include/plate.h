//
// Created by zqp on 18-8-28.
//

#ifndef PROJECT_PLATE_H
#define PROJECT_PLATE_H

#include "IPlate.h"
#include "ICharsRecognize.h"
#include "plateDetection.h"
#include "fineTuning.h"
namespace Vehicle
{

    class Plate: public IPlate
    {
    public:
        Plate(const string& model_dir,const int gpu_id);
        virtual void detect(const cv::Mat &im,vector<PlateInfo>& plateinfos,const float confidence_threshold=0.5);
        ~Plate(){delete(detector_);delete(recognizer_);delete(fineTuning_);}

    private:
        std::pair<string,float> recognize(const cv::Mat& im, const PlateColorType type);
        std::pair<string,float> recognizeLayer2(const cv::Mat& im);

    private:
        PlateDetector *detector_ = nullptr;
        ICharsRecognize *recognizer_ = nullptr;
        FineTuning *fineTuning_ = nullptr;
    };
}

#endif //PROJECT_PLATE_H
