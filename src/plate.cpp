//
// Created by zqp on 18-8-28.
//
#include "plate.h"
#include "util.h"

namespace Vehicle
{
    using namespace cv;
    Plate::Plate(const std::string &model_dir, const int gpu_id)
    {
        string temp_dir = model_dir;
        if(temp_dir[temp_dir.size()-1]!='/')
            temp_dir += "/plate/";

        string deploy_file = temp_dir + "detector/deploy.prototxt";
        string weights_file = temp_dir + "detector/MobileNetSSD_deploy_iter_48000.caffemodel";
        string cascade_file = temp_dir +"detector/cascade.xml";

        detector_ = new PlateDetector(cascade_file,deploy_file,weights_file,gpu_id);

        deploy_file = temp_dir+"recogniser/deploy.prototxt";
        weights_file = temp_dir+"recogniser/weights.caffemodel";
        string label_file = temp_dir+"recogniser/labels.txt";

        vector<float> mean_value = {152,152,152};
        recognizer_ = CreateICharsRecognize(deploy_file,weights_file,label_file,mean_value);


        deploy_file = temp_dir + "finetuning/deploy.prototxt";
        weights_file = temp_dir + "finetuning/weights.caffemodel";
        fineTuning_ = new FineTuning(deploy_file,weights_file);
    }

    std::pair<string,float> Plate::recognizeLayer2(const cv::Mat &im)
    {
        cv::Mat im_resized;
        cv::resize(im, im_resized, cv::Size(140, 60));
        cv::Mat im_gray;
        if (im.channels() == 3)
            cv::cvtColor(im_resized, im_gray, cv::COLOR_BGR2GRAY);
        else
            im_resized.copyTo(im_gray);
        cv::bitwise_not(im_gray,im_gray);

        //cut the double layer
        int k = 0.01, win = 22;
        Mat im_binary = cv::Mat::zeros(im_gray.size(), CV_8U);
        util::NiblackSauvolaWolfJolion(im_gray, im_binary, SAUVOLA, win, win, 0.18 * k);
        int cut_position = util::cut2LayerPlatePositionYellow2(im_binary);
        if (cut_position > 35 || cut_position < 15)cut_position = 23;

        Rect up_zone(0,2,im_resized.cols,cut_position);
        cv::Mat up_im(im_resized, up_zone);
        Rect down_zone(0,cut_position,im_resized.cols,im_resized.rows-cut_position);
        cv::Mat down_im(im_resized, down_zone);

        std::pair<string,float> up_result = recognizer_->recognize(up_im);
        std::pair<string,float> down_result = recognizer_->recognize(down_im);

        std::pair<string,float> result;
        result.first = up_result.first + down_result.first;
        result.second = (up_result.second + down_result.second)/2;

        return result;
    }

    std::pair<string,float> Plate::recognize(const cv::Mat &im,const PlateColorType type)
    {
        if(type!=COLOR_PLATE_WHITE2 && type!=COLOR_PLATE_YELLOW2)
        {
            std::pair<string,float> result = recognizer_->recognize(im);
            return result;
        }
        else
        {
            return recognizeLayer2(im);
        }
    }

    void Plate::detect(const cv::Mat &im, std::vector<Vehicle::PlateInfo> &plateinfos,const float confidence_threshold)
    {
        plateinfos.clear();
        vector<ObjZoneDetect::Object> plate_zones;
        detector_->detectPlate(im,plate_zones,confidence_threshold);
        if(plate_zones.empty())
            return;

        ObjZoneDetect::Object &zone = plate_zones[0];
        Vehicle::PlateInfo plateinfo;
        plateinfo.zone = zone.zone;

        cv::Mat plate = im(zone.zone).clone();
        cv::Mat plate_finetuned = fineTuning_->fineTuningVertical(plate,zone.cls);
#if 1
        imshow("im_plate",plate);
        imshow("im_finetuned",plate_finetuned);
//            cv::waitKey(0);
#endif
        std::pair<string,float> result = recognize(plate_finetuned,PlateColorType(zone.cls));
        plateinfo.license = result.first;
        plateinfo.score = result.second;

        switch(PlateColorType(zone.cls))
        {
            case COLOR_PLATE_BLUE:
                plateinfo.color = "blue";
                break;
            case COLOR_PLATE_YELLOW2:
            case COLOR_PLATE_YELLOW1:
                plateinfo.color = "yellow";
                break;
            case COLOR_PLATE_NEW:
                plateinfo.color = "green";
                break;
            case COLOR_PLATE_BLACK:
                plateinfo.color = "black";
                break;
            case COLOR_PLATE_WHITE2:
            case COLOR_PLATE_WHITE1:
                plateinfo.color = "white";
                break;
            default:
                plateinfo.color = "unknown";
        }
        plateinfos.push_back(plateinfo);
    }

    IPlate *CreateIPlateRecognize(const string& model_dir,const int gpu_id)
    {
        return new Plate(model_dir,gpu_id);
    }
}
