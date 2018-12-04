#include "plateDetection.h"

namespace Vehicle{
    PlateDetector::PlateDetector(const std::string &cascade_file, const string &deploy_file, const string &weight_file,
                                 const int gpu_id)
    {
        vector<float> mean_values = {0.5,0.5,0.5};
        float normal_val = 0.007843;
        detector_ = ObjZoneDetect::CreateObjZoneSSDDetector(deploy_file,weight_file,mean_values,normal_val,gpu_id);
        cascade.load(cascade_file);
    }

    void PlateDetector::detect(const cv::Mat &im, std::vector<ObjZoneDetect::Object> &plate_zones, const int min_w, const int max_w)
    {
        plate_zones.clear();
        cv::Mat im_gray;
        cv::cvtColor(im,im_gray,cv::COLOR_BGR2GRAY);
        std::vector<cv::Rect> zones;
        cv::Size min_size(min_w,min_w/4);
        cv::Size max_size(max_w,max_w/4);
        cascade.detectMultiScale(im_gray, zones,1.1, 3, cv::CASCADE_SCALE_IMAGE,min_size,max_size);

        for(auto zone:zones)
        {
            int zeroadd_w  = static_cast<int>(zone.width*0.28);
            int zeroadd_h = static_cast<int>(zone.height*1.2);
            int zeroadd_x = static_cast<int>(zone.width*0.14);
            int zeroadd_y = static_cast<int>(zone.height*0.6);
            zone.x-=zeroadd_x;
            zone.y-=zeroadd_y;
            zone.height += zeroadd_h;
            zone.width += zeroadd_w;

            if(zone.x<0) zone.x=0;
            if(zone.y<0) zone.y=0;
            if(zone.width+zone.x>=im.cols) zone.width=im.cols-zone.x;
            if(zone.height+zone.y>im.rows) zone.height=im.rows - zone.y;

            ObjZoneDetect::Object object;
            object.zone = zone;
            object.cls = 1;
            object.score = 1.f;

            plate_zones.push_back(object);

            cout<<"opencv***************************"<<endl;
        }
    }

    void PlateDetector::detect(const cv::Mat &im, std::vector<ObjZoneDetect::Object> &plate_zones,
                               const float confidence_threshold)
    {
        plate_zones.clear();
        detector_->detect(im,plate_zones,confidence_threshold);

        for (int i=0; i<plate_zones.size(); ++i)
        {
            cv::Rect &zone = plate_zones[i].zone;
            zone.y = zone.y-5;
            zone.height = zone.height + 10;
            if (zone.y < 0)zone.y = 0;
            if (zone.y + zone.height > im.rows)zone.height = im.rows - zone.y;
        }
    }

    void PlateDetector::detectPlate(const cv::Mat &im, std::vector<ObjZoneDetect::Object> &plate_zones,
                                    const float confidence_threshold)
    {
        detect(im,plate_zones,confidence_threshold);
        if(plate_zones.empty())
            detect(im,plate_zones,36,700);
    }
}//namespace pr
