#include "fineTuning.h"
#include "plateDetection.h"
#include "util.h"

using namespace cv;
using namespace std;
using namespace util;
namespace Vehicle
{
	enum ImType
	{
		SINGLE,
		SINGLE_NOT,
		DOUBLE,
		DOUBLE_NOT
	} ;

    const int FINEMAPPING_H = 50;
    const int FINEMAPPING_W = 120;
    const int PADDING_UP_DOWN = 30;

	std::pair<int,int> fitLineRansac(const std::vector<cv::Point>& pts,const int zeroadd);

	cv::Mat plateWrap(const Mat& im_resized,const vector<cv::Point>& line_upper, const vector<cv::Point>& line_lower,
					  const enum PlateColorType plate_type)
	{
		cv::Mat im_border;
		cv::copyMakeBorder(im_resized, im_border, 30, 30, 0, 0, cv::BORDER_REPLICATE);

		std::pair<int, int> A;
		std::pair<int, int> B;
		A = fitLineRansac(line_upper, -2);
		B = fitLineRansac(line_lower, 2);
		int leftyB = A.first;
		int rightyB = A.second;
		int leftyA = B.first;
		int rightyA = B.second;
		int cols = im_border.cols;
		int rows = im_border.rows;

		std::vector<cv::Point2f> corners(4);//4
		if(plate_type==COLOR_PLATE_WHITE2 || plate_type==COLOR_PLATE_YELLOW2)
		{

		    int upcut = rightyA - rightyB;
			corners[0] = cv::Point2f(cols - 1, rightyA);
			corners[1] = cv::Point2f(0, leftyA);
			if (plate_type == COLOR_PLATE_WHITE2)
			{
				corners[2] = cv::Point2f(cols - 1, rightyB - upcut + 5);
				corners[3] = cv::Point2f(0, leftyB - upcut + 5);
			}
			if (plate_type == COLOR_PLATE_YELLOW2)
			{
				corners[2] = cv::Point2f(cols - 1, rightyB - upcut + 7);
				corners[3] = cv::Point2f(0, leftyB - upcut + 7);
			}
		}
		else
		{
			corners[0] = cv::Point2f(cols - 1, rightyA + 2);
			corners[1] = cv::Point2f(0, leftyA + 2);
			corners[2] = cv::Point2f(cols - 1, rightyB - 2);
			corners[3] = cv::Point2f(0, leftyB - 2);
		}

		std::vector<cv::Point2f> corners_trans(4);
		if(plate_type==COLOR_PLATE_WHITE2 || plate_type==COLOR_PLATE_YELLOW2)
		{
			corners_trans[0] = cv::Point2f(136, 60);//36
			corners_trans[1] = cv::Point2f(0, 60);//36
		}
		else
		{
			corners_trans[0] = cv::Point2f(136, 36);//36
			corners_trans[1] = cv::Point2f(0, 36);//36
		}

		corners_trans[2] = cv::Point2f(136, 0);
		corners_trans[3] = cv::Point2f(0, 0);

		cv::Mat transform = cv::getPerspectiveTransform(corners, corners_trans);
		cv::Mat quad;
		if(plate_type==COLOR_PLATE_WHITE2 || plate_type==COLOR_PLATE_YELLOW2)
            cv::warpPerspective(im_border, quad, transform, cv::Size(136,60));
		else
			cv::warpPerspective(im_border, quad, transform, cv::Size(136,36));

		return quad;
	}

	std::pair<int,int> fitLineRansac(const std::vector<cv::Point>& pts,const int zeroadd)
	{
		std::pair<int,int> res;
		if(pts.size()>2)
		{
			cv::Vec4f line;
			cv::fitLine(pts,line,CV_DIST_HUBER,0,0.01,0.01);
			float vx = line[0];
			float vy = line[1];
			float x = line[2];
			float y = line[3];
			int lefty = static_cast<int>(y - x * vy / vx );
			int righty = static_cast<int>((136- x) * vy / vx + y);
			res.first = lefty+PADDING_UP_DOWN+zeroadd;
			res.second = righty+PADDING_UP_DOWN+zeroadd;
			return res;
		}
		res.first = zeroadd;
		res.second = zeroadd;
		return res;
	}

	void processPlate(const Mat& im_gray,const int slice_num, const int lower,
							   const float diff,vector<cv::Point>& line_upper, vector<cv::Point>& line_lower,
							   int &contours_num, const int win, const float alpha,const int window_size,
							   const float ratio1, const int size1,const int size2,const float ratio2,
							   const int size3,const int size4, const enum ImType flag)
	{
		for (int i = 0; i < slice_num; i++)
		{
			std::vector<std::vector<cv::Point> > contours;
			float k = lower + i * diff;
			Mat im_threshold = cv::Mat::zeros(im_gray.size(), CV_8U);
			if(flag==SINGLE || flag==SINGLE_NOT)
                util::NiblackSauvolaWolfJolion(im_gray, im_threshold, SAUVOLA, win, win, alpha * k);
			util::clearLiuDingOnlyWhite(im_threshold);

			cv::findContours(im_threshold, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
			for (auto contour : contours)
			{
				cv::Rect bdbox = cv::boundingRect(contour);
				float ratio = bdbox.height / static_cast<float>(bdbox.width);
				int size = bdbox.width * bdbox.height;
#if flag==DOUBLE
				if ((ratio>ratio1 && size1<size && size<size2)
                    || (ratio>ratio2 && size3<size && size<size4&& 4<bdbox.x&& bdbox.x<(im_threshold.cols-4)))
				{
#else
				if ((ratio>ratio1 && size1<size && size<size2) || (ratio>ratio2 && size3<size && size>size4))
				{
#endif
					cv::Point p1(bdbox.x, bdbox.y);
					cv::Point p2(bdbox.x + bdbox.width, bdbox.y + bdbox.height);
					line_upper.push_back(p1);
					line_lower.push_back(p2);
					contours_num += 1;
				}
			}
		}
	}

    FineTuning::FineTuning(const std::string &deploy_file, const std::string &weight_file)
    {
		net_ = cv::dnn::readNetFromCaffe(deploy_file, weight_file);
    }

    cv::Mat FineTuning::fineTuningHorizon(const cv::Mat &im, int pad_l, int pad_r)
    {
		cv::Mat inputBlob = cv::dnn::blobFromImage(im, 1 / 255.0, cv::Size(66, 16),cv::Scalar(0,0,0),false);

        net_.setInput(inputBlob,"data");
        cv::Mat prob = net_.forward();
        int front = static_cast<int>(prob.at<float>(0,0)*im.cols);
        int back = static_cast<int>(prob.at<float>(0,1)*im.cols);
        front -= pad_l ;
        if(front<0) front = 0;
        back += pad_r;
        if(back>im.cols-1) back=im.cols - 1;
        cv::Mat cropped  = im.colRange(front,back).clone();
		return cropped;
    }



    cv::Mat FineTuning::fineTuningVertical(const cv::Mat &im, const int plate_class, const int slice_num,
											 const int upper, const int lower, const int win_size)
	{
        cv::Mat im_resized;
        cv::Mat im_gray;
        cv::resize(im, im_resized, cv::Size(FINEMAPPING_W,FINEMAPPING_H));
        if(im_resized.channels()==3)
        	cv::cvtColor(im_resized, im_gray, cv::COLOR_BGR2GRAY);
		else
			im_resized.copyTo(im_gray);

		//new yellow white
		//1.blue	2.new	3.black		4.yellow2	5.yellow1	6.white1	7.white2
		if(plate_class==2 || plate_class==4 || plate_class==5 || plate_class==6 || plate_class==7)
			cv::bitwise_not(im_gray,im_gray);

		float diff = static_cast<float>(upper - lower)/static_cast<float>(slice_num - 1);
		std::vector<cv::Point> line_upper;
		std::vector<cv::Point> line_lower;
		int contours_nums = 0;

		cv::Mat quad;
		if(plate_class==1 || plate_class==2 || plate_class==3 || plate_class==5 || plate_class==6)
		{
		    processPlate(im_gray,slice_num,lower,diff,line_upper,line_lower,contours_nums,25,0.0001f,0,0.6f,50,500,3.0f,10,110,SINGLE);

            if (contours_nums<50)
            {
                cv::Mat im_not;
                cv::bitwise_not(im_resized, im_not);
                cv::Mat kernal = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(1, 5));
                cv::erode(im_not, im_not, kernal);//erode
                if (im.channels() == 3)cv::cvtColor(im_not, im_gray, cv::COLOR_BGR2GRAY);
                else im_gray = im_not;

                processPlate(im_gray,slice_num,lower,diff,line_upper,line_lower,contours_nums,15,0.001f,0,0.7f,120,400,3.0f,10,100,SINGLE_NOT);
            }
            quad = plateWrap(im_resized,line_upper,line_lower,PlateColorType(plate_class));
        }
        else if(plate_class==4 || plate_class==7)
		{
            processPlate(im_gray,slice_num,lower,diff,line_upper,line_lower,contours_nums,0,0.0f,win_size,0.7f,120,400,3.0f,10,110,DOUBLE);
            if(contours_nums<20)
			{
				cv::Mat im_not;
				cv::bitwise_not(im_resized, im_not);
				cv::Mat kernal = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(1, 3));
				cv::erode(im_not, im_not, kernal);//erode
                if (im.channels() == 3)cv::cvtColor(im_not, im_gray, cv::COLOR_BGR2GRAY);
                else im_gray = im_not;

				processPlate(im_gray,slice_num,lower,diff,line_upper,line_lower,contours_nums,0,0.0f,win_size,0.5f,80,500,3.0f,10,120,DOUBLE_NOT);
            }
			quad = plateWrap(im_resized,line_upper,line_lower,PlateColorType(plate_class));
		}

		return quad;
	}

	cv::Mat FineTuning::fineTuningVertical2(const cv::Mat &im, const int slice_num, const int upper, const int lower,
											  const int window_size)
	{
        cv::Mat im_resized;
		cv::Mat im_gray;
		cv::resize(im, im_resized, cv::Size(FINEMAPPING_W,FINEMAPPING_H));
		if(im_resized.channels()==3)
			cv::cvtColor(im_resized, im_gray, cv::COLOR_BGR2GRAY);
		else
			im_resized.copyTo(im_gray);

		//1.blue	2.new	3.black		4.yellow2	5.yellow1	6.white1	7.white2
        cv::bitwise_not(im_gray,im_gray);

		float diff = static_cast<float>(upper - lower)/static_cast<float>(slice_num - 1);
		std::vector<cv::Point> line_upper;
		std::vector<cv::Point> line_lower;
		int contours_nums = 0;

		processPlate(im_gray,slice_num,lower,diff,line_upper,line_lower,contours_nums,0,0.f,window_size,0.7,120,300,3.f,10,100,DOUBLE);
		if(contours_nums<41)
		{
			cv::Mat im_not;
			cv::bitwise_not(im_resized, im_not);
			cv::Mat kernal = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(1, 5));
			cv::erode(im_not, im_not, kernal);//erode
			if (im.channels() == 3)cv::cvtColor(im_not, im_gray, cv::COLOR_BGR2GRAY);
			else im_gray = im_not;

			processPlate(im_gray,slice_num,lower,diff,line_upper,line_lower,contours_nums,0,0.f,window_size,0.7,120,300,3.f,10,100,DOUBLE);
		}

		cv::Mat quad;
        quad = plateWrap(im_resized,line_upper,line_lower,COLOR_PLATE_BLUE);

        return quad;
	}
}


