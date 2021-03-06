#pragma once

#include "VisualReading.h"
#include <opencv2/core/core.hpp>

class ObservationManager
{
	private:
		std::vector<cv::Rect> filterBoundingBoxVertical(cv::Mat&,cv::Rect&,int,float);
		
	public:
		ObservationManager();
		
		VisualReading process(const cv::Mat&,const cv::Mat&,int,int);
};
