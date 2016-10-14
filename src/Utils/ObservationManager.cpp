#include "ObservationManager.h"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

ObservationManager::ObservationManager() {;}

vector<Rect> ObservationManager::filterBoundingBoxVertical(Mat& image, Rect& boundingBox, int split, float whitePercentage)
{
	vector<Rect> boundingBoxes;
	int area, counter, height, i, minY, maxY, startBoundingBoxX, step, width;
	
	area = ((boundingBox.br().x - boundingBox.tl().x) * (boundingBox.br().y - boundingBox.tl().y)) / split;
	counter = 0;
	i = 0;
	minY = boundingBox.tl().y ;
	maxY = boundingBox.br().y;
	width = boundingBox.br().x - boundingBox.tl().x;
	height = boundingBox.br().y - boundingBox.tl().y;
	step = width / split;
	
	startBoundingBoxX = boundingBox.tl().x;
	
	while (i < split)
	{
		int minX, maxX;
		
		minX = boundingBox.tl().x + (i * step);
		maxX = minX + step;
		
		counter = 0;
		
		for (int j = minY; j < maxY; ++j)
		{
			for (int k = minX; k < maxX; ++k)
			{
				if ((uchar) image.at<uchar>(j,k) == 255) ++counter;
			}
		}
		
		if ((counter / (float) area) < whitePercentage)
		{
			if ((minX - startBoundingBoxX) > 0)
			{
				Rect newBoundingBox;
				
				newBoundingBox.x = startBoundingBoxX;
				newBoundingBox.y = minY;
				newBoundingBox.width = minX - startBoundingBoxX;
				newBoundingBox.height = height;
				
				boundingBoxes.push_back(newBoundingBox);
			}
			
			startBoundingBoxX = maxX;
		}
		else
		{
			if (i == (split - 1))
			{
				Rect newBoundingBox;
				
				newBoundingBox.x = startBoundingBoxX;
				newBoundingBox.y = minY;
				newBoundingBox.width = maxX - startBoundingBoxX;
				newBoundingBox.height = height;
				
				boundingBoxes.push_back(newBoundingBox);
			}
		}
		
		++i;
	}
	
	for (vector<Rect>::iterator it = boundingBoxes.begin(); it != boundingBoxes.end(); )
	{
		for (int i = it->y; i < (it->y + it->height); ++i)
		{
			for (int j = it->x; j < (it->x + it->width); ++j)
			{
				if ((uchar) image.at<uchar>(i,j) == 255) ++counter;
			}
		}
		
		if ((counter / ((float) (it->width * it->height))) > 0.4) ++it;
		else it = boundingBoxes.erase(it);
	}
	
	return boundingBoxes;
}

VisualReading ObservationManager::process(const Mat& frame, const Mat& fgMask, int minArea, int maxArea)
{
	/// Scale factor to increase the bounding box. Setting it to 1 will keep the bounding box at the original size.
	float scaleFactor = 1;
	
	// Filtering out shadows.
	Mat tmpBinaryImage = fgMask.clone();
	
	for (int i = 0; i < tmpBinaryImage.rows; ++i)
	{
		for (int j = 0; j < tmpBinaryImage.cols; ++j)
		{
			if (tmpBinaryImage.at<uchar>(i,j) <= 80)
			{
				tmpBinaryImage.at<uchar>(i,j) = 0;
			}
		}
	}
	
	vector<VisualReading::Observation> obs;
	vector<vector<Point> > contours;
	VisualReading visualReading;
	Mat element3(3,3,CV_8U,Scalar(1));
	
	morphologyEx(tmpBinaryImage,tmpBinaryImage,MORPH_CLOSE,element3);
	dilate(tmpBinaryImage,tmpBinaryImage,element3,Point(-1,-1),5);
	findContours(tmpBinaryImage,contours,CV_RETR_LIST,CV_CHAIN_APPROX_NONE);
	
	vector<vector<Point> > contoursPoly(contours.size());
	vector<Rect> boundRect(contours.size());
	
	tmpBinaryImage = Scalar(0);
	
	for (size_t contourIdx = 0; contourIdx < contours.size(); ++contourIdx)
    {
		Moments moms = moments(Mat(contours[contourIdx]));
		float area = moms.m00;
		
        if ((area < minArea) || (area >= maxArea)) continue;
		else
		{
			approxPolyDP(Mat(contours[contourIdx]),contoursPoly[contourIdx],3,true);
			boundRect[contourIdx] = boundingRect(Mat(contoursPoly[contourIdx]));
			drawContours(tmpBinaryImage,contours,contourIdx,Scalar(255),CV_FILLED);
			
			const vector<Rect>& boundingBoxes = filterBoundingBoxVertical(tmpBinaryImage,boundRect[contourIdx],8,0.1);
			
			for (vector<Rect>::const_iterator it = boundingBoxes.begin(); it != boundingBoxes.end(); ++it)
			{
				VisualReading::Observation observation;
				int counter;
				uchar hue, saturation, value;
				
				boundRect[contourIdx] = *it;
				counter = 0;
				
				const Mat& roi = frame(boundRect[contourIdx]);
				
				Mat hsvRoi;
				
				cvtColor(roi,hsvRoi,CV_BGR2HSV);
				
				for (int y = 0; y < hsvRoi.rows; ++y)
				{
					for (int x = 0; x < hsvRoi.cols; ++x)
					{
						hue = (uchar) hsvRoi.data[y * hsvRoi.step + x];
						saturation = (uchar) hsvRoi.data[y * hsvRoi.step + x + 1];
						value = (uchar) hsvRoi.data[y * hsvRoi.step + x + 2];
						
						observation.model.histograms[0][hue]++;
						observation.model.histograms[1][saturation]++;
						observation.model.histograms[2][value]++;
						
						++counter;
					}
				}
				
				for (int i = 0; i < VisualReading::Model::HISTOGRAM_VECTOR_LENGTH; ++i)
				{
					observation.model.histograms[0][i] /= counter;
					observation.model.histograms[1][i] /= counter;
					observation.model.histograms[2][i] /= counter;
				}
				
				float imageX, imageHeadX, imageY, imageHeadY;
				float barycenter, barycenterHead;
				int whiteBins[roi.cols];
				int endX, endY, index, whiteCounter;
				
				endX = boundRect[contourIdx].tl().x + roi.cols;
				endY = boundRect[contourIdx].tl().y + roi.rows;
				
				for (int i = 0; i < roi.cols; ++i)
				{
					whiteBins[i] = 0;
				}
				
				index = 0;
				whiteCounter = 0;
				
				/// Calculating center of the feet.
				for (int x = boundRect[contourIdx].tl().x; x < endX; ++x, ++index)
				{
					for (int y = (boundRect[contourIdx].tl().y + (roi.rows * 0.8)); y < endY; ++y)
					{
						if (((int) tmpBinaryImage.at<uchar>(y,x)) == 255)
						{
							whiteBins[index] += 1;
							++whiteCounter;
						}
					}
				}
				
				if (whiteCounter == 0) continue;
				
				barycenter = 0.0;
				
				for (int i = 0; i < roi.cols; ++i)
				{
					barycenter += (whiteBins[i] * i);
				}
				
				barycenter /= whiteCounter;
				
				if (barycenter == 0.0) continue;
				
				barycenter += boundRect[contourIdx].tl().x;
				
				imageX = barycenter;
				imageY = (int) boundRect[contourIdx].br().y;
				/// This instruction increases the bounding box by a factor equals to scaleFactor.
				imageY += (scaleFactor - 1) * (boundRect[contourIdx].height / 2);
				
				circle(tmpBinaryImage,Point(imageX,imageY),6,cvScalar(180),2);
				
				for (int i = 0; i < roi.cols; ++i)
				{
					whiteBins[i] = 0;
				}
				
				index = 0;
				whiteCounter = 0;
				
				/// Calculating center of the head.
				for (int x = boundRect[contourIdx].tl().x; x < endX; ++x, ++index)
				{
					for (int y = boundRect[contourIdx].tl().y; y < (endY - (roi.rows * 0.8)); ++y)
					{
						if (((int) tmpBinaryImage.at<uchar>(y,x)) == 255)
						{
							whiteBins[index] += 1;
							++whiteCounter;
						}
					}
				}
				
				if (whiteCounter == 0) continue;
				
				barycenterHead = 0.0;
				
				for (int i = 0; i < roi.cols; ++i)
				{
					barycenterHead += (whiteBins[i] * i);
				}
				
				barycenterHead /= whiteCounter;
				
				if (barycenterHead == 0.0) continue;
				
				barycenterHead += boundRect[contourIdx].tl().x;
				
				imageHeadX = barycenterHead;
				imageHeadY = (int) boundRect[contourIdx].tl().y + (((int) boundRect[contourIdx].br().y - (int) boundRect[contourIdx].tl().y) / 2);
				
				circle(tmpBinaryImage,Point(imageHeadX,imageHeadY),6,cvScalar(0),2);
				
				/// These instruction increases the bounding box by a factor equals to scaleFactor.
				boundRect[contourIdx].x -= ((scaleFactor - 1) * (boundRect[contourIdx].width / 2));
				boundRect[contourIdx].y -= ((scaleFactor - 1) * (boundRect[contourIdx].height / 2));
				boundRect[contourIdx].width *= scaleFactor;
				boundRect[contourIdx].height *= scaleFactor;
				
				rectangle(tmpBinaryImage,boundRect[contourIdx].tl(),boundRect[contourIdx].br(),CV_RGB(255,255,255),1,8,0);
				
				observation.observation.x = imageX;
				observation.observation.y = imageY;
				observation.head.x = imageHeadX;
				observation.head.y = imageHeadY;
				observation.model.barycenter = barycenter;
				observation.model.boundingBox = make_pair(Point2f(boundRect[contourIdx].tl().x - barycenter,-roi.rows),Point2f(roi.cols + boundRect[contourIdx].tl().x - barycenter,0));
				observation.model.height = (int) boundRect[contourIdx].br().y - (int) boundRect[contourIdx].tl().y;
				observation.model.width = (int) boundRect[contourIdx].br().x - (int) boundRect[contourIdx].tl().x;
				
				obs.push_back(observation);
			}
		}
	}
	
	//imshow("Foreground (with rectangles)",tmpBinaryImage);
	
	visualReading.setObservations(obs);
	visualReading.setObservationsAgentPose(Point2f(0.0,0.0));
	
	return visualReading;
}
