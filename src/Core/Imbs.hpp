/*
 *  IMBS Background Subtraction Library 
 *
 *  This file imbs.hpp contains the C++ OpenCV based implementation for
 *  IMBS algorithm described in
 *  D. D. Bloisi and L. Iocchi
 *  "Independent Multimodal Background Subtraction"
 *  In Proc. of the Third Int. Conf. on Computational Modeling of Objects
 *  Presented in Images: Fundamentals, Methods and Applications, pp. 39-44, 2012.
 *  Please, cite the above paper if you use IMBS.
 *  
 *  This software is provided without any warranty about its usability. 
 *  It is for educational purposes and should be regarded as such.
 *
 *  Written by Domenico D. Bloisi
 *
 *  Please, report suggestions/comments/bugs to
 *  domenico.bloisi@gmail.com
 *
 */

#ifndef __IMBS_HPP__
#define __IMBS_HPP__

//OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
//C++
#include <iostream>
#include <vector>
#include <fstream>

using namespace cv;
using namespace std;

class BackgroundSubtractorIMBS
{
public:
    //! the default constructor
    BackgroundSubtractorIMBS();
	//! fps constructor
	BackgroundSubtractorIMBS(double fps);
    //! the full constructor
    BackgroundSubtractorIMBS(
        double fps,
        unsigned char fgThreshold,
        unsigned char associationThreshold,
        unsigned int samplingPeriod,
        unsigned int minBinHeight,
        unsigned int numSamples,
        float alpha,
        float beta,
        unsigned char tau_s,
        unsigned char tau_h,
        unsigned int minArea,
        unsigned int persistencePeriod,
        bool morphologicalFiltering
    );
	
    //! the destructor
    ~BackgroundSubtractorIMBS();
    //! the update operator
    void apply(InputArray image, OutputArray fgmask, double learningRate=-1.);

    //! computes a background image which shows only the highest bin for each pixel
    void getBackgroundImage(OutputArray backgroundImage) const;

	void rgbSuppression();
	
    //! re-initiaization method
    void initialize(Size frameSize, int frameType);
	
	bool loadBg(const char* filename);
	void saveBg(string* filename);
	
	typedef struct _BgModel
	{
		_BgModel()
		{
			values = NULL;
			isValid = NULL;
			isFg = NULL;
			counter = NULL;
		}
		
		Vec3b* values;
		bool* isValid;
		bool* isFg;
		uchar* counter;
	} BgModel;

	//struct for modeling the background values for a single pixel
	typedef struct _Bins
	{
		_Bins()
		{
			binValues = NULL;
			binHeights = NULL;
			isFg = NULL;
		}
	
		Vec3b* binValues;
		uchar* binHeights;
		bool* isFg;
	} Bins;

	bool isBackgroundCreated;

private:
    //method for creating the background model
    void createBg(unsigned int bg_sample_number);
    //method for updating the background model
    void updateBg();
		//method for computing the foreground mask
    void getFg();
    //method for suppressing shadows and highlights
    void hsvSuppression();    
    //method for refining foreground mask
    void filterFg();
    //method for filtering out blobs smaller than a given area
    void areaThresholding();
    //method for getting the current time
    double getTimestamp();
    //method for converting from RGB to HSV
    Mat convertImageRGBtoHSV(const Mat& imageRGB);
    //method for changing the bg in case of sudden changes 
    void changeBg();
	
    //current input RGB frame
    Mat frame;
    vector<Mat> frameBGR;
    //frame size
    Size frameSize;
    //frame type
    int frameType;
    //total number of pixels in frame
    unsigned int numPixels;
    //current background sample
    Mat bgSample;
    vector<Mat> bgSampleBGR;
    //current background image which shows only the highest bin for each pixel
    //(just for displaying purposes)
    Mat bgImage;
    //current foreground mask
    Mat fgmask;

    Mat fgfiltered;
	
	string* bgFilename;
	bool loadedBg;
	
    //number of fps
    double fps;
    //time stamp in milliseconds (ms)
    double timestamp;
    //previous time stamp in milliseconds (ms)
    double prev_timestamp;
    double initial_tick_count;
	  //initial message to be shown until the first bg model is ready 
    Mat initialMsgGray;
    Mat initialMsgRGB;
	
	Bins* bgBins;
	
private:
	BgModel* bgModel;

	//SHADOW SUPPRESSION PARAMETERS
	float alpha;
	float beta;
	uchar tau_s;
	uchar tau_h;

	unsigned int minBinHeight;
	unsigned int numSamples;
	unsigned int samplingPeriod;
	unsigned long prev_bg_frame_time;
	unsigned int bg_frame_counter;
	unsigned int associationThreshold;
	unsigned int maxBgBins;
	unsigned int nframes;

	double minArea;
	bool bg_reset;
	unsigned int persistencePeriod;
	bool prev_area;
	bool sudden_change;
	unsigned int fgThreshold;
	uchar SHADOW_LABEL;
	uchar PERSISTENCE_LABEL;
	uchar FOREGROUND_LABEL;
    //persistence map
	unsigned int* persistenceMap;
    Mat persistenceImage;

    bool morphologicalFiltering;

public:
    unsigned int getMaxBgBins() {
    	return maxBgBins;
    }
    unsigned int getFgThreshold() {
        return fgThreshold;
    }
    void getBgModel(BgModel bgModel_copy[], int size);
};

#endif //__IMBS_HPP__
