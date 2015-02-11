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

#include "Imbs.hpp"

using namespace std;
using namespace cv;

// default parameters of IMBS algorithm
static const float defaultFps = 25.0;
static const unsigned char defaultFgThreshold = 15;
static const unsigned char defaultAssociationThreshold = 5;
static const unsigned int defaultSamplingPeriod = 500;
static const unsigned int defaultMinBinHeight = 2;
static const unsigned int defaultNumSamples = 30;
static const float defaultAlpha = 0.65f;
static const float defaultBeta = 1.15f;
static const unsigned char defaultTau_s = 60;
static const unsigned char defaultTau_h = 40;
static const unsigned int defaultMinArea = 30;
static const unsigned int defaultPersistencePeriod = 10000;
static const bool defaultMorphologicalFiltering = true;

class Parallel_getFg: public cv::ParallelLoopBody
{   
private:
	Mat fgmask;
	BackgroundSubtractorIMBS::BgModel* bgModel;
	int maxBgBins;
	Mat frameB;
	Mat frameG;
	Mat frameR;
	int fgThreshold;
	unsigned int* persistenceMap;
	unsigned char PERSISTENCE_LABEL;
	const unsigned char FOREGROUND_LABEL;
	
	float timestamp;
	float prev_timestamp;
	unsigned int persistencePeriod;
	
public:
	Parallel_getFg(Mat& _fgmask, Mat& _frameB, Mat& _frameG, Mat& _frameR,
				   unsigned int* _persistenceMap,
				   BackgroundSubtractorIMBS::BgModel* bufferToProcess,
				   const int _maxBgBins, const int _fgThreshold,
				   const unsigned char _PERSISTENCE_LABEL, const unsigned char _FOREGROUND_LABEL,
				   const float _timestamp, const float _prev_timestamp, const unsigned int _persistencePeriod)
		: fgmask(_fgmask), bgModel(bufferToProcess), maxBgBins(_maxBgBins), frameB(_frameB), frameG(_frameG), frameR(_frameR), fgThreshold(_fgThreshold),
		  persistenceMap(_persistenceMap), PERSISTENCE_LABEL(_PERSISTENCE_LABEL), FOREGROUND_LABEL(_FOREGROUND_LABEL),
		  timestamp(_timestamp), prev_timestamp(_prev_timestamp), persistencePeriod(_persistencePeriod) {}
	
	virtual void operator()( const cv::Range &r ) const {
		//register BackgroundSubtractorIMBS::BgModel *inputOutputBufferPTR=bgModel+r.start;
		//for (register int p = r.start; p != r.end; ++p, ++inputOutputBufferPTR)
		for (register int p = r.start; p != r.end; ++p)
		{
			bool isFg = true;
			bool conditionalUpdated = false;
			int d = 0;
			for(int n = 0; n < maxBgBins; ++n) {
				//if(!((*inputOutputBufferPTR).isValid[n])) {
				if(!(bgModel[p].isValid[n])) {
					if(n == 0) {
						isFg = false;
					}
					break;
				}
				else { //the model is valid
					d = std::max(
								//(int)std::abs((*inputOutputBufferPTR).values[n][0] - frameBGR[0].data[p]),
								(int)std::abs(bgModel[p].values[n][0] - frameB.data[p]),
							//std::abs((*inputOutputBufferPTR).values[n][1] - frameBGR[1].data[p])
							std::abs(bgModel[p].values[n][1] - frameG.data[p])
							);
					d = std::max(
								//(int)d, std::abs((*inputOutputBufferPTR).values[n][2] - frameBGR[2].data[p])
								(int)d, std::abs(bgModel[p].values[n][2] - frameR.data[p])
							);
					if(d < fgThreshold){
						//check if it is a potential background pixel
						//from stationary object
						//if((*inputOutputBufferPTR).isFg[n]) {
						if(bgModel[p].isFg[n]) {
							conditionalUpdated = true;
							break;
						}
						else {
							isFg = false;
							persistenceMap[p] = 0;
						}
					}
				} //end else (the model is valid)
			} //end for
			if(isFg) {
				if(conditionalUpdated) {
					fgmask.data[p] = PERSISTENCE_LABEL;
					persistenceMap[p] += (timestamp - prev_timestamp);
					if(persistenceMap[p] > persistencePeriod) {
						for(int n = 0; n < maxBgBins; ++n) {
							if(!bgModel[p].isValid[n]) {
								break;
							}
							bgModel[p].isFg[n] = false;
						}
					}
				}
				else {
					fgmask.data[p] = FOREGROUND_LABEL;
					persistenceMap[p] = 0;
				}
			} //end if isFg
		}//end for
	}//end function
};

class Parallel_createBg: public cv::ParallelLoopBody
{   
private:
	Mat fgmask;
	Mat bgSampleB;
	Mat bgSampleG;
	Mat bgSampleR;  
	unsigned int bg_sample_number;
	unsigned int numSamples;
	BackgroundSubtractorIMBS::BgModel* bgModel;
	BackgroundSubtractorIMBS::Bins* bgBins;
	unsigned int minBinHeight;
	unsigned int maxBgBins;
	unsigned int fgThreshold;
	
	unsigned char PERSISTENCE_LABEL;
	unsigned char FOREGROUND_LABEL;  
	unsigned char associationThreshold;  
	
public:
	Parallel_createBg(Mat& _fgmask, Mat& _bgSampleB, Mat& _bgSampleG, Mat& _bgSampleR,
					  const unsigned int _bg_sample_number,
					  const unsigned int _numSamples,
					  BackgroundSubtractorIMBS::BgModel* _bgModel,
					  BackgroundSubtractorIMBS::Bins* _bgBins,
					  const unsigned int _minBinHeight,
					  const unsigned int _maxBgBins, const unsigned int _fgThreshold,
					  const unsigned char _PERSISTENCE_LABEL, const unsigned char _FOREGROUND_LABEL,
					  const unsigned char _associationThreshold )
	{
		fgmask = _fgmask;
		bgSampleB = _bgSampleB;
		bgSampleG = _bgSampleG;
		bgSampleR = _bgSampleR;
		bg_sample_number = _bg_sample_number;
		numSamples = _numSamples;
		bgModel = _bgModel;
		bgBins = _bgBins;
		minBinHeight = _minBinHeight;
		maxBgBins = _maxBgBins;
		fgThreshold = _fgThreshold;
		PERSISTENCE_LABEL = _PERSISTENCE_LABEL;
		FOREGROUND_LABEL = _FOREGROUND_LABEL;
		associationThreshold = _associationThreshold;
	}
	
	
	
	virtual void operator()( const cv::Range &r ) const
	{
		/* create a statistical model for each pixel (a set of bins of variable size) */
		for (register int p = r.start; p != r.end; ++p)
		{
			/* create an initial bin for each pixel from the first sample (bg_sample_number = 0) */
			if (bg_sample_number == 0) {
				bgBins[p].binValues[0][0] = bgSampleB.data[p];
				bgBins[p].binValues[0][1] = bgSampleG.data[p];
				bgBins[p].binValues[0][2] = bgSampleR.data[p];
				
				bgBins[p].binHeights[0] = 1;
				
				for(unsigned int s = 1; s < numSamples; ++s)  {
					bgBins[p].binHeights[s] = 0;
				}
				
				/* if the sample pixel is from foreground keep track of that situation */
				if(fgmask.data[p] == FOREGROUND_LABEL) {
					bgBins[p].isFg[0] = true;
				}
				else {
					bgBins[p].isFg[0] = false;
				}
				
				
			}
			else {
				Vec3b currentPixel;
				currentPixel[0] = bgSampleB.data[p];
				currentPixel[1] = bgSampleG.data[p];
				currentPixel[2] = bgSampleR.data[p];
				
				int den = 0;
				
				for (unsigned int s = 0; s < bg_sample_number; ++s) {
					/* try to associate the current pixel values to an existing bin */
					if (std::abs(currentPixel[2] - bgBins[p].binValues[s][2]) <= associationThreshold &&
							std::abs(currentPixel[1] - bgBins[p].binValues[s][1]) <= associationThreshold &&
							std::abs(currentPixel[0] - bgBins[p].binValues[s][0]) <= associationThreshold )
					{
						den = (bgBins[p].binHeights[s] + 1);
						for (int k = 0; k < 3; ++k) {
							bgBins[p].binValues[s][k] =
									(bgBins[p].binValues[s][k] * bgBins[p].binHeights[s] + currentPixel[k]) / den;
						}
						
						bgBins[p].binHeights[s]++; //increment the height of the bin
						if (fgmask.data[p] == FOREGROUND_LABEL) {
							bgBins[p].isFg[s] = true;
						}
						break;
					}
					//if the association is not possible, create a new bin
					else if (bgBins[p].binHeights[s] == 0) {
						bgBins[p].binValues[s] = currentPixel;
						bgBins[p].binHeights[s]++;
						if(fgmask.data[p] == FOREGROUND_LABEL) {
							bgBins[p].isFg[s] = true;
						}
						else {
							bgBins[p].isFg[s] = false;
						}
						break;
					}
					else
						continue;
				}//for(unsigned int s = 0; s <= bg_sample_number; ++s)
				
				//if all samples have been processed
				//it is time to compute the fg mask
				if (bg_sample_number == (numSamples - 1)) {
					unsigned int index = 0;
					int max_height = -1;
					
					for(unsigned int s = 0; s < numSamples; ++s) {
						if (bgBins[p].binHeights[s] == 0 && index < maxBgBins) {
							bgModel[p].isValid[index] = false;
							break;
						}
						
						if(index == maxBgBins) {
							break;
						}
						else if(bgBins[p].binHeights[s] >= (int) minBinHeight) {
							if(fgmask.data[p] == PERSISTENCE_LABEL) {
								for(unsigned int n = 0; n < maxBgBins; n++) {
									if(!bgModel[p].isValid[n]) {
										break;
									}
									
									unsigned int d =
											std::max((int)std::abs(bgModel[p].values[n][0] - bgBins[p].binValues[s][0]),
											std::abs(bgModel[p].values[n][1] - bgBins[p].binValues[s][1]) );
									
									d = std::max((int)d, std::abs(bgModel[p].values[n][2] - bgBins[p].binValues[s][2]) );
									
									if(d < fgThreshold) {
										bgModel[p].isFg[n] = false;
										bgBins[p].isFg[s] = false;
									}
								}//for maxbgbins
							}//if persistence label
							
							if(bgBins[p].binHeights[s] > max_height) {
								max_height = bgBins[p].binHeights[s];
								
								for (int k = 0; k < 3; ++k) {
									bgModel[p].values[index][k] = bgModel[p].values[0][k];
									bgModel[p].values[0][k] = bgBins[p].binValues[s][k];
								}
								
								bgModel[p].isValid[index] = true;
								bgModel[p].isFg[index] = bgModel[p].isFg[0];
								bgModel[p].counter[index] = bgModel[p].counter[0];
								
								bgModel[p].isValid[0] = true;
								bgModel[p].isFg[0] = bgBins[p].isFg[s];
								bgModel[p].counter[0] = bgBins[p].binHeights[s];
							}
							else {
								for(int k = 0; k < 3; ++k) {
									bgModel[p].values[index][k] = bgBins[p].binValues[s][k];
								}
								
								bgModel[p].isValid[index] = true;
								bgModel[p].isFg[index] = bgBins[p].isFg[s];
								bgModel[p].counter[index] = bgBins[p].binHeights[s];
							}
							
							++index;
						}
					} //for all numSamples
				}//bg_sample_number == (numSamples - 1)
			}//else --> if(frame_number == 0)
		}//numPixels
	}
};

BackgroundSubtractorIMBS::BackgroundSubtractorIMBS() :
	numPixels(0),
	bgFilename(0),
	loadedBg(false),
	bgBins(NULL),
	bgModel(NULL),
	nframes(0),
	persistenceMap(NULL)
{
	fps = defaultFps;
	fgThreshold = defaultFgThreshold;
	associationThreshold = defaultAssociationThreshold;
	samplingPeriod = defaultSamplingPeriod;
	minBinHeight = defaultMinBinHeight;
	numSamples = defaultNumSamples;
	alpha = defaultAlpha;
	beta = defaultBeta;
	tau_s = defaultTau_s;
	tau_h = defaultTau_h;
	minArea = defaultMinArea;
	persistencePeriod = defaultPersistencePeriod;
	morphologicalFiltering = defaultMorphologicalFiltering;
	
	initial_tick_count = (float)getTickCount();
	
}

BackgroundSubtractorIMBS::BackgroundSubtractorIMBS(float fps) :
	numPixels(0),
	bgFilename(0),
	loadedBg(false),
	bgBins(NULL),
	bgModel(NULL),
	nframes(0),
	persistenceMap(NULL)
{
	this->fps = fps;
	fgThreshold = defaultFgThreshold;
	associationThreshold = defaultAssociationThreshold;
	samplingPeriod = defaultSamplingPeriod;
	minBinHeight = defaultMinBinHeight;
	numSamples = defaultNumSamples;
	alpha = defaultAlpha;
	beta = defaultBeta;
	tau_s = defaultTau_s;
	tau_h = defaultTau_h;
	minArea = defaultMinArea;
	persistencePeriod = defaultPersistencePeriod;
	morphologicalFiltering = defaultMorphologicalFiltering;
	
	initial_tick_count = (float)getTickCount();
	
}

BackgroundSubtractorIMBS::BackgroundSubtractorIMBS(
		float fps,
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
		bool morphologicalFiltering) : 
	numPixels(0),
	bgFilename(0),
	bgBins(NULL),
	bgModel(NULL),  
	nframes(0),
	persistenceMap(NULL)
{
	this->nframes = 0;
	this->numPixels = 0;
	this->fps = fps;
	this->fgThreshold = fgThreshold;
	this->persistencePeriod = persistencePeriod;
	
	if (minBinHeight <= 1){
		this->minBinHeight = 1;
	}
	else {
		this->minBinHeight = minBinHeight;
	}
	
	this->associationThreshold = associationThreshold;
	this->samplingPeriod = samplingPeriod;//ms
	this->minBinHeight = minBinHeight;
	this->numSamples = numSamples;
	this->alpha = alpha;
	this->beta = beta;
	this->tau_s = tau_s;
	this->tau_h = tau_h;
	this->minArea = minArea;
	
	if (fps == 0.) {
		initial_tick_count = (float)getTickCount();
	} else {
		initial_tick_count = 0;
	}
	
	this->morphologicalFiltering = morphologicalFiltering;  
}

BackgroundSubtractorIMBS::~BackgroundSubtractorIMBS()
{
	delete[] bgBins;
	delete[] bgModel;
	delete[] persistenceMap;
	delete bgFilename;
}

void BackgroundSubtractorIMBS::initialize(Size frameSize, int frameType)
{
	if (loadedBg)
		return;
	
	this->frameSize = frameSize;
	this->frameType = frameType;
	this->numPixels = frameSize.width*frameSize.height;
	
	delete[] bgBins;
	delete[] bgModel;
	delete[] persistenceMap;
	
	persistenceMap = new unsigned int[numPixels];
	
	for(unsigned int i = 0; i < numPixels; i++) {
		persistenceMap[i] = 0;
	}
	
	bgBins = new Bins[numPixels];
	bgModel = new BgModel[numPixels];
	maxBgBins = numSamples / minBinHeight;
	
	timestamp = 0.;//ms
	prev_timestamp = 0.;//ms
	prev_bg_frame_time = 0;
	bg_frame_counter = 0;
	bg_reset = false;
	prev_area = 0;
	sudden_change = false;
	
	SHADOW_LABEL = 80;
	PERSISTENCE_LABEL = 180;
	FOREGROUND_LABEL = 255;
	
	fgmask.create(frameSize, CV_8UC1);
	fgfiltered.create(frameSize, CV_8UC1);
	persistenceImage = Mat::zeros(frameSize, CV_8UC1);
	bgSample.create(frameSize, CV_8UC3);
	bgImage = Mat::zeros(frameSize, CV_8UC3);
	
	//initial message to be shown until the first fg mask is computed
	initialMsgGray = Mat::zeros(frameSize, CV_8UC1);
	putText(initialMsgGray, "Creating", Point(10,20), FONT_HERSHEY_SIMPLEX, 0.4, CV_RGB(255, 255, 255));
	putText(initialMsgGray, "initial", Point(10,40), FONT_HERSHEY_SIMPLEX, 0.4, CV_RGB(255, 255, 255));
	putText(initialMsgGray, "background...", Point(10,60), FONT_HERSHEY_SIMPLEX, 0.4, CV_RGB(255, 255, 255));
	
	initialMsgRGB = Mat::zeros(frameSize, CV_8UC3);
	putText(initialMsgRGB, "Creating", Point(10,20), FONT_HERSHEY_SIMPLEX, 0.4, CV_RGB(255, 255, 255));
	putText(initialMsgRGB, "initial", Point(10,40), FONT_HERSHEY_SIMPLEX, 0.4, CV_RGB(255, 255, 255));
	putText(initialMsgRGB, "background...", Point(10,60), FONT_HERSHEY_SIMPLEX, 0.4, CV_RGB(255, 255, 255));
	
	if(minBinHeight <= 1){
		minBinHeight = 1;
	}
	
	for(unsigned int p = 0; p < numPixels; ++p)
	{
		bgBins[p].binValues = new Vec3b[numSamples];
		bgBins[p].binHeights = new int[numSamples];
		bgBins[p].isFg = new bool[numSamples];
		
		bgModel[p].values = new Vec3b[maxBgBins];
		bgModel[p].isValid = new bool[maxBgBins];
		bgModel[p].isValid[0] = false;
		bgModel[p].isFg = new bool[maxBgBins];
		bgModel[p].counter = new int[maxBgBins];
	}	
}

void BackgroundSubtractorIMBS::rgbSuppression() {
	Mat sMat(frame.size(), CV_32FC1);
	for(int i = 0; i < frame.rows; ++i) {
		for(int j = 0; j < frame.cols; ++j) {
			sMat.at<float>(i,j) = frame.at<Vec3b>(i,j)[0] +
					frame.at<Vec3b>(i,j)[1] + frame.at<Vec3b>(i,j)[2];
		}
	}
	float* sMat_data = (float*) sMat.data;
	for(unsigned int p = 0; p < numPixels; ++p) {
		if(fgmask.data[p]) {
			for(unsigned int n = 0; n < maxBgBins; ++n) {
				if(!bgModel[p].isValid[n]) {
					break;
				}
				if(bgModel[p].isFg[n]) {
					continue;
				}
				float sPixel = bgModel[p].values[n].val[0] +
						bgModel[p].values[n].val[1] + bgModel[p].values[n].val[2];
				float ratio = sMat_data[p]/sPixel;
				if(    ratio > 0.7 && ratio < 1)
				{
					fgmask.data[p] = SHADOW_LABEL;
					break;
				}
			}//for
		}//if
	}//numPixels
}

void BackgroundSubtractorIMBS::apply(InputArray _frame, OutputArray _fgmask, float learningRate)
{
	frame = _frame.getMat();
	
	CV_Assert(frame.depth() == CV_8U);
	CV_Assert(frame.channels() == 3);
	
	bool needToInitialize = nframes == 0 || frame.type() != frameType;
	
	if( needToInitialize ) {
		initialize(frame.size(), frame.type());
	}
	
	_fgmask.create(frameSize, CV_8UC1);
	fgmask = _fgmask.getMat();
	fgmask = Scalar(0);
	
	//get current time
	prev_timestamp = timestamp;
	if(fps == 0.) {
		timestamp = getTimestamp();//ms
	}
	else {
		timestamp += 1000./fps;//ms
	}
	
	//check for global changes
	if(sudden_change) {
		changeBg();
	}
	
	if(bgModel[0].isValid[0]) {
		getFg();
		//hsvSuppression();
		rgbSuppression();
		filterFg();
	}
	
	//update the bg model
	updateBg();
	
	//show an initial message if the first bg is not yet ready
	if(!bgModel[0].isValid[0]) {
		initialMsgGray.copyTo(fgmask);
		initialMsgRGB.copyTo(bgImage);
	}
	++nframes;
}

void BackgroundSubtractorIMBS::updateBg() {
	///
	/// Uncomment to enable the background update over time.
	///
#if 1
	static bool isFirstTime = true;
	
	if(bg_reset) {
		if(bg_frame_counter > numSamples - 1) {
			bg_frame_counter = numSamples - 1;
		}
	}
	
	if(prev_bg_frame_time > timestamp) {
		prev_bg_frame_time = timestamp;
	}
	
	if(bg_frame_counter == numSamples - 1) {
		createBg(bg_frame_counter);
		if (isFirstTime)
		{
			isFirstTime = false;
			samplingPeriod = 2000.0;
			bg_reset = true;
		}
		bg_frame_counter = 0;
	}
	else { //bg_frame_counter < (numSamples - 1)
		if((timestamp - prev_bg_frame_time) >= samplingPeriod)
		{
			//get a new sample for creating the bg model
			prev_bg_frame_time = timestamp;
			frame.copyTo(bgSample);
			createBg(bg_frame_counter);
			bg_frame_counter++;
		}
	}
#endif
	///
	/// End comment.
	///
}

float BackgroundSubtractorIMBS::getTimestamp() {
	return ((float)getTickCount() - initial_tick_count)*1000./getTickFrequency();
}

void BackgroundSubtractorIMBS::hsvSuppression() {
	
	uchar h_i, s_i, v_i;
	uchar h_b, s_b, v_b;
	float h_diff, s_diff, v_ratio;
	
	Mat bgrPixel(cv::Size(1, 1), CV_8UC3);
	
	vector<Mat> imHSV;
	cv::split(convertImageRGBtoHSV(frame), imHSV);
	
	for(unsigned int p = 0; p < numPixels; ++p) {
		if(fgmask.data[p]) {
			
			h_i = imHSV[0].data[p];
			s_i = imHSV[1].data[p];
			v_i = imHSV[2].data[p];
			
			for(unsigned int n = 0; n < maxBgBins; ++n) {
				if(!bgModel[p].isValid[n]) {
					break;
				}
				
				if(bgModel[p].isFg[n]) {
					continue;
				}
				
				bgrPixel.at<cv::Vec3b>(0,0) = bgModel[p].values[n];
				
				cv::Mat hsvPixel = convertImageRGBtoHSV(bgrPixel);
				
				h_b = hsvPixel.at<cv::Vec3b>(0,0)[0];
				s_b = hsvPixel.at<cv::Vec3b>(0,0)[1];
				v_b = hsvPixel.at<cv::Vec3b>(0,0)[2];
				
				v_ratio = (float)v_i / (float)v_b;
				s_diff = std::abs(s_i - s_b);
				h_diff = std::min( std::abs(h_i - h_b), 255 - std::abs(h_i - h_b));
				
				if(	h_diff <= tau_h &&
						s_diff <= tau_s &&
						v_ratio >= alpha &&
						v_ratio < beta)
				{
					fgmask.data[p] = SHADOW_LABEL;
					break;
				}
			}//for
		}//if
	}//numPixels
}

void BackgroundSubtractorIMBS::createBg(unsigned int bg_sample_number) {
	//split bgSample in channels
	cv::split(bgSample, bgSampleBGR);
	
	const int n=numPixels;
	
	parallel_for_(cv::Range(0,n-1),
				  Parallel_createBg(fgmask, bgSampleBGR[0], bgSampleBGR[1], bgSampleBGR[2],
			bg_sample_number, numSamples, bgModel, bgBins,
			minBinHeight, maxBgBins, fgThreshold,
			PERSISTENCE_LABEL, FOREGROUND_LABEL,
			associationThreshold));
	
	if(bg_sample_number == (numSamples - 1)) {
		persistenceImage = Scalar(0);
		
		bg_reset = false;
		if(sudden_change) {
			numSamples *= 3.;
			samplingPeriod *= 2.;
			sudden_change = false;
		}
		
		for(unsigned int i = 0; i < numPixels; i++) {
			persistenceMap[i] = 0;
		}
		
		unsigned int p = 0;
		for(int i = 0; i < bgImage.rows; ++i) {
			for(int j = 0; j < bgImage.cols; ++j, ++p) {
				bgImage.at<cv::Vec3b>(i,j) = bgModel[p].values[0];
			}
		}
	}
}

void BackgroundSubtractorIMBS::getFg() {
	fgmask = Scalar(0);
	cv::split(frame, frameBGR);
	
	const int n=numPixels;
	parallel_for_(cv::Range(0,n-1),
				  Parallel_getFg(fgmask, frameBGR[0], frameBGR[1], frameBGR[2],
			persistenceMap, bgModel, maxBgBins, fgThreshold,
			PERSISTENCE_LABEL, FOREGROUND_LABEL,
			timestamp, prev_timestamp, persistencePeriod));
}

void BackgroundSubtractorIMBS::areaThresholding()
{
	float maxArea = 0.6 * numPixels;
	
	std::vector < std::vector<Point> > contours;
	Mat tmpBinaryImage = fgfiltered.clone();
	findContours(tmpBinaryImage, contours, RETR_LIST, CHAIN_APPROX_NONE);
	
	tmpBinaryImage = Scalar(0);
	
	for (size_t contourIdx = 0; contourIdx < contours.size(); ++contourIdx)
	{
		Moments moms = moments(Mat(contours[contourIdx]));
		float area = moms.m00;
		if (area < minArea || area >= maxArea)
			continue;
		else {
			drawContours( tmpBinaryImage, contours, contourIdx, Scalar(255), CV_FILLED );
		}
	}	
	for(int i = 0; i < fgfiltered.rows; ++i) {
		for(int j = 0; j < fgfiltered.cols; ++j) {
			if(!tmpBinaryImage.at<uchar>(i,j)) {
				fgfiltered.at<uchar>(i,j) = 0;
			}
		}
	}
}

// Create a HSV image from the RGB image using the full 8-bits, since OpenCV only allows Hues up to 180 instead of 255.
// ref: "http://cs.haifa.ac.il/hagit/courses/ist/Lectures/Demos/ColorApplet2/t_convert.html"
// Remember to free the generated HSV image.
Mat BackgroundSubtractorIMBS::convertImageRGBtoHSV(const Mat& imageRGB)
{
	float fR, fG, fB;
	float fH, fS, fV;
	const float FLOAT_TO_BYTE = 255.0f;
	const float BYTE_TO_FLOAT = 1.0f / FLOAT_TO_BYTE;
	
	// Create a blank HSV image
	Mat imageHSV(imageRGB.size(), CV_8UC3);
	//if (!imageHSV || imageRGB->depth != 8 || imageRGB->nChannels != 3) {
	//printf("ERROR in convertImageRGBtoHSV()! Bad input image.\n");
	//exit(1);
	//}
	
	int h = imageRGB.rows;		// Pixel height.
	int w = imageRGB.cols;		// Pixel width.
	//int rowSizeRGB = imageRGB->widthStep;	// Size of row in bytes, including extra padding.
	//char *imRGB = imageRGB->imageData;	// Pointer to the start of the image pixels.
	//int rowSizeHSV = imageHSV->widthStep;	// Size of row in bytes, including extra padding.
	//char *imHSV = imageHSV->imageData;	// Pointer to the start of the image pixels.
	for (int y = 0; y < h; ++y) {
		for (int x = 0; x < w; ++x) {
			// Get the RGB pixel components. NOTE that OpenCV stores RGB pixels in B,G,R order.
			//uchar *pRGB = (uchar*)(imRGB + y*rowSizeRGB + x*3);
			int bB = imageRGB.at<Vec3b>(y,x)[0]; //*(uchar*)(pRGB+0);	// Blue component
			int bG = imageRGB.at<Vec3b>(y,x)[1]; //*(uchar*)(pRGB+1);	// Green component
			int bR = imageRGB.at<Vec3b>(y,x)[2]; //*(uchar*)(pRGB+2);	// Red component
			
			// Convert from 8-bit integers to floats.
			fR = bR * BYTE_TO_FLOAT;
			fG = bG * BYTE_TO_FLOAT;
			fB = bB * BYTE_TO_FLOAT;
			
			// Convert from RGB to HSV, using float ranges 0.0 to 1.0.
			float fDelta;
			float fMin, fMax;
			int iMax;
			// Get the min and max, but use integer comparisons for slight speedup.
			if (bB < bG) {
				if (bB < bR) {
					fMin = fB;
					if (bR > bG) {
						iMax = bR;
						fMax = fR;
					}
					else {
						iMax = bG;
						fMax = fG;
					}
				}
				else {
					fMin = fR;
					fMax = fG;
					iMax = bG;
				}
			}
			else {
				if (bG < bR) {
					fMin = fG;
					if (bB > bR) {
						fMax = fB;
						iMax = bB;
					}
					else {
						fMax = fR;
						iMax = bR;
					}
				}
				else {
					fMin = fR;
					fMax = fB;
					iMax = bB;
				}
			}
			fDelta = fMax - fMin;
			fV = fMax;				// Value (Brightness).
			if (iMax != 0) {			// Make sure its not pure black.
				fS = fDelta / fMax;		// Saturation.
				float ANGLE_TO_UNIT = 1.0f / (6.0f * fDelta);	// Make the Hues between 0.0 to 1.0 instead of 6.0
				if (iMax == bR) {		// between yellow and magenta.
					fH = (fG - fB) * ANGLE_TO_UNIT;
				}
				else if (iMax == bG) {		// between cyan and yellow.
					fH = (2.0f/6.0f) + ( fB - fR ) * ANGLE_TO_UNIT;
				}
				else {				// between magenta and cyan.
					fH = (4.0f/6.0f) + ( fR - fG ) * ANGLE_TO_UNIT;
				}
				// Wrap outlier Hues around the circle.
				if (fH < 0.0f)
					fH += 1.0f;
				if (fH >= 1.0f)
					fH -= 1.0f;
			}
			else {
				// color is pure Black.
				fS = 0;
				fH = 0;	// undefined hue
			}
			
			// Convert from floats to 8-bit integers.
			int bH = (int)(0.5f + fH * 255.0f);
			int bS = (int)(0.5f + fS * 255.0f);
			int bV = (int)(0.5f + fV * 255.0f);
			
			// Clip the values to make sure it fits within the 8bits.
			if (bH > 255)
				bH = 255;
			if (bH < 0)
				bH = 0;
			if (bS > 255)
				bS = 255;
			if (bS < 0)
				bS = 0;
			if (bV > 255)
				bV = 255;
			if (bV < 0)
				bV = 0;
			
			// Set the HSV pixel components.
			imageHSV.at<Vec3b>(y, x)[0] = bH;		// H component
			imageHSV.at<Vec3b>(y, x)[1] = bS;		// S component
			imageHSV.at<Vec3b>(y, x)[2] = bV;		// V component
		}
	}
	return imageHSV;
}

void BackgroundSubtractorIMBS::getBackgroundImage(OutputArray backgroundImage) const
{
	bgImage.copyTo(backgroundImage);        
}

void BackgroundSubtractorIMBS::filterFg() {
	
	unsigned int cnt = 0;
	for(unsigned int p = 0; p < numPixels; ++p) {
		if(fgmask.data[p] == (uchar)255) {
			fgfiltered.data[p] = 255;
			cnt++;
		}
		else {
			fgfiltered.data[p] = 0;
		}
	}
	
	if(cnt > numPixels*0.5) {
		sudden_change = true;
	}
	
	if(morphologicalFiltering) {
		cv::Mat element3(3,3,CV_8U,cv::Scalar(1));
		cv::morphologyEx(fgfiltered, fgfiltered, cv::MORPH_OPEN, element3);
		cv::morphologyEx(fgfiltered, fgfiltered, cv::MORPH_CLOSE, element3);
	}
	
	areaThresholding();
	
	for(unsigned int p = 0; p < numPixels; ++p) {
		if(fgmask.data[p] == PERSISTENCE_LABEL) {
			fgfiltered.data[p] = PERSISTENCE_LABEL;
		}
		else if(fgmask.data[p] == SHADOW_LABEL) {
			//fgfiltered.data[p] = SHADOW_LABEL;
			fgfiltered.data[p] = 0;
		}
	}
	
	fgfiltered.copyTo(fgmask);
}

void BackgroundSubtractorIMBS::changeBg() {
	
	//samplingPeriod /= 2.;
	//numSamples /= 2.;
	//bg_reset = true;
	
	///
	/// Uncomment to enable the background update over time.
	///
#if 1
	if(!bg_reset) {
		numSamples /= 3.;
		samplingPeriod /= 2.;
		bg_frame_counter = 0;
		bg_reset = true;
	}
#endif
	///
	/// End comment.
	///
}

bool BackgroundSubtractorIMBS::loadBg(const char* filename) {
	string line;
	ifstream file(filename, ifstream::in);
	int c = 0;
	if(file.is_open()) {
		loadedBg = true;
		isBackgroundCreated = true;
		
		getline(file, line);
		
		int index = line.find_first_of(" ");
		string widthString = line.substr(0, index);
		int width;
		istringstream ss_w(widthString);
		ss_w >> width;
		line.erase(0, index+1);
		
		string heightString = line;
		int height;
		istringstream ss_h(heightString);
		ss_h >> height;
		
		Size frameSize(width, height);
		getline(file, line);
		
		int frameType = 0;
		
		this->frameSize = frameSize;
		this->frameType = frameType;
		this->numPixels = frameSize.width*frameSize.height;
		
		delete[] bgBins;
		delete[] bgModel;
		delete[] persistenceMap;
		
		persistenceMap = new unsigned int[numPixels];
		
		for(unsigned int i = 0; i < numPixels; i++) {
			persistenceMap[i] = 0;
		}
		
		bgBins = new Bins[numPixels];
		bgModel = new BgModel[numPixels];
		maxBgBins = numSamples / minBinHeight;
		
		timestamp = 0.;//ms
		prev_timestamp = 0.;//ms
		prev_bg_frame_time = 0;
		bg_frame_counter = 0;
		bg_reset = false;
		prev_area = 0;
		sudden_change = false;
		
		SHADOW_LABEL = 80;
		PERSISTENCE_LABEL = 180;
		FOREGROUND_LABEL = 255;
		
		fgmask.create(frameSize, CV_8UC1);
		fgfiltered.create(frameSize, CV_8UC1);
		persistenceImage = Mat::zeros(frameSize, CV_8UC1);
		bgSample.create(frameSize, CV_8UC3);
		bgImage = Mat::zeros(frameSize, CV_8UC3);
		
		if(minBinHeight <= 1){
			minBinHeight = 1;
		}
		
		for(unsigned int p = 0; p < numPixels; ++p)
		{
			bgBins[p].binValues = new Vec3b[numSamples];
			bgBins[p].binHeights = new int[numSamples];
			bgBins[p].isFg = new bool[numSamples];
			bgModel[p].values = new Vec3b[maxBgBins];
			bgModel[p].isValid = new bool[maxBgBins];
			bgModel[p].isValid[0] = false;
			bgModel[p].isFg = new bool[maxBgBins];
			bgModel[p].counter = new int[maxBgBins];
		}		
		
		while(!file.eof()) {
			getline(file, line);
			
			int n = 0;
			
			while(line.length() > 1){
				
				int index = line.find_first_of(" ");
				string red = line.substr(0, index);
				int r;
				istringstream ss_r(red);
				ss_r >> r;
				line.erase(0, index+1);
				
				index = line.find_first_of(" ");
				string green = line.substr(0, index);
				int g;
				istringstream ss_g(green);
				ss_g >> g;
				line.erase(0, index+1);
				
				index = line.find_first_of(" ");
				string blue = line.substr(0, index);
				int b;
				istringstream ss_b(blue);
				ss_b >> b;
				line.erase(0, index+1);
				
				bgModel[c].values[n].val[0] = b;
				bgModel[c].values[n].val[1] = g;
				bgModel[c].values[n].val[2] = r;
				bgModel[c].isValid[n] = true;
				bgModel[c].isFg[n] = false;
				bgModel[c].counter[n] = minBinHeight;
				
				if (n == 0) {
					int i = c/bgImage.cols;
					int j = c - i*bgImage.cols;
					
					bgImage.at<Vec3b>(i, j) = bgModel[c].values[0];
				}
				
				n++;
			}
			
			c++;
		}
		
		file.close();
		
		return true;
	}
	else return false;
}

void BackgroundSubtractorIMBS::saveBg(string* filename) {
	bgFilename = filename;
}
