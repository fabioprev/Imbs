#pragma once

#include <opencv2/core/core.hpp>

class VisualReading
{
	public:
		struct Model
		{
			static const int HISTOGRAM_VECTOR_LENGTH = 256;
			
			std::pair<cv::Point2f,cv::Point2f> boundingBox;
			
			int width;
			
			int height;
			
			float barycenter;
			
			float histograms[3][HISTOGRAM_VECTOR_LENGTH];
			
			Model() : width(0), height(0), barycenter(-1)
			{
				for (int i = 0; i < VisualReading::Model::HISTOGRAM_VECTOR_LENGTH; ++i)
				{
					histograms[0][i] = 0;
					histograms[1][i] = 0;
					histograms[2][i] = 0;
				}
			}
			
			Model& operator= (const Model& other)
			{
				boundingBox = other.boundingBox;
				width = other.width;
				height = other.height;
				barycenter = other.barycenter;
				
				for (int i = 0; i < HISTOGRAM_VECTOR_LENGTH; ++i)
				{
					histograms[0][i] = other.histograms[0][i];
					histograms[1][i] = other.histograms[1][i];
					histograms[2][i] = other.histograms[2][i];
				}
				
				return *this;
			}
		};
		
		struct Observation
		{
			cv::Point2f observation;
			
			cv::Point2f head;
			
			cv::Point2f sigma;
			
			Model model;
		};
		
		VisualReading();
		
		~VisualReading();
		
		std::vector<Observation>& getObservations() { return observations; }
		
		const std::vector<Observation>& getObservations() const { return observations; }
		
		const cv::Point2f& getObservationsAgentPose() const { return observationsAgentPose; }
		
		void setObservations(const std::vector<Observation>& obs);
		
		void setObservationsAgentPose(const cv::Point2f& pose);
		
	private:
		std::vector<Observation> observations;
		
		cv::Point2f observationsAgentPose;
};
