
#ifndef SUPERVISEDFEATURES_H
#define SUPERVISEDFEATURES_H

#include "daspr_vision.h"

class SupervisedFeatures
{

	public:

		int class_label;
		std::string class_name;
		cv::Mat features; 
		std::vector<cv::KeyPoint> keypoints;
		cv::Mat original_image;		

		// Each row specifies 1 feature point
				// The M columns define the dimensionality of the feature point
				// The N rows define the number of feature points

		SupervisedFeatures();
		~SupervisedFeatures();

		




};

#endif
