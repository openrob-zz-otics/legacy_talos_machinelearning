// Author: Devon Ash
// Copyright: DASpR Inc.
// Licensing rights: DASpR Inc.
// Usage rights: DASpR Inc.

#ifndef FEATUREEXTRACTOR_H
#define FEATUREEXTRACTOR_H

#include "daspr_vision.h"
#include "SupervisedFeatures.h"

// Interface class
// Usage:
// All feature extractors must implement these methods
// Inputs to a feature extractor: vector<cv::Mat>
// Outputs of a feature extractor: vector<cv::Mat> where each mat represents the features in that image, and the vector represents all of the images input.
class FeatureExtractor
{

	private:
		~FeatureExtractor(){}
	public:
		static void extractSURF(std::multimap<std::string, cv::Mat>& image_set, std::vector<SupervisedFeatures>& feats);
		

};




#endif
