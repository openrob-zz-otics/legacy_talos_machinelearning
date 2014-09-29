

#ifndef SURFFEATUREEXTRACTOR_H
#define SURFFEATUREEXTRACTOR_H
#include "FeatureExtractor.h"
#include "Dataset.h"
#include "SupervisedFeatures.h"

class SURFFeatureExtractor : public FeatureExtractor
{

	private:
		std::vector<cv::Mat*> dataset;


	public:

		std::vector<cv::Mat*> extracted_images;

		SURFFeatureExtractor();
		~SURFFeatureExtractor();

		SURFFeatureExtractor(std::vector<cv::Mat*>& dataset);

		// Returns the extracted features
		void extract(std::vector<SupervisedFeatures>& features);



};




#endif
