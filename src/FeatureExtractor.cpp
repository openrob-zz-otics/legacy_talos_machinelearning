

#include "FeatureExtractor.h"

void FeatureExtractor::extractSURF(std::multimap<std::string, cv::Mat>& image_set, std::vector<SupervisedFeatures>& features)
{
	
	cv::Ptr<cv::DescriptorExtractor> extractor(new cv::SurfDescriptorExtractor);
	cv::Ptr<cv::FeatureDetector> detector(new cv::SurfFeatureDetector(400));
	std::multimap<std::string, cv::Mat>::iterator itr, itr_s;

	for (itr = image_set.begin(); itr != image_set.end(); itr = itr_s)
	{
		std::string key = (*itr).first;

		std::pair<std::multimap<std::string, cv::Mat>::iterator, std::multimap<std::string, cv::Mat>::iterator> key_range = image_set.equal_range(key);

		// Iterates over all of the images in the class
		for (itr_s = key_range.first; itr_s != key_range.second; ++itr_s)
		{
			// Instead, we extract surf from these images and create the supervised feats.
			cv::Mat image = (*itr_s).second;
			cv::imshow("image", image);
			cv::Mat descriptors;
			std::vector<cv::KeyPoint> keypoints;
			detector->detect(image, keypoints);
			extractor->compute(image, keypoints, descriptors);
			
			SupervisedFeatures feats;

			ROS_INFO("Descriptors dims %d %d", descriptors.rows, descriptors.cols);
			feats.features = descriptors;
			feats.class_name = (*itr_s).first;
			feats.keypoints = keypoints;
			feats.original_image = image;
			features.push_back(feats);
		}
	} 



	/*Ptr<FeatureDetector> detector;
	Ptr<DescriptorExtractor> extractor;

	for (int i = 0; i < images.size(); i++)
	{
		cv::Mat descriptors;
		vector<cv::KeyPoint> keypoints;
		detector->detect(images[i],keypoints);
		extractor->compute(images[i], keypoints, descriptors);	

		SupervisedFeatures features;
		features.features = descriptors;
		features.class_name = 
	}*/
}
