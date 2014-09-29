

#ifndef DATASET_H
#define DATASET_H

#include "daspr_vision.h"
#include <map>
#include <vector>



// Input: dataset path
class Dataset
{


	public:
		int sample_size;
		int class_count;

		bool loaded;

		std::string dataset_path;
		
		std::vector<std::string> classes;
		std::map<std::string, int> class_labels;
		std::multimap<std::string, std::string> class_images;
		std::vector<std::string> all_images;

		std::multimap<std::string, std::string> training_image_paths;
		std::multimap<std::string, std::string> test_image_paths;
		
	//	std::vector<cv::Mat> training_images;
	//	std::vector<cv::Mat> test_images;
		std::multimap<std::string, cv::Mat> loaded_training_set;
		std::multimap<std::string, cv::Mat> loaded_test_set;

		Dataset();
		Dataset(std::string path);
		~Dataset();

		void generate_training_set(float percent);
};

#endif
