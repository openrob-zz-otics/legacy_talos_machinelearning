
#ifndef NORMALIZATION_H
#define NORMALIZATION_H

#include "daspr_vision.h"

class Normalization
{
	private:

		Normalization();
		~Normalization();

	public:

		static void differenceOfGaussians(std::vector<cv::Mat>& normalized_images, int kernel1, int kernel2);
		static void grayscale(std::vector<cv::Mat>& images);
		static void resize(std::vector<cv::Mat>& images, int width, int height);
	
};

#endif
