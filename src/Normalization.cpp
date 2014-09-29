#include "Normalization.h"


void Normalization::differenceOfGaussians(std::vector<cv::Mat>& images, int kernel1, int kernel2)
{
	for (int i = 0; i < images.size(); i++)
	{
		cv::Mat image = images[i];

		if (!image.data)
		{
			ROS_INFO("Error loading image");
		}

		cv::Mat dog_mat;
		cv::Mat dog1, dog2;

		cv::GaussianBlur(image, dog1, cv::Size(kernel1, kernel1), 0);
		cv::GaussianBlur(image, dog2, cv::Size(kernel2, kernel2), 0);

		dog_mat = (dog1 - dog2);

		images[i] = dog_mat;
	}
}

// Resizes an image set
void Normalization::resize(std::vector<cv::Mat>& images, int width, int height)
{
	for (int i = 0; i < images.size(); i++)
	{
		cv::resize(images[i], images[i], cv::Size(width, height));
	}
}


void Normalization::grayscale(std::vector<cv::Mat>& images)
{
	for (int i = 0; i < images.size(); i++)
	{
		cv::cvtColor(images[i], images[i], CV_RGB2GRAY);
	}
}
