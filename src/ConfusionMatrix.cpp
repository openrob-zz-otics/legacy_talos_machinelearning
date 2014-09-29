

#include "ConfusionMatrix.h"

ConfusionMatrix::~ConfusionMatrix(){}


ConfusionMatrix::ConfusionMatrix(){}

void ConfusionMatrix::print(std::string& output)
{
	cv::Mat mtx = this->confusion_matrix;

	for (int i = 0; i < this->class_labels.size(); i++)	
	{
		ROS_INFO("Class %d is %s", i, class_labels[i].c_str());
	}

	 printf("\n --> Matrix type is CV_32S \n");

       for( size_t i = 0; i < mtx.rows; i++ ) {
         for( size_t j = 0; j < mtx.cols; j++ ) {
           printf( " %d  ", mtx.at<int>(i,j) );
         } printf(" %s ", class_labels[i].c_str()); printf("\n");
       }

}
