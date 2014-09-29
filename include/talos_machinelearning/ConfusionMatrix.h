

#ifndef CONFUSIONMATRIX_H
#define CONFUSIONMATRIX_H


#include "daspr_vision.h"

class ConfusionMatrix
{

	public:

		
		cv::Mat confusion_matrix;
		std::vector<std::string> class_labels;

		ConfusionMatrix();
		~ConfusionMatrix();

		// Prints confusion matrix to file
		void print(std::string& output);

};







#endif
