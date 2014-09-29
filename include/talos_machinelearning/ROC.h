

#ifndef ROC_H
#define ROC_H


#include "daspr_vision.h"
#include "Model.h"

class ROC
{


	public:

		ROC();
		ROC(Model m);
		~ROC();

		void showPlot();

		// Prints image of plot out
		void printPlot();
		float AUC();

};
#endif 
