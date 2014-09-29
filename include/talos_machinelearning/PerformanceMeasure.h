

#ifndef PERFORMANCEMEASURE_H
#define PERFORMANCEMEASURE_H

#include "daspr_vision.h"
#include "ConfusionMatrix.h"
#include "ROC.h"
#include "Model.h"


class PerformanceMeasure
{

	private:
		PerformanceMeasure();
		~PerformanceMeasure();
	public:
		static float accuracy(Model m);
		static float predictionThreshold(Model m);
		static ConfusionMatrix confusionMatrix(Model m);
		static ROC getROC(Model m);
	
};


#endif
