


#ifndef EXPERIMENT_H
#define EXPERIMENT_H

#include "daspr_vision.h"
#include "Dataset.h"
#include "ModelParams.h"

class Experiment 
{

	public:

		std::string name;
		int iterations;

		Experiment();
		~Experiment();
	
		Experiment(Dataset, ModelParams, std::string, int);

		// Runs the experiment for # of iterations specified
		void perform();

		// Saves the experiment output to a .txt file with name experiment name
		// In case of successive iterations, appends the iteration number
		void save();

};


#endif
