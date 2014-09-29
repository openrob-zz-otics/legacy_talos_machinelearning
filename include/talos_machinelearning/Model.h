

#ifndef MODEL_H
#define MODEL_H

#include "Experiment.h"
#include "Dataset.h"
#include "ConfusionMatrix.h"
#include <vector>

class Model
{

	public:

		Model(){}
		~Model(){}

		// Tests the model
		void test();

		// Visualizes the model		
		void show();

		// Trains the model
		void train();

		// Gives a live demonstration
		void demonstrate();
		
		// Adds an experiment to perform, each experiment has its own params and stuff.
		void addExperiment(Experiment& e);

		// Loads the images and returns them in a vector
		void load(std::string model_path);

		void save(std::string model_path);

		void confusionMatrix(ConfusionMatrix& mtx);

		float accuracy();
};

#endif
