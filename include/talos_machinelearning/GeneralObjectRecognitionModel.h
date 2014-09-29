

#ifndef GENERALOBJECTRECOGNITIONMODEL_H
#define GENERALOBJECTRECOGNITIONMODEL_H

#include "Normalization.h"
#include "SURFFeatureExtractor.h"

#include "daspr_vision.h"
#include "Dataset.h"
#include "Model.h"
#include "Experiment.h"


class GeneralObjectRecognitionModel : public Model
{

	private:

		std::vector<Experiment> experiments;

	public:

		std::map<std::string, int> model_parameters;

	//cv::BOWKMeansTrainer bag_of_words;
		void get_training_images(std::multimap<std::string, cv::Mat>&);

		void computeTrainingMatrix(cv::Mat& trainingMat, cv::Mat& labelMat);

		void cluster();

		int validation_percentage;

		Dataset dataset;

		GeneralObjectRecognitionModel();

		~GeneralObjectRecognitionModel();
		
		GeneralObjectRecognitionModel(Dataset dataset);

		void addExperiment(Experiment& e);

		float accuracy(ConfusionMatrix& mtx);
		
		void confusionMatrix(ConfusionMatrix& mtx);

		void test();

		void show();

		void demonstrate();

		void train();

		void load(std::string model_path);

		void save(std::string model_path);
};






#endif
