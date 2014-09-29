
#include "Dataset.h"


char* getFileType(char* filename)
{
	// Returns the file type of a given filename.
	char* stf = strtok(filename, ".");
	return strtok(NULL, ".");
}

void Dataset::generate_training_set(float percent)
{
	// Iterate the map, for all of the values in the keys, iterate up to split% size of the bucket
	// and put them in the training side.

	std::multimap<std::string, std::string>::iterator itr, itr_s;
		
	int class_number = 0;
	// Iterates over all of the classes 
	for (itr = class_images.begin(); itr != class_images.end(); itr = itr_s)
	{
		std::string key = (*itr).first;
		
		ROS_INFO("b4");
		ROS_INFO("b3");
		std::pair<std::multimap<std::string, std::string>::iterator, std::multimap<std::string, std::string>::iterator> key_range = class_images.equal_range(key);


		int num_elems = std::distance(key_range.first, key_range.second);
		int stopping = num_elems*percent;

		int k = 0;

		this->class_labels.insert(std::pair<std::string, int>(key, class_number));
		ROS_INFO("Inserting key %s", key.c_str());
		class_number++;
		
		ROS_INFO("Class %d", class_number);
		// Iterates over all of the images in the class
		for (itr_s = key_range.first; itr_s != key_range.second; ++itr_s)
		{


			cv::Mat image;
			std::string path;			
			// We use test set if k > stop
			path = dataset_path + "/" + (*itr_s).first + "/" + (*itr_s).second;
			image = cv::imread(path+".jpg");
			ROS_INFO("Opening image %s", path.c_str());			

			if (k >= stopping) 
			{
				test_image_paths.insert(std::pair<std::string, std::string>((*itr_s).first, (*itr_s).second));
				loaded_test_set.insert(std::pair<std::string, cv::Mat>(key, image));

			} else // Else we use the training set
			{
				training_image_paths.insert(std::pair<std::string, std::string>((*itr_s).first,(*itr_s).second));		
				loaded_training_set.insert(std::pair<std::string, cv::Mat>(key, image));
			}
//TODO
// We can also load the images right here, why not?
			k++;
		}
	} 
}


Dataset::Dataset()
{

}

// The dataset should automatically do a 70/30 partition split for generalization and cross-validation errors
Dataset::Dataset(std::string path)
{
	

	//TODO
	// Iterate through the directory and partition the images in a random 70/30 split. Takes the entire directory
	// and then split its in a 70/30, does this for all directories. 

	this->dataset_path = path;
	DIR *dir = opendir(path.c_str());
	struct dirent *entry = readdir(dir);


	int k = 0; 
	while (entry != NULL)
	{
		if (strcmp(entry->d_name, ".") != 0 && strcmp(entry->d_name, "..") != 0)
		{

			// We're going to need to iterate this directory to get the .jpgs
			std::string temp_dir = path + "/" + entry->d_name;		
			this->classes.push_back(temp_dir);
			
			DIR *inner_dir = opendir(temp_dir.c_str());	
			struct dirent *inner_entry = readdir(inner_dir);	

			while (inner_entry != NULL)
			{

				if (strcmp(inner_entry->d_name, ".") != 0 && strcmp(entry->d_name, "..") != 0)
				{	
		
					char* filetype = getFileType(inner_entry->d_name);
					
					if (filetype != NULL)
					{
						if (strcmp( filetype,  "jpg") == 0)
						{	
							std::string temp(path+"/"+entry->d_name+"/"+inner_entry->d_name);
							// This adds a .jpg to a class bucket 
							this->class_images.insert(std::pair<std::string, std::string>(entry->d_name, inner_entry->d_name));
							// Have to append the dirs
							all_images.push_back(temp);

							// Class databases; 
							ROS_INFO("Class: %s || Image: %s", entry->d_name, inner_entry->d_name);
						}
					}
				}
				
				inner_entry = readdir(inner_dir);
			}
			k++;
		}

		entry = readdir(dir);
	}

	// By now we should have all of the classes in the class_images map. Now have to divide to 70/30
	// generate_training_set

	ROS_INFO("Gentest");
	generate_training_set(0.7);
}

Dataset::~Dataset()
{
	
}
