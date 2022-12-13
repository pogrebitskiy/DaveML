#include "ArgumentSettings.hpp"

#include <iostream>
#include <filesystem>

namespace neu{

	// Handles arguments for the data loader
	ArgumentSettings::ArgumentSettings(int argc, char* argv[]){
	
		// Make sure enough args were passed
		if (argc < 5){
			std::cout << "Error running program " << argv[0] << ":\n:"
					   <<"e.g. ./prog data.csv , 1 0\n\n"
					   <<"First argument is filepath of data (including extension)\n"
					   <<"Second argument is the delimiter that separates the data\n"
					   <<"Third argument is if the data contains a header row (1=yes, 0=no)\n"
					   <<"Fourth argument is the index of the target column in the data"
					   <<std::endl;
		exit(1);
		}

		if (!std::filesystem::exists(argv[1])){
			std::cout << "Could not find data file, please check path" << std::endl;

		exit(1);
		}

		// Set the arguments
		m_filename = argv[1];
		m_delimiter = argv[2];
		m_header = std::stoi(argv[3]);
		m_target_col = std::stoi(argv[4]);
		
	}	
}
