#ifndef DATALOADER_HPP
#define DATALOADER_HPP
#include <iostream>
#include <fstream>
#include <eigen/Eigen/Dense>
#include <string>

namespace neu {
    struct DataLoader {
    private:
        // Full data
        Eigen::MatrixXd data;

        // Features
        Eigen::MatrixXd X;

        // Target
        Eigen::VectorXd y;

        // Read in a datafile as a vector of vectors
        static std::vector<std::vector<std::string>> readDataFile(std::string &filepath, std::string &delim);

        // Function to convert the data file to a data matrix
        static Eigen::MatrixXd fileToMatrix(std::string &filepath, std::string &delim, bool header);

        // Extracts the columns containing the features
        Eigen::MatrixXd extractFeatures(int target_col);

        // Extracts the column containing the target
        Eigen::VectorXd extractTarget(int target_col);


    public:
        // Constructor
        DataLoader(std::string filepath, std::string delim, bool head, int target_col);

        // Getter method to get features
        Eigen::MatrixXd getX();

        // Getter method to get target
        Eigen::VectorXd gety();

        // Prepend a column of 1s to the features matrix to
        // account for the constant term in a linear equation
        static Eigen::MatrixXd add_constant(Eigen::MatrixXd X);

        // Split train and test sets based on specified size
        // Shuffles rows for in case data is not normally distributed
        // Use std::tie to unpack the output tuple
        static std::tuple<Eigen::MatrixXd, Eigen::VectorXd, Eigen::MatrixXd, Eigen::VectorXd>
        trainTestSplit(Eigen::MatrixXd &X, Eigen::VectorXd &y, float test_size);

        // Method to scale and center a feature matrix. Useful for distance based algorithms.
        static Eigen::MatrixXd standardizeFeatures(Eigen::MatrixXd X);
    };
} // end of namespace
#endif
