#include "DataLoader.hpp"

#include <vector>
#include <stdlib.h>
#include <cmath>
#include <boost/algorithm/string.hpp>
#include <random>
#include <algorithm>
#include <iterator>


namespace neu {

// Constructor
    DataLoader::DataLoader(std::string filepath, std::string delim, bool head, int target_col) {
        // Convert csv to data matrix
        this->data = DataLoader::fileToMatrix(filepath, delim, head);

        // Extract the features matrix
        this->X = DataLoader::extractFeatures(target_col);

        // Extract the target column
        this->y = DataLoader::extractTarget(target_col);

    }

    std::vector<std::vector<std::string>> DataLoader::readDataFile(std::string &filepath, std::string &delim) {
        // Open file
        std::ifstream file(filepath);
        std::vector<std::vector<std::string>> stringdata;

        std::string line;

        // Read each line and split it based on the specified delimiter
        while (getline(file, line)) {
            std::vector<std::string> v;
            boost::algorithm::split(v, line, boost::is_any_of(delim));

            // Replace with "NaN" if there's an empty cell in the dataset
            std::replace_if(v.begin(), v.end(), [](std::string &in) { return in.empty(); },
                            "NaN");
            stringdata.push_back(v);
        }


        file.close();

        return stringdata;
    }

    Eigen::MatrixXd DataLoader::fileToMatrix(std::string &filepath, std::string &delim, bool header) {
        std::vector<std::vector<std::string>> dataset = readDataFile(filepath, delim);
        int rows = dataset.size();
        int cols = dataset[0].size();

        // Convert the read-in data to an Eigen::Matrix
        Eigen::MatrixXd mat(cols, rows);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; ++j) {
                auto elem = atof(dataset[i][j].c_str());
                if (!std::isnan(elem)) {
                    mat(j, i) = elem;
                    // Exit program and indicate that dataset can't contain NaNs (empty cells)
                } else {
                    std::cout << "Could not load data, please remove NaN (empty) values" << std::endl;
                    exit(1);
                }
            }
        }

        if (header) {
            // If the data contains a header, subset the matrix to exclude the first row
            return mat.transpose()(Eigen::seq(1, Eigen::placeholders::last), Eigen::placeholders::all);
        } else {
            // Assign the resulting matrix to data member variable
            return mat.transpose();
        }
    }

    Eigen::MatrixXd DataLoader::extractFeatures(int target_col) {
        std::vector<int> indices;
        // Select all the columns that aren't the target column
        for (int i = 0; i < data.cols(); i++) {
            if (i != target_col) {
                indices.push_back(i);
            } else {
                continue;
            }
        }
        return data(Eigen::placeholders::all, indices);
    }

    Eigen::VectorXd DataLoader::extractTarget(int target_col) {
        return Eigen::VectorXd(data.col(target_col));
    }


    Eigen::MatrixXd DataLoader::getX() {
        return X;
    }

    Eigen::VectorXd DataLoader::gety() {
        return y;
    }

    Eigen::MatrixXd DataLoader::add_constant(Eigen::MatrixXd X) {
        Eigen::MatrixXd new_x;
        // Set the size of the new matrix to allocate space for the new column
        new_x.resize(X.rows(), X.cols() + 1);
        // Add the col of 1s
        new_x << Eigen::VectorXd::Ones(X.rows(), 1), X;
        return new_x;
    }

    std::tuple<Eigen::MatrixXd, Eigen::VectorXd, Eigen::MatrixXd, Eigen::VectorXd>
    DataLoader::trainTestSplit(Eigen::MatrixXd &X, Eigen::VectorXd &y, float test_size) {
        // Create vector of all row indices
        std::vector<int> rows(X.rows());
        std::iota(std::begin(rows), std::end(rows), 0);

        // Find how many rows should be seperated for testing
        int test_num = round(X.rows() * test_size);

        std::vector<int> test_idx;
        std::vector<int> train_idx;

        // Randomly pick the indices for the test data
        std::sample(rows.begin(), rows.end(), std::back_inserter(test_idx), test_num,
                    std::mt19937{std::random_device{}()});

        // Find the set difference between all the rows indices and the test indices
        std::set_difference(rows.begin(), rows.end(), test_idx.begin(), test_idx.end(),
                            std::back_inserter(train_idx));

        // Initialize train data
        Eigen::MatrixXd X_train = X(train_idx, Eigen::placeholders::all);
        Eigen::VectorXd y_train = y(train_idx);

        // Initialize test data
        Eigen::MatrixXd X_test = X(test_idx, Eigen::placeholders::all);
        Eigen::VectorXd y_test = y(test_idx);

        // Return all sets packed together
        return std::make_tuple(X_train, y_train, X_test, y_test);
    }

    Eigen::MatrixXd DataLoader::standardizeFeatures(Eigen::MatrixXd X) {
        // Calculate mean of each column and subtract it from each element in that column to center the data at 0
        auto means = X.colwise().mean();
        Eigen::MatrixXd centered = X.rowwise() - means;

        // Calculate the standard deviation of each column and divide each element in that column to scale it.
        Eigen::RowVectorXd std_devs = ((centered.array().square().colwise().sum()) / (centered.rows() - 1)).sqrt();
        for (int i = 0; i < std_devs.cols(); i++) {
            if (std_devs(i) == 0) {
                std_devs(i) = 1;
            }
        }
        Eigen::MatrixXd standardized = (centered.array().rowwise()) / std_devs.array();

        return standardized;
    }
} // end of namespace
