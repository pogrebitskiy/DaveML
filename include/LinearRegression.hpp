#ifndef LINEARREGRESSION_HPP
#define LINEARREGRESSION_HPP

#include "DataLoader.hpp"
#include <iostream>
#include <vector>
#include <eigen/Eigen/Dense>
#include <string>

namespace neu {
    struct LinearRegression {
        // Member variable that holds linear model coefficients.
        Eigen::VectorXd coefficients;

        // fit the model using the input data
        void fit(Eigen::MatrixXd &X_train, Eigen::VectorXd &y_train);

        // Use the fitted model to make a prediction
        Eigen::VectorXd predict(Eigen::MatrixXd &X) const;

        // Calculate the RSquared score with the input data
        double score(Eigen::MatrixXd &X_test, Eigen::VectorXd &y_test) const;


    };
} // End of namespace
#endif