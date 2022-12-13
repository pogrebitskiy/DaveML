#ifndef RIDGEREGRESSION_HPP
#define RIDGEREGRESSION_HPP

#include "LinearRegression.hpp"

namespace neu {

    // Ridge regression inherits from LinearRegression
    struct RidgeRegression : LinearRegression {

        // Function to train the ridge regression (variation of the normal equation)
        void fit(Eigen::MatrixXd &X_train, Eigen::VectorXd &y_train, double alpha);
    };
} //end of namespace

#endif
