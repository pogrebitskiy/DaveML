#include "RidgeRegression.hpp"
#include "DataLoader.hpp"
#include <eigen/Eigen/Dense>

namespace neu {

    void RidgeRegression::fit(Eigen::MatrixXd &X_train, Eigen::VectorXd &y_train, double alpha) {
        // Solving ridge equation using version of normal equation
        this->coefficients =
                (X_train.transpose() * X_train + alpha *
                                                 Eigen::MatrixXd::Identity(X_train.cols(), X_train.cols())).inverse() *
                (X_train.transpose() * y_train);

    }

} // end of namespace