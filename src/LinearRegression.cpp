#include "LinearRegression.hpp"
#include "DataLoader.hpp"
#include <eigen/Eigen/Dense>

namespace neu {

    void LinearRegression::fit(Eigen::MatrixXd &X_train, Eigen::VectorXd &y_train) {
        this->coefficients = X_train.fullPivHouseholderQr().solve(y_train);
    }

    Eigen::VectorXd LinearRegression::predict(Eigen::MatrixXd &X) const {
        return (X) * (this->coefficients);
    }

    double LinearRegression::score(Eigen::MatrixXd &X_test, Eigen::VectorXd &y_test) const {

        // Calculate the prediction using the model's fitted coefficients
        Eigen::VectorXd y_hat = this->predict(X_test);

        // Calculate each term in the RSquared Formula
        double numerator = pow((y_test - y_hat).array(), 2).sum();
        double denominator = pow(y_test.array() - y_hat.mean(), 2).sum();

        return 1 - (numerator / denominator);
    }


} // end of namespace