#include "LogisticRegression.hpp"
#include <iostream>
#include <cmath>

namespace neu {
    Eigen::MatrixXd LogisticRegression::sigmoid(Eigen::MatrixXd &array) {
        // Sigmoid equation
        return 1.0 / (1.0 + (-array).array().exp());
    }

    Eigen::VectorXd
    LogisticRegression::gradient_cost(const Eigen::VectorXd &theta, Eigen::MatrixXd X, const Eigen::VectorXd &y) {
        Eigen::MatrixXd z = X * theta;
        return (1.0 / X.rows()) * (X.transpose() * (sigmoid(z) - y));
    }

    void
    LogisticRegression::fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &y, Eigen::VectorXd theta,
                                 double alpha,
                                 int iterations) {
        Eigen::VectorXd theta0 = theta;
        Eigen::MatrixXd theta_new;
        for (int i = 0; i < iterations; i++) {
            theta_new = theta0 - alpha * gradient_cost(theta0, X, y);
            theta0 = theta_new;
        }
        this->coefficients = theta_new;
    }

    int LogisticRegression::predict(Eigen::VectorXd x) {
        double prob = 1.0 / (1.0 + std::exp(-coefficients.dot(x)));
        if (prob >= 0.5){
            return 1;
        }else{
            return 0;
        }
    }
} // end of namespace