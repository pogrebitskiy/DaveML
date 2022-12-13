
#ifndef LOGISTICREGRESSION_HPP
#define LOGISTICREGRESSION_HPP

#include <eigen/Eigen/Dense>

namespace neu {

    struct LogisticRegression {
        Eigen::VectorXd coefficients;

        // Sigmoid function to map values to probabilities
        static Eigen::MatrixXd sigmoid(Eigen::MatrixXd &array);

        // The gradient cost function that we try to minimize
        static Eigen::VectorXd gradient_cost(const Eigen::VectorXd &theta, Eigen::MatrixXd X, const Eigen::VectorXd &y);

        // Runs a batch gradient descent to find coefficients
        void fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &y, Eigen::VectorXd theta,
                 double alpha, int iterations);

        // Using the probability function, makes the binary prediction
        int predict(Eigen::VectorXd x);

    };
}// end of namespace

#endif
