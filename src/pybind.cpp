#include "RidgeRegression.hpp"
#include "DataLoader.hpp"
#include "LinearRegression.hpp"
#include "LogisticRegression.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

namespace py = pybind11;

PYBIND11_MODULE(DaveML, m){
    m.doc() = "Welcome to my simple ML library implemented in C++";

    // DataLoader Class
    py::class_<neu::DataLoader>(m, "DataLoader")
            .def(py::init<std::string, std::string, bool, int>())
            .def("getX", &neu::DataLoader::getX)
            .def("gety", &neu::DataLoader::gety)
            .def("add_constant", &neu::DataLoader::add_constant)
            .def("standardizeFeatures", &neu::DataLoader::standardizeFeatures)
            .def("trainTestSplit", &neu::DataLoader::trainTestSplit);

    // LinearRegression Class
    py::class_<neu::LinearRegression>(m, "LinearRegression")
            .def(py::init<>())
            .def_readonly("coefficients", &neu::LinearRegression::coefficients)
            .def("fit", &neu::LinearRegression::fit)
            .def("predict", &neu::LinearRegression::predict)
            .def("score", &neu::LinearRegression::score);


    // RidgeRegression Class inheriting from LinearRegression
    py::class_<neu::RidgeRegression, neu::LinearRegression>(m, "RidgeRegression")
            .def(py::init<>())
            .def("fit", &neu::RidgeRegression::fit);


    // LogisticRegression Class
    py::class_<neu::LogisticRegression>(m, "LogisticRegression")
            .def(py::init<>())
            .def_readonly("coefficients", &neu::LogisticRegression::coefficients)
            .def("sigmoid", &neu::LogisticRegression::sigmoid)
            .def("gradient_cost", &neu::LogisticRegression::gradient_cost)
            .def("fit", &neu::LogisticRegression::fit)
            .def("predict", &neu::LogisticRegression::predict);
}
