#pragma once
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

extern int nthreads;

//DEFINE CUSTOM REDUCTION FOR EIGEN MATRICES
#pragma omp declare reduction(+ : Eigen::MatrixXd : omp_out += omp_in) initializer(omp_priv = Eigen::MatrixXd::Zero(omp_orig.rows(), omp_orig.cols()))

//DEFINE CUSTOM REDUCTION FOR EIGEN RANK 4 TENSOR
#pragma omp declare reduction(+ : Eigen::Tensor<double, 4> : omp_out += omp_in)  initializer(omp_priv = Eigen::Tensor<double, 4>(omp_orig.dimensions()).setZero())