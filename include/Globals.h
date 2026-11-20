#pragma once
#include <Eigen/Dense>

extern int nthreads;

//DEFINE CUSTOM REDUCTION FOR EIGEN MATRICES
#pragma omp declare reduction(+ : Eigen::MatrixXd : omp_out += omp_in) initializer(omp_priv = Eigen::MatrixXd::Zero(omp_orig.rows(), omp_orig.cols()))