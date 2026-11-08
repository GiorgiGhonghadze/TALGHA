#include "DIIS.h"
#include <iostream>
#include <libint2.hpp>
#include <Eigen/Dense>

using std::cout;
using std::string;
using std::vector;
using Eigen::MatrixXd;

DIIS::DIIS(const int &maxDIIS) : maxDIIS(maxDIIS){

    B_mat.resize(maxDIIS+1,maxDIIS+1);
    B_mat.setOnes();
    RHS.resize(maxDIIS+1);
    RHS.setZero();
}

void DIIS::calculate_coefficients(Eigen::VectorXd &flattened, Eigen::VectorXd &error) {
    flattened_vec.resize(flattened.size());
    flattened_vec.setZero();

    if (history.size() == maxDIIS) {
        history.erase(history.begin());
        errors.erase(errors.begin());
    }

    history.push_back(flattened);
    errors.push_back(error);

    int n = history.size();
    if (n >= 3) {
        Eigen::MatrixXd B_sliced = Eigen::MatrixXd::Ones(n + 1, n + 1);
        B_sliced(n, n) = 0.0;

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                B_sliced(i, j) = errors[i].dot(errors[j]);
            }
        }

        RHS.resize(n + 1);
        RHS.setZero();
        RHS(n) = 1.0;

        coefficients = (B_sliced.colPivHouseholderQr().solve(RHS)).head(n);

        for (int i = 0; i < n; i++) {
            flattened_vec += coefficients(i) * history[i];
        }
    } else {
        flattened_vec = flattened;
    }
}
