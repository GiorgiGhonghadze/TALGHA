#pragma once
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <vector>
#include <string>

using std::vector;

class DIIS {
public:

    DIIS(const int &maxDIIS);
      
    
    void calculate_coefficients(Eigen::VectorXd &flattened, Eigen::VectorXd &error);
    Eigen::VectorXd get_flattened() const { return flattened_vec; }

    
private:

    int maxDIIS, size_active;

    Eigen::array<Eigen::Index, 2> offsets, extents;

    vector<Eigen::VectorXd> history, errors;
    Eigen::MatrixXd B_mat; 
    Eigen::VectorXd RHS, coefficients, flattened_vec;

}; 


