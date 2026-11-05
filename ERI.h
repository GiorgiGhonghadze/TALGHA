#pragma once
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <libint2.hpp>
#include <vector>
#include <string>

using Eigen::MatrixXd;

class ERI {
public:

    ERI(const std::vector<libint2::Atom>& input_atoms, const std::string& basisSet);
      
    
    void compute_eri();
    Eigen::Tensor<double, 4> motransform_eri(const Eigen::Tensor<double, 2> &C1, const Eigen::Tensor<double, 2> &C2, const Eigen::Tensor<double, 2> &C3,const Eigen::Tensor<double, 2> &C4);
    // Eigen::Tensor<double, 4> get_contracted_tensor() const { return contracted_tensor; }
    Eigen::Tensor<double, 4> get_eri() const { return ERI_tensor; }
 
    
private:

    std::vector<libint2::Atom> atoms;
    std::string basis;
    libint2::BasisSet obs;
    int nbf, total_electrons;

    Eigen::Tensor<double, 4> ERI_tensor; 
    // Eigen::Tensor<double, 4> contracted_tensor; 
    
}; 










