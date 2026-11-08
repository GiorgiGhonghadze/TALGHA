#pragma once
#include <Eigen/Dense>
#include <libint2.hpp>
#include <vector>
#include <string>

class MP2 {
public:

    MP2(const std::vector<libint2::Atom>& input_atoms, const std::string& basisSet);
      
    
    void compute_correlation(double &Escf, int &n_occ, int &n_vir, Eigen::MatrixXd &C_occ, Eigen::MatrixXd &C_vir, Eigen::VectorXd &eps);
    
private:

    std::vector<libint2::Atom> atoms;
    std::string basis;
    libint2::BasisSet obs;
    int nbf, total_electrons;
    
}; 









