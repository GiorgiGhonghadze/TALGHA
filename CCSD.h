#pragma once
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <vector>
#include <string>

using Eigen::Tensor;
using Eigen::MatrixXd;

class CCSD {
public:

    CCSD(const std::vector<libint2::Atom>& input_atoms, const std::string& basisSet);
      
    
    void compute_ccsd(MatrixXd &C, Eigen::VectorXd &eps);
    
private:

    std::vector<libint2::Atom> atoms;
    std::string basis;
    libint2::BasisSet obs;
    int nbf, total_electrons, n_occ, n_vir;
    Eigen::VectorXd f;
    Tensor<double, 4> ERI_AO, ERI_MO, V;
    Tensor<double, 4> V_oooo, V_vvvv, V_oovv, V_ovvo, V_ooov, V_ovvv;
    Tensor<double, 2> t1;
    Tensor<double, 4> t2, tau, tau_tilde;
    Tensor<double, 2> F_me, F_mi, F_ae;
    

    void init_integrals(MatrixXd &C, Eigen::VectorXd &eps);
    void init_amplitudes();
    void compute_intermediates();
    
}; 









