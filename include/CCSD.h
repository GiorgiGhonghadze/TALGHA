#pragma once
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <vector>
#include <string>
#include "DIIS.h"

using Eigen::Tensor;
using Eigen::MatrixXd;

class CCSD {
public:

    CCSD(const std::vector<libint2::Atom>& input_atoms, const std::string& basisSet);
      
    
    void compute_ccsd(MatrixXd &C, Eigen::VectorXd &eps, double &E_scf);
    
private:

    std::vector<libint2::Atom> atoms;
    std::string basis;
    libint2::BasisSet obs;
    int nbf, total_electrons, n_occ, n_vir;
    double E_corr, residual_thresh, residual;
    
    
    Eigen::VectorXd f;
    Tensor<double, 4> ERI_AO, ERI_MO, V;
    Tensor<double, 4> V_oooo, V_vvvv, V_oovv, V_ovvo, V_ooov, V_ovvv;
    //AMPLITUDES
    Tensor<double, 2> t1;
    Tensor<double, 4> t2, tau, tau_tilde;
    //INTERMIDIATES
    Tensor<double, 2> F_me, F_mi, F_ae;
    Tensor<double, 4> W_mnij, W_abef, W_mbej;
    //RESIDUALS
    Tensor<double, 2> R1;
    Tensor<double, 4> R2;
    

    void init_integrals(MatrixXd &C, Eigen::VectorXd &eps);
    void init_amplitudes();
    void compute_intermediates();
    void compute_residuals(DIIS &DIISsolver);
    void evaluate_energy();
    
}; 









