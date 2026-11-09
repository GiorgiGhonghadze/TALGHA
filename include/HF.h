#pragma once
#include <Eigen/Dense>
#include <libint2.hpp>
#include <vector>
#include <string>

class HF {
public:

    HF(const std::vector<libint2::Atom>& input_atoms, const std::string& basisSet);
      
    
    void scf(std::string& scf_mode);
    double get_Escf() const {return Ecurr;};

    Eigen::MatrixXd get_C_occ() const { return C_occ; }
    Eigen::MatrixXd get_C_vir() const { return C_vir; }
    Eigen::MatrixXd get_C() const { return C; }
    int get_n_occ() const {return total_electrons/2;}
    int get_n_vir() const {return nbf - total_electrons/2;}
    Eigen::VectorXd get_Eorb() const { return Eorb; }
    
private:

    std::vector<libint2::Atom> atoms;
    std::string basis;
    libint2::BasisSet obs;
    int nbf, total_electrons;
    double Enuc, Eprev = 0, Ecurr = 1, thresh = 1e-9, screening_thresh = 1e-12, damp;

    Eigen::MatrixXd S_mat, T_mat, V_mat, Hcore_mat, Fock_mat, Fock_mat_o, X_mat, C_mat, P_mat, P_mat_new, J_mat, K_mat, C, C_occ, C_vir, Q_mat;
    Eigen::VectorXd Eorb; 

    double compute_Vnn(const std::vector<libint2::Atom>& atoms);
    void compute_1e();
    void compute_JK(libint2::Engine &engine_2e);
    
}; 


