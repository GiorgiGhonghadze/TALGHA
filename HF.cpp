#include "HF.h"
#include <iostream>
#include <libint2.hpp>
#include <Eigen/Dense>

using std::cout;
using std::string;
using std::vector;
using Eigen::MatrixXd;

const double CODATA_10_TO_18 = 0.9999999995597658;

HF::HF(const std::vector<libint2::Atom>& input_atoms, const std::string& basisSet)
 : atoms(input_atoms), basis(basisSet), obs(basisSet, input_atoms) {

    obs.set_pure(true);
    nbf = obs.nbf();
    total_electrons = std::accumulate(atoms.begin(), atoms.end(), 0, [](int sum, const libint2::Atom& atom) { return sum + atom.atomic_number;} ); 

    S_mat = MatrixXd::Zero(nbf, nbf);
    T_mat = MatrixXd::Zero(nbf, nbf);
    V_mat = MatrixXd::Zero(nbf, nbf);
    Hcore_mat = MatrixXd::Zero(nbf, nbf);
    Fock_mat = MatrixXd::Zero(nbf, nbf);
    Fock_mat_o = MatrixXd::Zero(nbf, nbf);
    X_mat = MatrixXd::Zero(nbf, nbf);
    C_mat = MatrixXd::Zero(nbf, nbf);
    P_mat = MatrixXd::Zero(nbf, nbf);
    P_mat_new = MatrixXd::Zero(nbf, nbf);
    J_mat = MatrixXd::Zero(nbf, nbf);
    K_mat = MatrixXd::Zero(nbf, nbf);

    C_occ = MatrixXd::Zero(nbf, total_electrons/2);
    C_vir = MatrixXd::Zero(nbf, nbf - total_electrons/2);
    Eorb = Eigen::VectorXd::Zero(nbf);

}

double HF::compute_Vnn(const std::vector<libint2::Atom>& atoms) {
    double Enuc = 0.0;
    int natoms = atoms.size();

    for (int i = 0; i < natoms; i++) {
        for (int j = i + 1; j < natoms; j++) {
            
            double dx = (atoms[i].x - atoms[j].x) * CODATA_10_TO_18;
            double dy = (atoms[i].y - atoms[j].y) * CODATA_10_TO_18;
            double dz = (atoms[i].z - atoms[j].z) * CODATA_10_TO_18;
            
            double distance = std::sqrt(pow(dx,2) + pow(dy,2) + pow(dz,2));
            
            Enuc += (atoms[i].atomic_number * atoms[j].atomic_number) / distance;
        }
    }
    return Enuc;
}

void HF::compute_1e() {

    libint2::initialize();
    
    //OVERLAP INTEGRAL CALC

    libint2::Engine s_engine(libint2::Operator::overlap,  // will compute overlap ints
                obs.max_nprim(),    // max # of primitives in shells this engine will accept
                obs.max_l()         // max angular momentum of shells this engine will accept
               );
    
    auto shell2bf = obs.shell2bf(); // maps shell index to basis function index
                                // shell2bf[0] = index of the first basis function in shell 0
                                // shell2bf[1] = index of the first basis function in shell 1
                                // ...
    const auto& buf_vec_s = s_engine.results(); // will point to computed shell sets

    for(auto s1=0; s1!=obs.size(); ++s1) {
      for(auto s2=0; s2!=obs.size(); ++s2) {

       // cout << "compute shell set {" << s1 << "," << s2 << "} ... ";
        s_engine.compute(obs[s1], obs[s2]);
        //cout << "done" << "\n";
        auto ints_shellset = buf_vec_s[0];  // location of the computed integrals
        if (ints_shellset == nullptr)
            continue;  // nullptr returned if the entire shell-set was screened out

        auto bf1 = shell2bf[s1];  // first basis function in first shell
        auto n1 = obs[s1].size(); // number of basis functions in first shell
        auto bf2 = shell2bf[s2];  // first basis function in second shell
        auto n2 = obs[s2].size(); // number of basis functions in second shell

        // integrals are packed into ints_shellset in row-major (C) form
        // this iterates over integrals in this order
        for(auto f1=0; f1!=n1; ++f1){
          for(auto f2=0; f2!=n2; ++f2){

            S_mat(bf1+f1,bf2+f2) = ints_shellset[f1*n2+f2];

           // cout << "  " << bf1+f1 << " " << bf2+f2 << " " << ints_shellset[f1*n2+f2] << "\n";
            }
        }
      }
    }

    // cout <<  "\n" << S_mat << "\n";

    //KINETIC INTEGRAL CALC

    libint2::Engine t_engine(libint2::Operator::kinetic,  // will compute overlap ints
                obs.max_nprim(),    // max # of primitives in shells this engine will accept
                obs.max_l()         // max angular momentum of shells this engine will accept
               );

    const auto& buf_vec_t = t_engine.results();
    

    for(auto s1=0; s1!=obs.size(); ++s1) {
      for(auto s2=0; s2!=obs.size(); ++s2) {
        t_engine.compute(obs[s1], obs[s2]);
        auto ints_shellset = buf_vec_t[0];  // location of the computed integrals
        if (ints_shellset == nullptr)
            continue;  // nullptr returned if the entire shell-set was screened out

        auto bf1 = shell2bf[s1];  // first basis function in first shell
        auto n1 = obs[s1].size(); // number of basis functions in first shell
        auto bf2 = shell2bf[s2];  // first basis function in second shell
        auto n2 = obs[s2].size(); // number of basis functions in second shell

        // integrals are packed into ints_shellset in row-major (C) form
        // this iterates over integrals in this order
        for(auto f1=0; f1!=n1; ++f1)
          for(auto f2=0; f2!=n2; ++f2)

            T_mat(bf1+f1,bf2+f2) = ints_shellset[f1*n2+f2];
           // cout << "  " << bf1+f1 << " " << bf2+f2 << " " << ints_shellset[f1*n2+f2] << "\n";
      }
    }   

    // cout <<  "\n" << T_mat << "\n";

    //NUCLEAR ATTRACTION INTEGRAL CALC

    libint2::Engine v_engine(libint2::Operator::nuclear, obs.max_nprim(), obs.max_l());
    v_engine.set_params(make_point_charges(atoms));

    const auto& buf_vec_v = v_engine.results();

    for(auto s1=0; s1!=obs.size(); ++s1) {
      for(auto s2=0; s2!=obs.size(); ++s2) {
        v_engine.compute(obs[s1], obs[s2]);
        auto ints_shellset = buf_vec_v[0];  // location of the computed integrals
        if (ints_shellset == nullptr)
            continue;  // nullptr returned if the entire shell-set was screened out

        auto bf1 = shell2bf[s1];  // first basis function in first shell
        auto n1 = obs[s1].size(); // number of basis functions in first shell
        auto bf2 = shell2bf[s2];  // first basis function in second shell
        auto n2 = obs[s2].size(); // number of basis functions in second shell

        // integrals are packed into ints_shellset in row-major (C) form
        // this iterates over integrals in this order
        for(auto f1=0; f1!=n1; ++f1)
          for(auto f2=0; f2!=n2; ++f2)

            V_mat(bf1+f1,bf2+f2) = ints_shellset[f1*n2+f2];
           // cout << "  " << bf1+f1 << " " << bf2+f2 << " " << ints_shellset[f1*n2+f2] << "\n";
      }
    }   

    // cout <<  "\n" << V_mat << "\n";

    Hcore_mat = V_mat + T_mat;

    // cout <<  "\nHcore Matrix";
    // cout <<  "\n" << Hcore_mat << "\n";

    libint2::finalize();

}

void HF::compute_JK() {

    libint2::initialize();

    libint2::Engine eri_engine(libint2::Operator::coulomb, obs.max_nprim(), obs.max_l());
    const auto& buf_eri = eri_engine.results();
    auto shell2bf = obs.shell2bf(); 

    J_mat.setZero();
    K_mat.setZero();
    
    for (size_t s1 = 0; s1 < obs.size(); ++s1) {
        size_t bf1_start = shell2bf[s1]; // First basis function index in shell 1
        size_t n1 = obs[s1].size();      // Number of functions in shell 1
    
        for (size_t s2 = 0; s2 < obs.size(); ++s2) {
            size_t bf2_start = shell2bf[s2];
            size_t n2 = obs[s2].size();
        
            for (size_t s3 = 0; s3 < obs.size(); ++s3) {
                size_t bf3_start = shell2bf[s3];
                size_t n3 = obs[s3].size();
            
                for (size_t s4 = 0; s4 < obs.size(); ++s4) {
                    size_t bf4_start = shell2bf[s4];
                    size_t n4 = obs[s4].size();
                
                    // 2. Instruct Libint2 to compute the entire shell quartet block
                    eri_engine.compute(obs[s1], obs[s2], obs[s3], obs[s4]);// buf[0] points to a flat array of size (n1 * n2 * n3 * n4)
                    const double* quartet_buffer = buf_eri[0]; 
                    if (quartet_buffer == nullptr) continue; // Screen out zero blocks
                
                    // 3. Unpack the buffer into your individual basis function indices
                    size_t index = 0;
                    for (size_t f1 = 0; f1 < n1; ++f1) {
                        size_t mu = bf1_start + f1;
                        for (size_t f2 = 0; f2 < n2; ++f2) {
                            size_t nu = bf2_start + f2;
                            for (size_t f3 = 0; f3 < n3; ++f3) {
                                size_t lam = bf3_start + f3;
                                for (size_t f4 = 0; f4 < n4; ++f4, ++index) {
                                    size_t sig = bf4_start + f4;
                                
                                    // The specific analytical integral value (μν|λσ)
                                    double ERI_value = quartet_buffer[index];
                                
                                    J_mat(mu, nu) += P_mat(lam, sig) * ERI_value;
                                
                                    K_mat(mu, lam) += P_mat(nu, sig) * ERI_value;
                                
                                    // Store ERI_value in your tensor or 2D compound matrix here!
                                }
                            }
                        }
                    }
                }
            }
        }
    }


}

void HF::scf(std::string& scf_mode) {
   
    cout << std::fixed << std::setprecision(10);

    //COMPUTE NUCLEAR-NUCLEAR REPULSION
    Enuc = compute_Vnn(atoms);

    //COMPUTE ALL ONE-ELECTRON INTEGRALS
    compute_1e();

    //DIAGONALIZE OVERLAP MATRIX AND CONSTRUCT TRANSFORMATION MATRIX X
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> Ssolver(S_mat);
    if (Ssolver.info() != Eigen::Success) abort();
    X_mat = Ssolver.eigenvectors() * Ssolver.eigenvalues().cwiseSqrt().cwiseInverse().asDiagonal() *  Ssolver.eigenvectors().transpose();

    //cout <<  "\n" << X_mat << "\n";
        
    //INITIAL GUESS FOR THE FOCK MATRIX - Hcore
    Fock_mat = X_mat.transpose() * Hcore_mat * X_mat;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> Focksolver(Fock_mat);
    if (Focksolver.info() != Eigen::Success) abort();

    C_mat = (X_mat * Focksolver.eigenvectors()).leftCols(total_electrons/2);

    P_mat = 2.0 * C_mat * C_mat.transpose();

    int iter = 1;

    while (abs(Eprev-Ecurr) > thresh){


        Eprev = Ecurr;

        //COMPUTE COLOUMB AND EXCHANGE INTEGRALS
        compute_JK();

        //CONSTRUCT THE FOCK MATRIX
        Fock_mat = Hcore_mat + J_mat - 0.5 * K_mat;
        Fock_mat_o = X_mat.transpose() * Fock_mat * X_mat;

        //DIAGONILIZE FOCK MATRIX AND BUILD NEW DENSITY
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> Focksolver(Fock_mat_o);
        if (Focksolver.info() != Eigen::Success) abort();
        C_mat = (X_mat * Focksolver.eigenvectors()).leftCols(total_electrons/2);
        C = X_mat * Focksolver.eigenvectors();
        P_mat_new = 2.0 * C_mat * C_mat.transpose();
        P_mat = 0.5 * P_mat + 0.5 * P_mat_new;

        Ecurr = 0.5 * P_mat.cwiseProduct(Hcore_mat + Fock_mat).sum() + Enuc;
        cout << "Iteration " << iter << ": "  << Ecurr << "\n";
        
        Eorb = Focksolver.eigenvalues();
        C_occ = (X_mat * Focksolver.eigenvectors()).leftCols(total_electrons/2);
        C_vir = (X_mat * Focksolver.eigenvectors()).rightCols(nbf - total_electrons/2);
        iter++;
    }

}

