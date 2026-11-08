#include "HF.h"
#include "DIIS.h"
#include <iostream>
#include <libint2.hpp>
#include <Eigen/Dense>
#include <chrono>

using std::cout;
using std::string;
using std::vector;
using Eigen::MatrixXd;
using namespace std::chrono;

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
            
            double dx = (atoms[i].x - atoms[j].x);
            double dy = (atoms[i].y - atoms[j].y);
            double dz = (atoms[i].z - atoms[j].z);
            
            double distance = std::sqrt(pow(dx,2) + pow(dy,2) + pow(dz,2));
            
            Enuc += (atoms[i].atomic_number * atoms[j].atomic_number) / distance;
        }
    }
    return Enuc;
}

void HF::compute_1e() {

    
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

}

void HF::compute_JK() {


    libint2::Engine eri_engine(libint2::Operator::coulomb, obs.max_nprim(), obs.max_l());
    const auto& buf_eri = eri_engine.results();
    auto shell2bf = obs.shell2bf(); 
    int nshells = obs.size();

    J_mat.setZero();
    K_mat.setZero();

    Eigen::MatrixXd P_shell = Eigen::MatrixXd::Zero(nshells, nshells);
    for (size_t s1 = 0; s1 < nshells; ++s1) {
        size_t bf1 = shell2bf[s1];
        size_t n1  = obs[s1].size();
        for (size_t s2 = 0; s2 <= s1; ++s2) {
            size_t bf2 = shell2bf[s2];
            size_t n2  = obs[s2].size();

            //MAXIMUM DENISTY ELEMENT IN THE BLOCK
            double max_p = P_mat.block(bf1, bf2, n1, n2).cwiseAbs().maxCoeff();
            P_shell(s1, s2) = max_p;
            P_shell(s2, s1) = max_p;
        }
    }

    for (size_t s1 = 0; s1 < obs.size(); ++s1) {
    size_t bf1_start = shell2bf[s1]; 
    size_t n1 = obs[s1].size();      

        //S2 ONLY GOES UPTO S1 TO ENFORCE SYMMETRY IN PAIR 1
        for (size_t s2 = 0; s2 <= s1; ++s2) {
            size_t bf2_start = shell2bf[s2];
            size_t n2 = obs[s2].size();
        
            //S3 ONLY GOES UPTO S1 TO ENFORCE SWAPPING PAIR SYMMETRY
            for (size_t s3 = 0; s3 <= s1; ++s3) {
                size_t bf3_start = shell2bf[s3];
                size_t n3 = obs[s3].size();
                double q12 = Q_mat(s1, s2);
                
                //S4 ONLY GOES UPTO S3 TO ENFORCE SYMMETRY IN PAIR 2 OR UPTO S2 IF S3==S1
                size_t s4_max = (s3 == s1) ? s2 : s3;
                for (size_t s4 = 0; s4 <= s4_max; ++s4) {
                    size_t bf4_start = shell2bf[s4];
                    size_t n4 = obs[s4].size();

                    double p_max = std::max({
                        4.0 * P_shell(s1, s2),
                        4.0 * P_shell(s3, s4),
                        P_shell(s1, s3),
                        P_shell(s1, s4),
                        P_shell(s2, s3),
                        P_shell(s2, s4)
                    });

                    //CHECK MAXIMUM BOUNDS FOR PAIR 1 AND PAIR 2 SCALED WITH THE MAXIMUM ELEMENT OF DENSITY
                    double max_contribution_estimate = p_max * q12 * Q_mat(s3, s4);

                    if (max_contribution_estimate < screening_thresh) continue;
                      
                    eri_engine.compute(obs[s1], obs[s2], obs[s3], obs[s4]);
                    const double* quartet_buffer = buf_eri[0]; 
                    if (quartet_buffer == nullptr) continue; 
                
                    size_t index = 0;
                    for (size_t f1 = 0; f1 < n1; ++f1) {
                        size_t mu = bf1_start + f1;
                        for (size_t f2 = 0; f2 < n2; ++f2) {
                            size_t nu = bf2_start + f2;

                            if (mu < nu) {
                                index += n3 * n4; 
                                continue;
                            }
                            int p12 = mu * (mu + 1) / 2 + nu;

                            for (size_t f3 = 0; f3 < n3; ++f3) {
                                size_t lam = bf3_start + f3;
                                for (size_t f4 = 0; f4 < n4; ++f4, ++index) {
                                    size_t sig = bf4_start + f4;
                                
                                    
                                    if (lam < sig) continue;
                                    int p34 = lam * (lam + 1) / 2 + sig;
                                    if (s1 == s3 && s2 == s4 && p12 < p34) continue;

                                    double ERI_value = quartet_buffer[index];

                                    //SCALE BY 0.5 IF MU==NU TO AVOID DOUBLE COUNTING
                                    double factor = 1.0;
                                    if (mu == nu)   factor *= 0.5;
                                    if (lam == sig) factor *= 0.5;
                                    if (p12 == p34) factor *= 0.5;

                                    double val = ERI_value * factor;

                                    J_mat(mu, nu) += val * P_mat(lam, sig);
                                    J_mat(nu, mu) += val * P_mat(lam, sig);
                                    J_mat(mu, nu) += val * P_mat(sig, lam);
                                    J_mat(nu, mu) += val * P_mat(sig, lam);

                                    K_mat(mu, lam) += val * P_mat(nu, sig);
                                    K_mat(nu, lam) += val * P_mat(mu, sig);
                                    K_mat(mu, sig) += val * P_mat(nu, lam);
                                    K_mat(nu, sig) += val * P_mat(mu, lam);

                                    //SWAPPED ELECTRON PAIRS
                                    J_mat(lam, sig) += val * P_mat(mu, nu);
                                    J_mat(sig, lam) += val * P_mat(mu, nu);
                                    J_mat(lam, sig) += val * P_mat(nu, mu);
                                    J_mat(sig, lam) += val * P_mat(nu, mu);

                                    K_mat(lam, mu) += val * P_mat(sig, nu);
                                    K_mat(sig, mu) += val * P_mat(lam, nu);
                                    K_mat(lam, nu) += val * P_mat(sig, mu);
                                    K_mat(sig, nu) += val * P_mat(lam, mu);
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

    //TIME 
    auto start = high_resolution_clock::now();
    //TIME

    //COMPUTE NUCLEAR-NUCLEAR REPULSION
    Enuc = compute_Vnn(atoms);

    //COMPUTE ALL ONE-ELECTRON INTEGRALS
    compute_1e();

    //COMPUTE Q_mat FOR SCHWARTZ PRESCREENING 
    Q_mat = Eigen::MatrixXd::Zero(obs.size(), obs.size());
    Q_mat.setZero();

    libint2::Engine engine_2center(libint2::Operator::coulomb, obs.max_nprim(), obs.max_l());
    const auto& buf_eri = engine_2center.results();
    auto shell2bf = obs.shell2bf(); 


    for (size_t s1 = 0; s1 < obs.size(); ++s1) {
        for (size_t s2 = 0; s2 <= s1; ++s2) {
            engine_2center.compute(obs[s1], obs[s2], obs[s1], obs[s2]);
            const double* buf = buf_eri[0];
            if (buf == nullptr) continue;

            double max_val = 0.0;
            size_t n_integrals = obs[s1].size() * obs[s2].size() * obs[s1].size() * obs[s2].size();
            for (size_t i = 0; i < n_integrals; ++i) {
                max_val = std::max(max_val, std::abs(buf[i]));
            }

            double q_val = std::sqrt(max_val);
            Q_mat(s1, s2) = q_val;
            Q_mat(s2, s1) = q_val;
        }
    }
    //DIAGONALIZE OVERLAP MATRIX AND CONSTRUCT TRANSFORMATION MATRIX X
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> Ssolver(S_mat);
    if (Ssolver.info() != Eigen::Success) abort();
    X_mat = Ssolver.eigenvectors() * Ssolver.eigenvalues().cwiseSqrt().cwiseInverse().asDiagonal() *  Ssolver.eigenvectors().transpose();

    //cout <<  "\n" << X_mat << "\n";
        
    //INITIAL GUESS FOR THE FOCK MATRIX - Hcore
    Fock_mat = X_mat.transpose() * Hcore_mat * X_mat;
    double guess_energy = 0.5 * P_mat.cwiseProduct(Hcore_mat + Fock_mat).sum() + Enuc;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> Focksolver(Fock_mat);
    if (Focksolver.info() != Eigen::Success) abort();

    C_mat = (X_mat * Focksolver.eigenvectors()).leftCols(total_electrons/2);

    P_mat = 2.0 * C_mat * C_mat.transpose();

    //TIME
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);
    double sec, msec = std::modf(duration.count()/1000.0, &sec);

    cout << "Initial Guess Time: " << (int)sec << " sec " << (int)std::round(msec*1000) << " msec\n";
    //TIME
     
    cout << "Initial Guess Energy: "  << guess_energy << "\n";

    int iter = 1;

    if(scf_mode == "DIIS"){
        
        cout << "Turning on DIIS.." << "\n";

    }

    DIIS DIISsolver(7);

    while (abs(Eprev-Ecurr) > thresh){


        Eprev = Ecurr;

        //TIME 
        auto start = high_resolution_clock::now();
        //TIME

        //COMPUTE COLOUMB AND EXCHANGE INTEGRALS
        compute_JK();

        //TIME
        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(stop - start);
        double sec, msec = std::modf(duration.count()/1000.0, &sec);

        cout << "\nJ and K Integral Time: " << (int)sec << " sec " << (int)std::round(msec*1000) << " msec\n";
        //TIME

        //CONSTRUCT THE FOCK MATRIX
        Fock_mat = Hcore_mat + J_mat - 0.5 * K_mat;

        if(scf_mode == "DIIS"){

            //TIME 
            auto start = high_resolution_clock::now();
            //TIME
            Eigen::VectorXd flat_vector = Fock_mat.reshaped<Eigen::RowMajor>();
            Eigen::VectorXd error = ( X_mat.transpose() * (Fock_mat * P_mat * S_mat - S_mat * P_mat * Fock_mat) * X_mat).reshaped<Eigen::RowMajor>();
            DIISsolver.calculate_coefficients(flat_vector, error);

            Fock_mat = DIISsolver.get_flattened().reshaped<Eigen::RowMajor>(nbf, nbf);

            Ecurr = 0.5 * P_mat.cwiseProduct(Hcore_mat + Fock_mat).sum() + Enuc;

            Fock_mat_o = X_mat.transpose() * Fock_mat * X_mat;

            //DIAGONILIZE FOCK MATRIX AND BUILD NEW DENSITY
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> Focksolver(Fock_mat_o);
            if (Focksolver.info() != Eigen::Success) abort();
            C_mat = (X_mat * Focksolver.eigenvectors()).leftCols(total_electrons/2);
            C = X_mat * Focksolver.eigenvectors();
            P_mat_new = 2.0 * C_mat * C_mat.transpose();

            if(abs(Eprev-Ecurr) > 1) {

                damp = 0.7;

            }else{

                damp = 0;
            }

            P_mat = damp * P_mat + (1.0-damp) * P_mat_new; 

            //TIME
            auto stop = high_resolution_clock::now();
            auto duration = duration_cast<milliseconds>(stop - start);
            double sec, msec = std::modf(duration.count()/1000.0, &sec);

            cout << "\nFock Matrix Diagonlization Time: " << (int)sec << " sec " << (int)std::round(msec*1000) << " msec\n";
            //TIME

            
            cout << "Iteration " << iter << ": "  << Ecurr << "\n";

            Eorb = Focksolver.eigenvalues();
            C_occ = (X_mat * Focksolver.eigenvectors()).leftCols(total_electrons/2);
            C_vir = (X_mat * Focksolver.eigenvectors()).rightCols(nbf - total_electrons/2);
            
        }else{
            damp = 0.7;
            Ecurr = 0.5 * P_mat.cwiseProduct(Hcore_mat + Fock_mat).sum() + Enuc;

            Fock_mat_o = X_mat.transpose() * Fock_mat * X_mat;

            //DIAGONILIZE FOCK MATRIX AND BUILD NEW DENSITY
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> Focksolver(Fock_mat_o);
            if (Focksolver.info() != Eigen::Success) abort();
            C_mat = (X_mat * Focksolver.eigenvectors()).leftCols(total_electrons/2);
            C = X_mat * Focksolver.eigenvectors();
            P_mat_new = 2.0 * C_mat * C_mat.transpose();
            P_mat = damp * P_mat + (1.0-damp) * P_mat_new;

           
            cout << "Iteration " << iter << ": "  << Ecurr << "\n";

            Eorb = Focksolver.eigenvalues();
            C_occ = (X_mat * Focksolver.eigenvectors()).leftCols(total_electrons/2);
            C_vir = (X_mat * Focksolver.eigenvectors()).rightCols(nbf - total_electrons/2);

        }
        
        iter++;
    }
}

