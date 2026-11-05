#include "MP2.h"
#include "ERI.h"
#include "CCSD.h"
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <libint2.hpp>
#include <cmath>
#include <vector>
#include <string>

using Eigen::Tensor;
using Eigen::MatrixXd;
using std::cout;

CCSD::CCSD(const std::vector<libint2::Atom>& input_atoms, const std::string& basisSet)
 : atoms(input_atoms), basis(basisSet), obs(basisSet, input_atoms) {

    obs.set_pure(true);
    nbf = obs.nbf();
    total_electrons = std::accumulate(atoms.begin(), atoms.end(), 0, [](int sum, const libint2::Atom& atom) { return sum + atom.atomic_number;} ); 

}

void CCSD::init_integrals(MatrixXd &C, Eigen::VectorXd &eps){

    //CALCULATE ERIs
    ERI ERIcompute(atoms, basis);
    Eigen::TensorMap<const Eigen::Tensor<double, 2>> C_tensor(C.data(), C.rows(), C.cols());
    
    ERIcompute.compute_eri();
    ERI_AO = ERIcompute.get_eri();
    ERI_MO = ERIcompute.motransform_eri(C_tensor, C_tensor, C_tensor, C_tensor);

    //SHUFFLE THE TENSOR FROM CHEMISTS TO PHYSICISTS
    Eigen::array<int, 4> shuffle_qr = {0, 2, 1, 3};
    Tensor<double, 4> ERI_DIRAC = ERI_MO.shuffle(shuffle_qr);

    //EXPAND THE TENSOR INTO SPIN-ORBITALS
    V.resize(nbf*2, nbf*2, nbf*2, nbf*2);
    V.setZero();
    for (int p = 0; p < nbf; p++){
        for (int q = 0; q < nbf; q++){
            for (int r = 0; r < nbf; r++){
                for (int s = 0; s < nbf; s++){
                    double val = ERI_DIRAC(p, q, r, s);

                    if (abs(val) < 1e-12) continue;

                    int pa = 2*p, pb = 2*p+1;
                    int qa = 2*q, qb = 2*q+1;
                    int ra = 2*r, rb = 2*r+1;
                    int sa = 2*s, sb = 2*s+1;

                    V(pa, qa, ra, sa) = val;
                    V(pb, qb, rb, sb) = val;
                    V(pa, qb, ra, sb) = val;
                    V(pb, qa, rb, sa) = val;
                }
            }
        }    
    }
    
    //ANTISYMETRIZE INTEGRALS
    Eigen::array<int, 4> anti = {0, 1, 3, 2};
    V = (V - V.shuffle(anti)).eval();

    //EXPAND ORBITAL ENERGY VECTOR INTO SPIN-ORBITALS
    f.resize(2*nbf);
    f.setZero();

    for(int i = 0; i < nbf; i++){

        f(i*2) = eps(i);
        f(i*2+1) = eps(i);

    }
    
}

void CCSD::init_amplitudes(){

    n_occ = total_electrons;
    n_vir = nbf*2 - total_electrons;


    //SLICE TENSOR INTO SUBBLOCKS
    Eigen::array<Eigen::Index, 4> offsets = {0, 0, 0, 0};
    Eigen::array<Eigen::Index, 4> extents = {n_occ, n_occ, n_occ, n_occ};

    V_oooo = V.slice(offsets, extents);

    offsets = {n_occ, n_occ, n_occ, n_occ};
    extents = {n_vir, n_vir, n_vir, n_vir};

    V_vvvv = V.slice(offsets, extents);

    offsets = {0, 0, n_occ, n_occ};
    extents = {n_occ, n_occ, n_vir, n_vir};

    V_oovv = V.slice(offsets, extents);

    offsets = {0, n_occ, n_occ, 0};
    extents = {n_occ, n_vir, n_vir, n_occ};

    V_ovvo = V.slice(offsets, extents);

    offsets = {0, 0, 0, n_occ};
    extents = {n_occ, n_occ, n_occ, n_vir};

    V_ooov = V.slice(offsets, extents);

    offsets = {0, n_occ,  n_occ, n_occ};
    extents = {n_occ, n_vir, n_vir, n_vir};

    V_ovvv = V.slice(offsets, extents);


    //INITIALIZE AMPLITUDES
    t1.resize(n_occ, n_vir);
    t2.resize(n_occ, n_occ, n_vir, n_vir);
    t1.setZero();
    t2.setZero();

    for (int i = 0; i < n_occ; i++) {
        for (int j = 0; j < n_occ; j++) {
            for (int a = 0; a < n_vir; a++) {
                for (int b = 0; b < n_vir; b++) {

                    double num = V_oovv(i, j, a, b);
                    double denom = f(i) + f(j) - f(n_occ + a) - f(n_occ + b);


                        t2(i, j, a, b) = num / denom;
                    
                }
            }
        }
    }

    Eigen::array<Eigen::IndexPair<int>, 4> contract_all = {
    Eigen::IndexPair<int>(0, 0), // i with i
    Eigen::IndexPair<int>(1, 1), // j with j
    Eigen::IndexPair<int>(2, 2), // a with a
    Eigen::IndexPair<int>(3, 3)  // b with b
    };

    //
    // Eigen::Tensor<double, 0> emp2_tensor = V_oovv.contract(t2, contract_all); 
    // double e_mp2 = 0.25 * emp2_tensor();

}

void CCSD::compute_intermediates(){


    //CALCULATE EFFECTIVE AMPLITUDES
    tau.resize(n_occ, n_occ, n_vir, n_vir);
    tau_tilde.resize(n_occ, n_occ, n_vir, n_vir);
    tau.setZero(); 
    tau_tilde.setZero(); 

    for (int i = 0; i < n_occ; i++) {
        for (int j = 0; j < n_occ; j++) {
            for (int a = 0; a < n_vir; a++) {
                for (int b = 0; b < n_vir; b++) {

                    tau(i, j, a, b) = t2(i, j, a, b) +  t1(i, a) * t1(j, b) - t1(i, b) * t1(j, a);
                    tau_tilde(i, j, a, b) = t2(i, j, a, b) +  0.5 * (t1(i, a) * t1(j, b) - t1(i, b) * t1(j, a));
                    
                }
            }
        }
    }


    //CALCULATE F INTERMIDIATES
    F_me.resize(n_occ, n_vir);
    F_me.setZero();

    for (int m = 0; m < n_occ; m++) {
        for (int n = 0; n < n_occ; n++) {
            for (int e = 0; e < n_vir; e++) {
                for (int f = 0; f < n_vir; f++) {

                    F_me(m,e) += t1(n, f) * V_oovv(m, n, e, f);
                    
                }
            }
        }
    }


    F_mi.resize(n_occ, n_occ);
    F_mi.setZero();

    for (int m = 0; m < n_occ; m++) {
        for (int i = 0; i < n_occ; i++) {
            double value = 0.0;
            for (int e = 0; e < n_vir; e++) {
                value += t1(i, e) * F_me(m,e);
                for (int n = 0; n < n_occ; n++){
                    value += t1(n,e) * V_ooov(m, n, i, e);
                    for (int f = 0; f < n_vir; f++) {

                        value += 0.5 * tau_tilde(i, n, e, f) * V_oovv(m,n,e,f);

                    }
                }
            }
            F_mi(m,i) = value;
        }
    }

    F_ae.resize(n_vir, n_vir);
    F_ae.setZero();

    for (int a = 0; a < n_vir; a++) {
        for (int e = 0; e < n_vir; e++) {
            double value = 0.0;
            for (int m = 0; m < n_occ; m++) {
                value -= t1(m, a) * F_me(m,e);
                for (int f = 0; f < n_vir; f++){
                    value -= t1(m,f) * V_ovvv(m, a, e, f);
                    for (int n = 0; n < n_occ; n++) {

                        value -= 0.5 * tau_tilde(m, n, a, f) * V_oovv(m,n,e,f);

                    }
                }
            }
            F_ae(a,e) = value;
        }
    }

    //CALCULATE W INTERMIDIATES
    W_mnij.resize(n_occ, n_occ, n_occ, n_occ);
    W_mnij.setZero();

    for (int m = 0; m < n_occ; m++) {
        for (int n = 0; n < n_occ; n++) {
            for (int i = 0; i < n_occ; i++) {
                for (int j = 0; j < n_occ; j++) {
                    double value = V_oooo(m,n,i,j);

                    for (int e = 0; e < n_vir; e++) {

                        value += t1(j,e) * V_ooov(m,n,i,e);
                        value -= t1(i,e) * V_ooov(m,n,j,e);

                    } 

                    for (int e = 0; e < n_vir; e++) {
                        for (int f = 0; f < n_vir; f++) {

                            value += 0.25 * tau(i,j,e,f) * V_oovv(m,n,e,f);
                    
                        }
                    }   
                    
                    W_mnij(m,n,i,j) = value;
                }
            }
        }
    }



    W_abef.resize(n_vir, n_vir, n_vir, n_vir);
    W_abef.setZero();

    for (int a = 0; a < n_vir; a++) {
        for (int b = 0; b < n_vir; b++) {
            for (int e = 0; e < n_vir; e++) {
                for (int f = 0; f < n_vir; f++) {
                    double value = V_vvvv(a,b,e,f);

                    for (int m = 0; m < n_occ; m++) {
                        
                        value += t1(m,b) * V_ovvv(m,a,e,f);
                        value -= t1(m,a) * V_ovvv(m,b,e,f);
                        
                    } 

                    for (int m = 0; m < n_occ; m++) {
                        for (int n = 0; n < n_occ; n++) {

                            value += 0.25 * tau(m,n,a,b) * V_oovv(m,n,e,f);
                    
                        }
                    }   
                    
                    W_abef(a,b,e,f) = value;
                }
            }
        }
    }

    W_mbej.resize(n_occ, n_vir, n_vir, n_occ);
    W_mbej.setZero();

    for (int m = 0; m < n_occ; m++) {
        for (int b = 0; b < n_vir; b++) {
            for (int e = 0; e < n_vir; e++) {
                for (int j = 0; j < n_occ; j++) {
                    double value = V_ovvo(m,b,e,j);

                    for (int f = 0; f < n_vir; f++) {
                    
                        value += t1(j,f) * V_ovvv(m,b,e,f);
                        
                    } 
                    
                    for (int n = 0; n < n_occ; n++) {
                        
                        value += t1(n,b) * V_ooov(m,n,j,e);
                        
                    } 

                    for (int f = 0; f < n_vir; f++) {
                        for (int n = 0; n < n_occ; n++) {

                            value += (0.5*t2(j,n,f,b) + t1(j,f)*t1(n,b) ) * V_oovv(m,n,e,f);
                    
                        }
                    }   
                    
                    W_mbej(m,b,e,j) = value;
                }
            }
        }
    }



    // cout << "\n" << F_mi(0,0) << "\n";
    // cout << "\n" << F_ae(0,0) << "\n";
    // cout << "\n" << W_mnij(0,1,0,1) << "\n";
    // cout << "\n" << W_abef(0,1,0,1) << "\n";
    // cout << "\n" << W_mbej(0,0,0,0) << "\n";



    // std::cout << "Shape: [ ";
    // for (auto dim : tau.dimensions()) {
    //     std::cout << dim << " ";
    // }
    // std::cout << "]\n";

}

void CCSD::compute_ccsd(MatrixXd &C, Eigen::VectorXd &eps){


    init_integrals(C, eps);
    init_amplitudes();
    compute_intermediates();
    // cout << "\n" << "MP2 CORRELATION ENERGY IS " << E_mp2 << " Eh\n";
    // cout << "FINAL MP2 ENERGY IS " << Escf + E_mp2 << " Eh\n";

}

