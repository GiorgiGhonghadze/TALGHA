#include "Globals.h"
#include "MP2.h"
#include "ERI.h"
#include <libint2.hpp>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <cblas.h>
#include <vector>
#include <string>
#include <chrono>

using namespace std::chrono;
using Eigen::Tensor;
using Eigen::MatrixXd;
using std::cout;


MP2::MP2(const std::vector<libint2::Atom>& input_atoms, const std::string& basisSet)
 : atoms(input_atoms), basis(basisSet), obs(basisSet, input_atoms) {

    obs.set_pure(true);
    nbf = obs.nbf();
    total_electrons = std::accumulate(atoms.begin(), atoms.end(), 0, [](int sum, const libint2::Atom& atom) { return sum + atom.atomic_number;} ); 

}

void MP2::compute_correlation(double &Escf, int &n_occ, int &n_vir, Eigen::MatrixXd &C_occ, Eigen::MatrixXd &C_vir, Eigen::VectorXd &eps){

    openblas_set_num_threads(nthreads);

    //TIME
    auto start = high_resolution_clock::now();
    //TIME
    
    //CALCULATE ERIs
    ERI ERIcompute(atoms, basis);
    ERIcompute.compute_eri();

    //TIME
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);
    double sec, msec = std::modf(duration.count()/1000.0, &sec);

    cout << "ERI Time: " << (int)sec << " sec " << (int)std::round(msec*1000) << " msec\n";
    //TIME

    //GET COEFFICIENTS AND ORBITAL ENERGIES
    Eigen::TensorMap<const Eigen::Tensor<double, 2>> C_occ_tensor(C_occ.data(), C_occ.rows(), C_occ.cols());  
    Eigen::TensorMap<const Eigen::Tensor<double, 2>> C_vir_tensor(C_vir.data(), C_vir.rows(), C_vir.cols());  
    
    //TIME
    start = high_resolution_clock::now();
    //TIME

    //TRANSFORM FROM AO TO MO BASIS 
    double E_mp2 = 0.0;
    Eigen::Tensor<double, 4> iajb_tensor = ERIcompute.motransform_eri(C_occ_tensor, C_vir_tensor, C_occ_tensor, C_vir_tensor);

    //TIME
    stop = high_resolution_clock::now();
    duration = duration_cast<milliseconds>(stop - start);
    sec, msec = std::modf(duration.count()/1000.0, &sec);

    cout << "Partial Transformation Time: " << (int)sec << " sec " << (int)std::round(msec*1000) << " msec\n";
    //TIME


    //TIME
    start = high_resolution_clock::now();
    //TIME

    //EVALUATE MP2 CORRELATION
    #pragma omp parallel
    {
        #pragma omp for schedule(dynamic) reduction(+:E_mp2)
        for (int i = 0; i < n_occ; ++i) {
            for (int a = 0; a < n_vir; ++a) {
                for (int j = 0; j < n_occ; ++j) {
                    for (int b = 0; b < n_vir; ++b) {
                    
                        double iajb = iajb_tensor(i, a, j, b);
                        double ibja = iajb_tensor(i, b, j, a);


                    
                        double denom = eps(i) + eps(j) - eps(n_occ + a) - eps(n_occ + b);
                    
                        E_mp2 += iajb * (2.0 * iajb - ibja) / denom;
                    }
                }
            }
        }
    }

    //TIME
    stop = high_resolution_clock::now();
    duration = duration_cast<milliseconds>(stop - start);
    sec, msec = std::modf(duration.count()/1000.0, &sec);

    cout << "MP2 Correlation Time: " << (int)sec << " sec " << (int)std::round(msec*1000) << " msec\n";
    //TIME

    cout << "\n" << "MP2 CORRELATION ENERGY IS " << E_mp2 << " Eh\n";
    cout << "FINAL MP2 ENERGY IS " << Escf + E_mp2 << " Eh\n";

}