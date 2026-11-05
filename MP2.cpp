#include "MP2.h"
#include "ERI.h"
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <libint2.hpp>
#include <vector>
#include <string>

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

    //CALCULATE ERIs
    ERI ERIcompute(atoms, basis);
    ERIcompute.compute_eri();

    //GET COEFFICIENTS AND ORBITAL ENERGIES
    Eigen::TensorMap<const Eigen::Tensor<double, 2>> C_occ_tensor(C_occ.data(), C_occ.rows(), C_occ.cols());  
    Eigen::TensorMap<const Eigen::Tensor<double, 2>> C_vir_tensor(C_vir.data(), C_vir.rows(), C_vir.cols());  
    
    //TRANSFORM FROM AO TO MO BASIS 
    ERIcompute.motransform_eri(C_occ_tensor, C_vir_tensor, C_occ_tensor, C_vir_tensor);

    
    double E_mp2 = 0.0;
    Eigen::Tensor<double, 4> ijab = ERIcompute.get_contracted_tensor();
    Eigen::Tensor<double, 4> iajb_tensor = ERIcompute.get_contracted_tensor();


    //EVALUATE MP2 CORRELATION
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

    cout << "\n" << "MP2 CORRELATION ENERGY IS " << E_mp2 << " Eh\n";
    cout << "FINAL MP2 ENERGY IS " << Escf + E_mp2 << " Eh\n";

}