#include "ERI.h"
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <libint2.hpp>
#include <vector>
#include <string>

using Eigen::Tensor;
using Eigen::MatrixXd;
using std::cout;

ERI::ERI(const std::vector<libint2::Atom>& input_atoms, const std::string& basisSet)
 : atoms(input_atoms), basis(basisSet), obs(basisSet, input_atoms) {

    obs.set_pure(true);
    nbf = obs.nbf();
    total_electrons = std::accumulate(atoms.begin(), atoms.end(), 0, [](int sum, const libint2::Atom& atom) { return sum + atom.atomic_number;} ); 

    ERI_tensor = Eigen::Tensor<double, 4>(nbf, nbf, nbf, nbf);

}

void ERI::compute_eri(){

    libint2::initialize();

    libint2::Engine eri_engine(libint2::Operator::coulomb, obs.max_nprim(), obs.max_l());
    const auto& buf_eri = eri_engine.results();
    auto shell2bf = obs.shell2bf(); 

    ERI_tensor.setZero();
    
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
                
                
                    eri_engine.compute(obs[s1], obs[s2], obs[s3], obs[s4]);
                    const double* quartet_buffer = buf_eri[0]; 
                    if (quartet_buffer == nullptr) continue; 
                
                    
                    size_t index = 0;
                    for (size_t f1 = 0; f1 < n1; ++f1) {
                        size_t mu = bf1_start + f1;
                        for (size_t f2 = 0; f2 < n2; ++f2) {
                            size_t nu = bf2_start + f2;
                            for (size_t f3 = 0; f3 < n3; ++f3) {
                                size_t lam = bf3_start + f3;
                                for (size_t f4 = 0; f4 < n4; ++f4, ++index) {
                                    size_t sig = bf4_start + f4;
                                
                                    //integral value (μν|λσ)
                                    double ERI_value = quartet_buffer[index];
                                
                                    ERI_tensor(mu, nu, lam, sig) =  ERI_value;
                                
                                    
                                }
                            }
                        }
                    }
                }
            }
        }
    }

}

void ERI::motransform_eri(const Eigen::Tensor<double, 2> &C1, const Eigen::Tensor<double, 2> &C2, const Eigen::Tensor<double, 2> &C3,const Eigen::Tensor<double, 2> &C4){
    
    
    Eigen::array<Eigen::IndexPair<int>, 1> contract = {
        Eigen::IndexPair<int>(0, 0)
    };


    Eigen::Tensor<double, 4> I1 = ERI_tensor.contract(C1, contract);
    Eigen::Tensor<double, 4> I2 = I1.contract(C2, contract);
    Eigen::Tensor<double, 4> I3 = I2.contract(C3, contract);
    Eigen::Tensor<double, 4> I4 = I3.contract(C4, contract);

    contracted_tensor = I4;

    // cout << "\n" << I4 << "\n";

}