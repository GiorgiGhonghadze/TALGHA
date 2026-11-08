#include "ERI.h"
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <libint2.hpp>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <iostream>

using Eigen::Tensor;
using Eigen::MatrixXd;
using std::cout;

ERI::ERI(const std::vector<libint2::Atom>& input_atoms, const std::string& basisSet)
    : atoms(input_atoms), basis(basisSet), obs(basisSet, input_atoms) {

    obs.set_pure(true);
    nbf = obs.nbf();
    total_electrons = std::accumulate(atoms.begin(), atoms.end(), 0,[](int sum, const libint2::Atom& atom) { return sum + atom.atomic_number; });

    ERI_tensor = Eigen::Tensor<double, 4>(nbf, nbf, nbf, nbf);
    Q_mat = Eigen::MatrixXd::Zero(obs.size(), obs.size());
}

void ERI::compute_eri() {
    libint2::Engine eri_engine(libint2::Operator::coulomb, obs.max_nprim(), obs.max_l());
    const auto& buf_eri = eri_engine.results();
    auto shell2bf = obs.shell2bf();

    ERI_tensor.setZero();
    Q_mat.setZero();


    for (size_t s1 = 0; s1 < obs.size(); ++s1) {
        size_t n1 = obs[s1].size();
        for (size_t s2 = 0; s2 <= s1; ++s2) {
            size_t n2 = obs[s2].size();

         
            eri_engine.compute(obs[s1], obs[s2], obs[s1], obs[s2]);
            const double* diag_buf = buf_eri[0];

            if (diag_buf == nullptr) {
                Q_mat(s1, s2) = 0.0;
                Q_mat(s2, s1) = 0.0;
                continue;
            }

            double max_val = 0.0;
            for (size_t f1 = 0; f1 < n1; ++f1) {
                for (size_t f2 = 0; f2 < n2; ++f2) {
                    // Row-major offset for (f1, f2, f3 = f1, f4 = f2)
                    size_t diag_idx = ((f1 * n2 + f2) * n1 + f1) * n2 + f2;
                    double val = diag_buf[diag_idx];
                    if (val > max_val) {
                        max_val = val;
                    }
                }
            }

            double q_bound = std::sqrt(std::max(0.0, max_val));
            Q_mat(s1, s2) = q_bound;
            Q_mat(s2, s1) = q_bound;
        }
    }


    for (size_t s1 = 0; s1 < obs.size(); ++s1) {
        size_t bf1_start = shell2bf[s1];
        size_t n1 = obs[s1].size();

        for (size_t s2 = 0; s2 <= s1; ++s2) {
            size_t bf2_start = shell2bf[s2];
            size_t n2 = obs[s2].size();
            double q12 = Q_mat(s1, s2);

            for (size_t s3 = 0; s3 <= s1; ++s3) {
                size_t s4_max = (s3 == s1) ? s2 : s3;

                for (size_t s4 = 0; s4 <= s4_max; ++s4) {

                    double max_bound = q12 * Q_mat(s3, s4);
                    if (max_bound < screening_thresh) {
                        continue;
                    }

                    size_t bf3_start = shell2bf[s3];
                    size_t n3 = obs[s3].size();
                    size_t bf4_start = shell2bf[s4];
                    size_t n4 = obs[s4].size();

                    eri_engine.compute(obs[s1], obs[s2], obs[s3], obs[s4]);
                    const double* quartet_buffer = buf_eri[0];
                    if (quartet_buffer == nullptr) {
                        continue;
                    }

                    size_t index = 0;
                    for (size_t f1 = 0; f1 < n1; ++f1) {
                        size_t mu = bf1_start + f1;

                        for (size_t f2 = 0; f2 < n2; ++f2) {
                            size_t nu = bf2_start + f2;

                            if (mu < nu) {
                                index += n3 * n4;
                                continue;
                            }
                            size_t p12 = (mu * (mu + 1)) / 2 + nu;

                            for (size_t f3 = 0; f3 < n3; ++f3) {
                                size_t lam = bf3_start + f3;

                                for (size_t f4 = 0; f4 < n4; ++f4, ++index) {
                                    size_t sig = bf4_start + f4;

                                    if (lam < sig) {
                                        continue;
                                    }
                                    size_t p34 = (lam * (lam + 1)) / 2 + sig;

                                    if (s1 == s3 && s2 == s4 && p12 < p34) {
                                        continue;
                                    }

                                    double ERI_value = quartet_buffer[index];


                                    ERI_tensor(mu, nu, lam, sig) = ERI_value;
                                    ERI_tensor(nu, mu, lam, sig) = ERI_value;
                                    ERI_tensor(mu, nu, sig, lam) = ERI_value;
                                    ERI_tensor(nu, mu, sig, lam) = ERI_value;
                                    ERI_tensor(lam, sig, mu, nu) = ERI_value;
                                    ERI_tensor(lam, sig, nu, mu) = ERI_value;
                                    ERI_tensor(sig, lam, mu, nu) = ERI_value;
                                    ERI_tensor(sig, lam, nu, mu) = ERI_value;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

Eigen::Tensor<double, 4> ERI::motransform_eri(
    const Eigen::Tensor<double, 2>& C1,
    const Eigen::Tensor<double, 2>& C2,
    const Eigen::Tensor<double, 2>& C3,
    const Eigen::Tensor<double, 2>& C4) {

    Eigen::array<Eigen::IndexPair<int>, 1> contract = {
        Eigen::IndexPair<int>(0, 0)
    };

    Eigen::Tensor<double, 4> I1 = ERI_tensor.contract(C1, contract);
    Eigen::Tensor<double, 4> I2 = I1.contract(C2, contract);
    Eigen::Tensor<double, 4> I3 = I2.contract(C3, contract);
    Eigen::Tensor<double, 4> I4 = I3.contract(C4, contract);

    return I4;
}