# TALGHA: Quantum Chemistry Program

TALGHA (ტალღა in Georgian) means wave and subsequently the program computes energies based on Ab Initio Quantum Chemistry Methods.

## Key Features

* **Hartree-Fock SCF Engine:** All integrals are evaluated using Libint2, utilizing 8-fold integral symmetry and Schwartz prescreening for accelerration of ERI calculations.
* **Post-HF Methods:** Features fully functional MP2 that scales as $O(N^{5})$ using partial MO transformation and CCSD code scaling $O(N^{6})$ as formulated by Stanton et al. J. Chem. Phys. 94, 4334–4345 (1991).

## Benchmarks & Validation

The TALGHA engine has been strictly validated against the ORCA 6.1.1(Same conditions, DIIS enabled, HCore guess, noRI, tolE of $10^{-9}$). Total energies show near-perfect agreement, with deviations consistently falling below $10^{-6}$ Hartree.

### Total Energies (Hartree)

| Molecule | Method | TALGHA ($E_h$) | ORCA ($E_h$) | Difference ($\Delta E_h$) |
| :--- | :--- | :--- | :--- | :--- |
| **N₂** | HF | -106.766593849 | -106.766593857 | 8.2E-09 |
| | MP2 | -106.903172870 | -106.903173013 | 1.4E-07 |
| | CCSD | -106.938741577 | -106.938743240 | 1.7E-06 |
| **CO** | HF | -111.224558696 | -111.224558696 | 2.9E-10 |
| | MP2 | -111.353081242 | -111.353081122 | 1.2E-07 |
| | CCSD | -111.355381823 | -111.355382180 | 3.6E-07 |
| **H₂O** | HF | -74.963023139 | -74.963023139 | 5.2E-10 |
| | MP2 | -74.998568790 | -74.998568805 | 1.5E-08 |
| | CCSD | -75.012461852 | -75.012461405 | 4.5E-07 |

### Dependencies
* **CMake**   (>= 3.15)
* **Eigen3**  (Linear algebra)
* **BLAS**    (Linear algebra, Eigen dependencies)
* **LAPACK**  (Linear algebra, Eigen dependencies)
* **Libint2** (Molecular integrals)
* **OpenMP**  (Parallelization) #NOT YET IMPLEMENTED

### Compilation
Clone the repository and build using CMake:

```bash
git clone https://github.com/GiorgiGhonghadze/TALGHA.git
cd TALGHA
mkdir build && cd build
cmake ..
make -j
```
