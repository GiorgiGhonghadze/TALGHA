#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <libint2.hpp>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include "HF.h"
#include "ERI.h"
#include "MP2.h"



using namespace std::chrono;
using std::cout;
using std::string;
using std::vector;


int main(int argc, char* argv[]) {
    
    auto start = high_resolution_clock::now();


    int counter = 0, dim, charge, mult;
    string line, method, basis, fline;
    
    vector<string> xyztext;
    vector<string>::size_type sz;
    vector<double> xyz;
    
    
    std::ifstream input(argv[1]);
    std::ofstream xyzFile("geom.xyz");

    // file checkkk
    if (argc < 2) {
        std::cerr << "no input dumbass\n";
        return 1;
    }

    cout << "Parsing input file: "  << argv[1] << std::endl;
    
    //PARSE THE INPUT AND CREATE XYZ
    while (std::getline(input, line)) {

        switch (counter)
        {
        case 0:
            method = line;
            break;
        case 1:
            basis =  line;
            break;
        default:
            break;
        }
        // cout << line << std::endl;
        if (counter >= 4)
        {
            
            xyzFile << line << "\n";
           
        }
        
        counter++; 
    }

    xyzFile.close();
    input.close();


    libint2::initialize(); // INITIALIZE LIBINT2

    //CREATE BASIS AND ATOM FILES FOR LIBINT2
    string xyzfilename = "geom.xyz";
    std::ifstream input_file(xyzfilename);
    vector<libint2::Atom> atoms = libint2::read_dotxyz(input_file);


    cout << "Basis set is set to " << basis << "\n";
    cout << method << "Calculation started!\n" << std::endl;

    string solver = "noDIIS";

    //RUN HF CALC
    HF HFsolver(atoms, basis);
    HFsolver.scf(solver);
    double Escf = HFsolver.get_Escf();
    cout << "\n" << "FINAL SCF ENERGY IS " << Escf << " Eh\n";

    if (method == "MP2"){

        int n_occ = HFsolver.get_n_occ();
        int n_vir = HFsolver.get_n_vir();
        Eigen::MatrixXd C_occ = HFsolver.get_C_occ();
        Eigen::MatrixXd C_vir = HFsolver.get_C_vir();
        Eigen::VectorXd eps = HFsolver.get_Eorb(); 

        MP2 MP2solver(atoms,basis);
        MP2solver.compute_correlation(Escf, n_occ, n_vir, C_occ, C_vir, eps);

    }else{

        cout << "\n" << "FINAL HF ENERGY IS " << Escf << " Eh\n";

    }
    
    libint2::finalize();

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);
    double sec, msec = std::modf(duration.count()/1000.0, &sec);

    cout << "\nEXECUTION TIME: " << (int)sec << " sec " << (int)std::round(msec*1000) << " msec\n";

    return 0;
    
}