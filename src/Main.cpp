#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <limits>
#include <chrono>
#include <libint2.hpp>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include "HF.h"
#include "ERI.h"
#include "MP2.h"
#include "CCSD.h"



using namespace std::chrono;
using std::cout;
using std::string;
using std::vector;

void print_header();

int main(int argc, char* argv[]) {

    int counter = 0, xyzstart = std::numeric_limits<int>::max() , charge, mult;
    string line, method, basis, fline;
    string delimiter = " ";
    string scfMode = "DIIS";
    vector<string> xyzlines;
    

    auto start = high_resolution_clock::now();
    std::ifstream input(argv[1]);
    std::ofstream xyzFile("geom.xyz");

    //file checkkk
    if (argc < 2) {
        std::cerr << "no input dumbass\n";
        return 1;
    }

    print_header();
    cout << "Parsing input file: "  << argv[1] << std::endl;
    
    //PARSE THE INPUT AND CREATE XYZ
    while (std::getline(input, line)) {

        if(counter == 0){

            method = line.substr(0, line.find(delimiter));
            method.erase (std::remove (method.begin(), method.end(), ' '), method.end());
            basis = line.substr(line.find(delimiter) + 1);
            basis.erase (std::remove (basis.begin(), basis.end(), ' '), basis.end());

        }
        if(line.substr(0, line.find(delimiter)) == "SCFMODE"){

            scfMode = line.substr(line.find(delimiter) + 1);

        }
        if(line.substr(0, line.find(delimiter)) == "xyz"){

            xyzstart = counter;

        } 
        if (counter > xyzstart)
        {

            xyzlines.push_back(line);
           
        }
        
        counter++; 
    }

    //CREATE AN XYZ FILE
    xyzFile << xyzlines.size() << "\n";
    xyzFile << "\n";
    for (const auto& line : xyzlines) {
        xyzFile << line << "\n"; 
    }
    xyzFile.close();
    input.close();


    libint2::initialize(); // INITIALIZE LIBINT2

    //CREATE BASIS AND ATOM FILES FOR LIBINT2
    string xyzfilename = "geom.xyz"; // see XYZ format description at http://en.wikipedia.org/wiki/XYZ_file_format
    std::ifstream input_file(xyzfilename);
    vector<libint2::Atom> atoms = libint2::read_dotxyz(input_file);


    cout << "Basis set is set to " << basis << "\n";
    cout << method << " Calculation started!\n" << "\n";

    //RUN HF CALC
    HF HFsolver(atoms, basis);
    HFsolver.scf(scfMode);
    double E_scf = HFsolver.get_Escf();
    cout << "\n" << "FINAL SCF ENERGY IS " << E_scf << " Eh\n";

    if (method == "MP2"){

        int n_occ = HFsolver.get_n_occ();
        int n_vir = HFsolver.get_n_vir();
        
        Eigen::MatrixXd C_occ = HFsolver.get_C_occ();
        Eigen::MatrixXd C_vir = HFsolver.get_C_vir();
        Eigen::VectorXd eps = HFsolver.get_Eorb(); 

        MP2 MP2solver(atoms,basis);
        MP2solver.compute_correlation(E_scf, n_occ, n_vir, C_occ, C_vir, eps);

    }else if (method == "CCSD"){
        
        Eigen::MatrixXd C = HFsolver.get_C();
        Eigen::VectorXd eps = HFsolver.get_Eorb();

        CCSD CCSDsolver(atoms, basis);
        CCSDsolver.compute_ccsd(C, eps, E_scf);
    
        libint2::finalize();



    }else{

        cout << "\n" << "FINAL HF ENERGY IS " << E_scf << " Eh\n";

    }

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);
    double sec, msec = std::modf(duration.count()/1000.0, &sec);

    cout << "\nEXECUTION TIME: " << (int)sec << " sec " << (int)std::round(msec*1000) << " msec\n";

    return 0;
    
}

void print_header(){





                                                                    
                                                                    
                                                                    
                                                                    
    cout << "\n                                          $$$$$$$$$   X$$     .$X         ;$$$$$:   :$$    X$;     X$x    ";  
    cout << "\n                                             +$x     ;$X$x    .$X       :$$:   +$$  :$$    X$;    X$$$;   ";  
    cout << "\n                                             +$x    .$X X$;   .$X       $$          :$$    X$;   ;$X $$.  ";  
    cout << "\n                                             +$x    X$.  X$.  .$X       $$   :$$$$: :$$++++X$;   $$  .$X  ";  
    cout << "\n                                             +$x   x$$$$$$$X  .$X       x$X     X$: :$$    X$;  $$$$$$$$X ";  
    cout << "\n                                             +$x  ;$X     +$X .$$$$$$$;  ;$$$$$$$X  :$$    X$; +$x     X$;";                            cout << "\n";                                                                                                                          
    cout << "\n";
    cout << "\n";                                                                                                
    cout << "\n";                                                                                                   
    cout << "\n";                                                                                                
    cout << "\n                                                                   ...------------..                           ";       
    cout << "\n                                                              ..-++-....        ...--++..                      ";       
    cout << "\n                                                           .-+-..                     ..-+-.                   ";       
    cout << "\n                                                       ..-+.                              .+-.   ....          ";       
    cout << "\n                                                     .--.           .  .....                .+-. .--.          ";       
    cout << "\n                                                   .--.           ...  ....   ......         .--  ..           ";       
    cout << "\n                                                 .--.    .. ..  -+-++..+--+..+..-- .  .--.    .+. ..           ";       
    cout << "\n                                               .--      .....+-+-..--....-+..  ..+-.+-..   .   .+              ";       
    cout << "\n                                              .-.       .+---   ..-  ..-.  ..--.......-+-+--.   #.             ";       
    cout << "\n                                            .--.  .    .+..-.   .. .-.  ......--------...--.   -+              ";       
    cout << "\n                                           .+-  .-..+++-. ....   .-.      ..-+..     ...---#. .+.              ";       
    cout << "\n                                          .+.     --.-   -.-.        -   .++    .-+        .###.               ";       
    cout << "\n                                         -+.     .-.-.    .-   .   ...  -+.       .+--                         ";       
    cout << "\n                                        .+.  .-+--. ..   .-   -   .-   .+.                  .+-                ";       
    cout << "\n                                       -+.  .-+.   .-    ..  ..  .- ...+.                    ..                ";       
    cout << "\n                                      -+  .+....     .-.+#-...+#-+  ..+-                                       ";       
    cout << "\n                                    .--.  -. ..  ....+-.       .-+....#.                                       ";       
    cout << "\n                                   .--.-..-. ..  .--.      .-.....+.. #.                                       ";       
    cout << "\n                                  .+.--.--. .  .++.  .-....-#+++-- .- #.                                       ";       
    cout << "\n                                .++-..  .-  ..+-.    .++-+.+.+. -. .. -+                                       ";       
    cout << "\n                             ..+-.  ...-....+-.  .-+++-.--.-+.- .-  ...+-                                      ";       
    cout << "\n                          ..--.     -  ----....----......- .-.....-. ...--.                                    ";       
    cout << "\n                                 ...-++-..--............ .-..--....  ..  .+-.                                  ";       
    cout << "\n                           .-+++++-... ... .    .-.  .-. .-.. .+++.-. .-. ..-+-..-+#-.                         ";       
    cout << "\n                                       .-.+++-.        .. ..-+..+...-++++++++++-.. .-++..  ....++.             ";       
    cout << "\n                                ...---++..-. .-..    ...-++-...  .+-..        ..-.  ....-+#+-....---.          ";       
    cout << "\n                                 ...   ...  .--..-.......  ..  ..  ..----.     . ..   .--...--+--...---------. ";       
    cout << "\n                                     ...--...... ..-. .-+-.      .     ..... .+-+-.     ....-............      ";       
    cout << "\n                                .....              ..++. .-+-... ...    ...++.   .-++.....                     ";       
    cout << "\n                                             ...-++-.....    .-+++----+++.. ....     ..--++--.                 ";       
    cout << "\n                                             ...                              .    .                           ";       
    cout << "\n";                                                              
    cout << "\n";                                                                           
    cout << "\n                                             .xXxX                                                   ";           
    cout << "\n                                              ;Xxx.     :&:     .;xx:.;xx:.;xx;.  :xx+.:xx+.  .$+    ";           
    cout << "\n                                           :;.X+;x;&+.   :$x:  ;&; .XX; .X$: .$$.x&. ;&X  ;&;  .X$;. ";           
    cout << "\n                                          X&. +$$; :&X.    .&X.$$.  x&.  X$.  $$:&X  :&+  :&x     x&:";           
    cout << "\n                                         +&+        X&:;$   +&.X&:          .Xx  X&:      +$.:&:  .&+";           
    cout << "\n                                         +&x        X&;;&x:+&:  +&X..            .$&+.   ::  .$$:;&+ ";           
    cout << "\n                                         .&&.      .&$            :$&&+..         ..X&&;.            ";           
    cout << "\n                                          .$&;    :&x.            :++x$&&&x.      .. .:x&&X.         ";           
    cout << "\n                                            .:+xx+..                     ..X.             .x.        ";           
    cout << "\n";
}