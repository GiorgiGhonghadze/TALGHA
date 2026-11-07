# 1. Mandatory CMake boilerplate
cmake_minimum_required(VERSION 3.15)
project(HartreeFock CXX)

# Force C++17 standard (recommended for modern Eigen and Libint)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 2. Find all required packages, including OpenMP
find_package(Eigen3 REQUIRED)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
find_package(Libint2 REQUIRED)
find_package(OpenMP REQUIRED)

# 3. Create the executable target FIRST
# Note: If your C++ file is not named 'main.cpp', change it here so CMake can find it.
add_executable(main  Main.cpp HF.cpp ERI.cpp MP2.cpp CCSD.cpp DIIS.cpp)

# 4. Link the libraries to the explicitly defined 'HF' target
target_link_libraries(main PRIVATE 
    Eigen3::Eigen
    Libint2::cxx 
    ${BLAS_LIBRARIES} 
    ${LAPACK_LIBRARIES}
    OpenMP::OpenMP_CXX
)
