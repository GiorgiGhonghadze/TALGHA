cmake_minimum_required(VERSION 3.15)
project(TALGHA CXX)

add_compile_options(-O3 -march=native -DNDEBUG)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Eigen3 REQUIRED)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
find_package(Libint2 REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(main src/Main.cpp src/HF.cpp src/ERI.cpp src/MP2.cpp src/CCSD.cpp src/DIIS.cpp)

include_directories(include)

target_link_libraries(main PRIVATE 
    Eigen3::Eigen
    Libint2::cxx 
    ${BLAS_LIBRARIES} 
    ${LAPACK_LIBRARIES}
    OpenMP::OpenMP_CXX
)
