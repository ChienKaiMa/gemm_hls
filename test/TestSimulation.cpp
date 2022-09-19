
// #include "simulatedBifurcationOptimizer.hpp"
#include "numberpartition.hpp"
#include <ctime>
#include <fstream>
#include <string>
#include <sstream>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <type_traits>
#include <vector>
#include "MatrixMultiplication.h"
#include "Utility.h"

int main(int argc, char *argv[])
{
    auto timer = std::time(nullptr);
    auto start_time = std::localtime(&timer);
    std::cerr << std::put_time(start_time, "%Y-%m-%d %T");
    std::cerr << " Test starts.\n";

    // --Read problem information--
    //std::string problemFilePath = "num_par_32_ising_0.txt";
    std::string problemFilePath = "num_par_32_ising_0.txt";
    if (argc == 2) problemFilePath = argv[1];

    std::vector<float> Q_flatten;
    std::vector<std::vector<float>> Q;
    NumberPartitionProblem problem = NumberPartitionProblem(problemFilePath, Q_flatten, Q);

    auto timer1 = std::time(nullptr);
    auto read_time = std::localtime(&timer1);
    std::cerr << std::put_time(read_time, "%Y-%m-%d %T");
    std::cerr << " Read problem information done.\n";

    // TODO: Measure problem read time

    /*
    ** Simulated bifurcation algorithm setup
    */
    int steps = 200;
    float dt = 0.01;
    float c0 = 0.001;
    uint problem_size = problem.getProblemSize();
    uint matrix_size = problem.getMatrixSize();
    std::vector<float> x(matrix_size, 0);
    std::vector<float> y(matrix_size, 0);
    for (int i = 0; i < matrix_size; ++i) {
        y[i] = 0.05;
    }

    for (int i = 0; i < matrix_size; ++i) {
        for (int j = 0; j < matrix_size; ++j) {
            std::cerr << Q_flatten[i * matrix_size + j] << " ";
        }
        std::cerr << "\n";
    }

    uint best_spin[matrix_size / 32] = {0};
    float myC0 = 4.2479e-06;
    float myDT = 0.01;

    float delta_a[steps];
    for (int i = 0; i < steps; ++i)
    {
        delta_a[i] = float(steps - i) / steps;
    }
    float x_history[matrix_size * steps]{0};

    auto jKernel = Pack<kMemoryWidthA>(Q_flatten);
    auto xKernel = Pack<kMemoryWidthM>(x);
    auto yKernel = Pack<kMemoryWidthM>(y);
    // auto cKernel = Pack<kMemoryWidthM>(cReference);


    auto t1 = clock();
    SimulatedBifurcationKernel(jKernel.data(), xKernel.data(), yKernel.data(), delta_a, myC0, myDT, matrix_size, steps, best_spin, x_history);
    auto t2 = clock();
    
    auto timer2 = std::time(nullptr);
    auto execute_time = std::localtime(&timer2);
    std::cerr << std::put_time(execute_time, "%Y-%m-%d %T") << " Simulated bifurcation done.\n";
    std::cout << "Execution time = " << ((float)(t2 - t1))/CLOCKS_PER_SEC << " seconds.\n\n";


    //
    // Direct SBM history (x) to file
    //
    std::cout << "---Dumping x history---\n";
    std::ofstream x_outfile;
    std::ofstream energy_outfile;
    // outfile.open(argv[2], std::ios::out);
    // TODO
    // Wrap the utilities in functions
    x_outfile.open("x_history.log", std::ios::out);

    for (int i = 0; i < steps; ++i)
    {
        for (int j = 0; j < matrix_size; ++j)
        {
            x_outfile << x_history[i * matrix_size + j] << " ";
        }
        x_outfile << "\n";
    }

    std::cout << "---Dumping x history done---\n\n";

    //
    // SBM summary
    //

    // TODO
    // Save the report as file
    std::cout << "---SBM summary---\n";
    std::cout << "Final spin: ";
    for (int i = 0; i < (matrix_size / 32); ++i)
    {
        ap_uint<32> u = best_spin[i];
        for (int j = 0; j < 32; ++j) {
            std::cout << u[j] << " ";
        }
    }
    std::cout << "\n";
}
