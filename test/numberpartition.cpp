
#include "numberpartition.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <random>

NumberPartitionProblem::NumberPartitionProblem(std::string filename, float *& Q_flatten, float **& Q)
{
    fromFile(filename, Q_flatten, Q);
}

NumberPartitionProblem::NumberPartitionProblem(std::string filename, std::vector<float>& Q_flatten, std::vector<std::vector<float> >& Q)
{
    fromFile(filename, Q_flatten, Q);
}

NumberPartitionProblem::~NumberPartitionProblem()
{
}

bool NumberPartitionProblem::fromFile(std::string problemFilePath, float *& Q_flatten, float **& Q)
{
    _is_from_file = true;
    _filename = problemFilePath;

    // --Initiate input file stream--
    // Check that the path exists
    std::ifstream ifs(problemFilePath.c_str());
    if (!ifs)
    {
        std::cerr << "Error: \"" << problemFilePath << "\" does not exist!!\n";
        return false;
    }
    
    // --Ignore the comments--
    // '%' indicates that the line is a comment
    // The code assumes '%' is followed by at least one space
    // and uses getline to read the rest of the line
    std::string word;
    ifs >> word;
    while (word == "%")
    {
        std::getline(ifs, word);
        ifs >> word;
    }

    // --Get the problem size and the number of matrix entries--
    // Halt the process if the problem is too large
    int problemSize{};
    int entries{};
    ifs >> problemSize >> entries;
    // if (problemSize > MAX_QUBITS)
    if (problemSize > 4096)
    {
        std::cerr << "The problem size " << problemSize << " is larger than the number of qubits " << 4096
                  << " and cannot be mapped to the architecture.\n";
        return false;
    }
    
    // 64 is CACHE_FACTOR in SBM
    if (problemSize % 64)
    {
        _padding = 64 - problemSize % 64;
    }
    _problem_size = problemSize;
    // getline reads the end of the problemSize line
    std::getline(ifs, word);
    std::getline(ifs, word);

    // --Store the numbers to partition--
    // word is now the numbers of the number partition problem
    // The numbers are stored for solution quality check
    _numbers = new float[problemSize]();
    std::stringstream numberStream(word);
    for (int i = 0; i < problemSize; ++i)
    {
        numberStream >> _numbers[i];
    }
#if 0
    printNumbers();
#endif

    // --Store the Ising model problem formulation in Q--
    int col{};
    int row{};
    float value{};
    int matrixSize = _problem_size + _padding;
    Q_flatten = new float[matrixSize * matrixSize]{0};
    Q = new float *[matrixSize];

    for (int i = 0; i < matrixSize; i++)
    {
        Q[i] = new float[matrixSize]{0};
    }
    for (int i = 0; i < entries; i++)
    {
        ifs >> col >> row >> value;
        Q_flatten[(col - 1) * matrixSize + (row - 1)] = value;
        Q_flatten[(row - 1) * matrixSize + (col - 1)] = value;
        Q[col - 1][row - 1] = value;
        Q[row - 1][col - 1] = value;
    }

#if 0
    for (int i = 0; i < problemSize; i++)
    {
        for (int j = 0; j < problemSize; j++)
        {
            std::cout << Q[i][j] << " ";
        }
        std::cout << "\n";
    }
#endif
    return true;
}

bool NumberPartitionProblem::fromFile(std::string problemFilePath, std::vector<float>& Q_flatten, std::vector<std::vector<float>>& Q)
{
    _is_from_file = true;
    _filename = problemFilePath;

    // --Initiate input file stream--
    // Check that the path exists
    std::ifstream ifs(problemFilePath.c_str());
    if (!ifs)
    {
        std::cerr << "Error: \"" << problemFilePath << "\" does not exist!!\n";
        return false;
    }
    
    // --Ignore the comments--
    // '%' indicates that the line is a comment
    // The code assumes '%' is followed by at least one space
    // and uses getline to read the rest of the line
    std::string word;
    ifs >> word;
    while (word == "%")
    {
        std::getline(ifs, word);
        ifs >> word;
    }

    // --Get the problem size and the number of matrix entries--
    // Halt the process if the problem is too large
    int problemSize{};
    int entries{};
    ifs >> problemSize >> entries;
    // if (problemSize > MAX_QUBITS)
    if (problemSize > 4096)
    {
        std::cerr << "The problem size " << problemSize << " is larger than the number of qubits " << 4096
                  << " and cannot be mapped to the architecture.\n";
        return false;
    }
    
    // 64 is CACHE_FACTOR in SBM
    // if (problemSize % 64)
    // {
    //     _padding = 64 - problemSize % 64;
    // }
    // if (problemSize % 32)
    // {
    //     _padding = 32 - problemSize % 32;
    // }
    if (problemSize % 4)
    {
        _padding = 4 - problemSize % 4;
    }
    _problem_size = problemSize;
    // getline reads the end of the problemSize line
    std::getline(ifs, word);
    std::getline(ifs, word);

    // --Store the numbers to partition--
    // word is now the numbers of the number partition problem
    // The numbers are stored for solution quality check
    _numbers = new float[problemSize]();
    std::stringstream numberStream(word);
    for (int i = 0; i < problemSize; ++i)
    {
        numberStream >> _numbers[i];
    }
#if 0
    printNumbers();
#endif

    // --Store the Ising model problem formulation in Q--
    int col{};
    int row{};
    float value{};
    int matrixSize = _problem_size + _padding;
    Q_flatten.resize(matrixSize * matrixSize, 0);
    Q.resize(matrixSize);

    for (int i = 0; i < matrixSize; i++)
    {
        Q[i].resize(matrixSize, 0);
    }
    for (int i = 0; i < entries; i++)
    {
        ifs >> col >> row >> value;
        Q_flatten[(col - 1) * matrixSize + (row - 1)] = value;
        Q_flatten[(row - 1) * matrixSize + (col - 1)] = value;
        Q[col - 1][row - 1] = value;
        Q[row - 1][col - 1] = value;
    }

#if 0
    for (int i = 0; i < problemSize; i++)
    {
        for (int j = 0; j < problemSize; j++)
        {
            std::cout << Q[i][j] << " ";
        }
        std::cout << "\n";
    }
#endif
    return true;
}

void NumberPartitionProblem::printNumbers()
{
    for (int i = 0; i < _problem_size; ++i)
    {
        std::cout << _numbers[i] << " ";
    }
    std::cout << "\n";
}
