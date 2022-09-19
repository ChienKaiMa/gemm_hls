
#include <iostream>
#include <string>
#include <vector>

class NumberPartitionProblem
{
protected:
    /* data */
    bool _is_from_file = false;
    std::string _filename;
    uint _problem_size = 0;
    uint _padding = 0;
    float* _numbers;
public:
    NumberPartitionProblem(/* args */); // TODO
    NumberPartitionProblem(std::string filename, float *& Q_flatten, float **& Q);
    NumberPartitionProblem(std::string filename, std::vector<float>& Q_flatten, std::vector<std::vector<float>>& Q);
    ~NumberPartitionProblem();
    bool fromFile(std::string filename, float *& Q_flatten, float **& Q);
    bool fromFile(std::string filename, std::vector<float>& Q_flatten, std::vector<std::vector<float>>& Q);
    uint getProblemSize() { return _problem_size; }
    uint getMatrixSize() { return _problem_size + _padding; }
    float getNumber(int index) { return _numbers[index]; }
    void printNumbers();
};
