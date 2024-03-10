#include <iostream>
#include <fstream>
#include <chrono>
#include <random>
#include <CL/cl.hpp>

using namespace std;
using namespace chrono;

const int MAX_SIZE = 10;

// Function to generate random values for a matrix
void generateRandomMatrix(int matrix[][MAX_SIZE], int N) {
    random_device rnd;
    mt19937 gen(rnd());
    uniform_int_distribution<> dis(1, 10);  // Range of random number set from 1 to 10
    for (int i = 0; i < N; ++i) {    // Iteration over rows of matrix
        for (int j = 0; j < N; ++j) {  // Iteration over columns of matrix
            matrix[i][j] = dis(gen);    // Random value generated
        }
    }
}

int main() {
    // Size of matrix taken from user
    int N;
    cout << "Enter the size of the matrix (up to " << MAX_SIZE << "): ";
    cin >> N;

    // Check if matrix size is within bounds
    if (N <= 0 || N > MAX_SIZE) {
        cout << "Invalid matrix size." << endl;
        return 1;
    }

    // Initialize OpenCL
    vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    if (platforms.empty()) {
        cerr << "No OpenCL platforms found." << endl;
        return 1;
    }

    cl::Platform platform = platforms.front();
    vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
    if (devices.empty()) {
        cerr << "No GPU devices found." << endl;
        return 1;
    }

    cl::Device device = devices.front();
    cl::Context context(device);
    cl::CommandQueue queue(context, device);

    // Load and compile OpenCL kernel code
    ifstream file("matrix_multiplication.cl");
    if (!file.is_open()) {
        cerr << "Failed to open kernel file." << endl;
        return 1;
    }
    string sourceCode(istreambuf_iterator<char>(file), (istreambuf_iterator<char>()));
    cl::Program::Sources sources(1, make_pair(sourceCode.c_str(), sourceCode.length() + 1));
    cl::Program program(context, sources);
    if (program.build({device}) != CL_SUCCESS) {
        cerr << "Failed to compile kernel." << endl;
        return 1;
    }

    // Generate random values for matrices A and B
    int A[MAX_SIZE][MAX_SIZE], B[MAX_SIZE][MAX_SIZE];
    generateRandomMatrix(A, N);
    generateRandomMatrix(B, N);

    // Create OpenCL buffers for matrices A, B, and C
    cl::Buffer bufferA(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * N * N, A);
    cl::Buffer bufferB(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * N * N, B);
    cl::Buffer bufferC(context, CL_MEM_WRITE_ONLY, sizeof(int) * N * N);

    // Create kernel and set arguments
    cl::Kernel kernel(program, "matrixMultiplication");
    kernel.setArg(0, bufferA);
    kernel.setArg(1, bufferB);
    kernel.setArg(2, bufferC);
    kernel.setArg(3, N);

    // Execute the kernel
    cl::NDRange global(N, N);
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, global);

    // Read result back to host memory
    int C[MAX_SIZE][MAX_SIZE];
    queue.enqueueReadBuffer(bufferC, CL_TRUE, 0, sizeof(int
