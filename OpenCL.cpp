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

    // Load OpenCL kernel code from file
    ifstream kernelFile("matrix_multiplication.cl");
    if (!kernelFile.is_open()) {
        cerr << "Failed to open kernel file." << endl;
        return 1;
    }
    string kernelCode((istreambuf_iterator<char>(kernelFile)), istreambuf_iterator<char>());
    kernelFile.close();

    // Get available OpenCL platforms
    vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    if (platforms.empty()) {
        cerr << "No OpenCL platforms found." << endl;
        return 1;
    }

    // Use the first available platform
    cl::Platform platform = platforms.front();

    // Get available OpenCL devices
    vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
    if (devices.empty()) {
        cerr << "No GPU devices found." << endl;
        return 1;
    }

    // Use the first available device
    cl::Device device = devices.front();

    // Create an OpenCL context and command queue
    cl::Context context(device);
    cl::CommandQueue queue(context, device);

    // Compile the kernel code
    cl::Program::Sources sources(1, make_pair(kernelCode.c_str(), kernelCode.length()));
    cl::Program program(context, sources);
    if (program.build({device}) != CL_SUCCESS) {
        cerr << "Failed to build kernel." << endl;
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

    // Create kernel object and set arguments
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
    queue.enqueueReadBuffer(bufferC, CL_TRUE, 0, sizeof(int) * N * N, C);

    // Output results
    ofstream outputFile("output.txt");
    if (outputFile.is_open()) {
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                outputFile << C[i][j] << " ";
            }
            outputFile << endl;
        }
        outputFile.close();
        cout << "Output written to output.txt" << endl;
    } else {
        cerr << "Unable to open file for writing." << endl;
    }

    return 0;
}
