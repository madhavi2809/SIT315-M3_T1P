#include <iostream>
#include <fstream>
#include <chrono>
#include <random>
#include <mpi.h>
#include <CL/cl.hpp>

using namespace std;
using namespace chrono;

const int MAX_SIZE = 10;

// Function to generate random values for a matrix
void generateRandomMatrix(int matrix[][MAX_SIZE], int N) {
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis(1, 10);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            matrix[i][j] = dis(gen);
        }
    }
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N;
    if (rank == 0) {
        cout << "Enter the size of the matrix (up to " << MAX_SIZE << "): ";
        cin >> N;

        // Check if matrix size is within bounds
        if (N <= 0 || N > MAX_SIZE) {
            cout << "Invalid matrix size." << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    // Broadcast matrix size to all processes
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Calculate rows per process
    int rowsPerProcess = N / size;
    int startRow = rank * rowsPerProcess;
    int endRow = (rank == size - 1) ? N : (rank + 1) * rowsPerProcess;

    // Allocate memory for matrices A, B, and C
    int A[MAX_SIZE][MAX_SIZE], B[MAX_SIZE][MAX_SIZE], C[MAX_SIZE][MAX_SIZE];

    // Generate random values for matrices A and B
    generateRandomMatrix(A, N);
    generateRandomMatrix(B, N);

    // Initialize OpenCL
    cl::Context context(CL_DEVICE_TYPE_GPU); // Change to CL_DEVICE_TYPE_CPU if GPU is not available
    vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
    cl::CommandQueue queue(context, devices[0]);
    ifstream sourceFile("matrix_multiplication.cl");
    string sourceCode(istreambuf_iterator<char>(sourceFile), (istreambuf_iterator<char>()));
    cl::Program::Sources sources(1, make_pair(sourceCode.c_str(), sourceCode.length() + 1));
    cl::Program program(context, sources);
    program.build(devices);
    cl::Kernel kernel(program, "matrixMultiplication");

    // Transfer matrices A and B to device memory
    cl::Buffer bufferA(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * N * N, A);
    cl::Buffer bufferB(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * N * N, B);
    cl::Buffer bufferC(context, CL_MEM_WRITE_ONLY, sizeof(int) * N * N);

    // Set kernel arguments
    kernel.setArg(0, bufferA);
    kernel.setArg(1, bufferB);
    kernel.setArg(2, bufferC);
    kernel.setArg(3, N);
    kernel.setArg(4, startRow);
    kernel.setArg(5, endRow);

    // Execute kernel
    cl::NDRange global(N);
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, cl::NullRange);

    // Read result back to host memory
    queue.enqueueReadBuffer(bufferC, CL_TRUE, 0, sizeof(int) * N * N, C);

    auto start = high_resolution_clock::now();
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);

    if (rank == 0) {
        // Print execution time
        cout << "Execution time using MPI and OpenCL: " << duration.count() << " microseconds" << endl;

        // Write output to a file
        ofstream outputFile("output_mpi_opencl.txt");
        if (outputFile.is_open()) {
            for (int i = 0; i < N; ++i) {
                for (int j = 0; j < N; ++j) {
                    outputFile << C[i][j] << " ";
                }
                outputFile << endl;
            }
            outputFile.close();
            cout << "Output written to output_mpi_opencl.txt" << endl;
        } else {
            cerr << "Unable to open file for writing." << endl;
        }
    }

    MPI_Finalize();
    return 0;
}
