#include <iostream>
#include <chrono>
#include <random>
#include <mpi.h>

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

// Function to perform matrix multiplication for a subset of rows
void matrixMultiplication(int A[][MAX_SIZE], int B[][MAX_SIZE], int C[][MAX_SIZE], int startRow, int endRow, int N) {
    for (int i = startRow; i < endRow; ++i) {
        for (int j = 0; j < N; ++j) {
            C[i][j] = 0;
            for (int k = 0; k < N; ++k) {
                C[i][j] += A[i][k] * B[k][j];
            }
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

    // Perform matrix multiplication
    auto start = high_resolution_clock::now();
    matrixMultiplication(A, B, C, startRow, endRow, N);

    // Gather results from all processes to process 0
    MPI_Gather(C[startRow], (endRow - startRow) * N, MPI_INT, C, (endRow - startRow) * N, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        // Print execution time
        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(stop - start);
        cout << "Execution time using MPI and OpenMP: " << duration.count() << " microseconds" << endl;

        // Print the matrix directly to the terminal
        cout << "Resulting Matrix:" << endl;
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                cout << C[i][j] << " ";
            }
            cout << endl;
        }
    }

    MPI_Finalize();
    return 0;
}
