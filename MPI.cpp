#include <iostream>
#include <fstream>
#include <chrono>
#include <random>
#include <mpi.h>

using namespace std;
using namespace chrono;

const int MAX_SIZE = 10;

// Function to perform matrix multiplication
void Multiplication(int A[][MAX_SIZE], int B[][MAX_SIZE], int C[][MAX_SIZE], int N) {
    for (int i = 0; i < N; ++i) {   // Iteartion over rows of matrix C
        for (int j = 0; j < N; ++j) {   //Iteration over columns of matrix C
            C[i][j] = 0;
            // Iteration over row elements of matrix A and column elements of matrix B
            for (int k = 0; k < N; ++k) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

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

int main(int argc, char* argv[]) {
    // Initialize MPI
    MPI_Init(&argc, &argv);

    // Get the total number of processes and the rank of this process
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Size of matrix taken from user
    int N;
    if (world_rank == 0) {
        cout << "Enter the size of the matrix (up to " << MAX_SIZE << "): ";
        cin >> N;

        // Check if matrix size is within bounds
        if (N <= 0 || N > MAX_SIZE) {
            cout << "Invalid matrix size." << endl;
            MPI_Finalize();
            return 1;
        }
    }

    // Broadcast matrix size to all processes
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Partition matrices A and B among processes
    const int rows_per_process = N / world_size;
    int A[MAX_SIZE][MAX_SIZE], B[MAX_SIZE][MAX_SIZE], C[MAX_SIZE][MAX_SIZE];

    // Generate random values for matrices A and B only on root process
    if (world_rank == 0) {
        generateRandomMatrix(A, N);
        generateRandomMatrix(B, N);
    }

    // Scatter matrices A and B among processes
    MPI_Scatter(A, rows_per_process * MAX_SIZE, MPI_INT, A, rows_per_process * MAX_SIZE, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(B, N * N, MPI_INT, 0, MPI_COMM_WORLD);

    // Perform local matrix multiplication
    int local_C[MAX_SIZE][MAX_SIZE];
    Multiplication(A, B, local_C, rows_per_process);

    // Gather results from all processes
    MPI_Gather(local_C, rows_per_process * MAX_SIZE, MPI_INT, C, rows_per_process * MAX_SIZE, MPI_INT, 0, MPI_COMM_WORLD);

    // Finalize MPI
    MPI_Finalize();

    // Output results on root process
    if (world_rank == 0) {
        // Write output to a txt file
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
    }

    return 0;
}
