#include <iostream>
#include <vector>
#include <chrono>
#include <cstddef>
#include <omp.h>  // Include OpenMP header

// Function to compute the dot product of two vectors
double computeDotProduct(const std::vector<int>& vec1, const std::vector<int>& vec2) {
    double result = 0.0;
    // Loop through the vectors and calculate the dot product

    // Parallelized loop using OpenMP
    // #pragma omp parallel for reduction(+:result)
    for (long i = 0; i < static_cast<long>(vec1.size()); ++i) { // Use signed 'long'
        result += vec1[i] * vec2[i];
    }
    return result;
}

std::vector<int> generateVector1(size_t size) {
    std::vector<int> vec(size);
    for (long i = 0; i < static_cast<long>(size); ++i) {  // Use signed 'long'
        vec[i] = rand() % 100;  // Random values between 0 and 99
    }
    return vec;
}

int main() {

    std::cout << "vector Size, Execution Time (ms)\n";

    for (size_t size = 16; size <= 65536; size *= 2) {

        // Generate two random vectors of the current size
        std::vector<int> vec1 = generateVector1(size);
        std::vector<int> vec2 = generateVector1(size);

        // Measure the execution time of the dot product
        auto start = std::chrono::high_resolution_clock::now();
        double result = computeDotProduct(vec1, vec2);
        auto end = std::chrono::high_resolution_clock::now();

        // Calculate elapsed time in milliseconds
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        std::cout << "Vector size: " << size << ", Time taken: " << duration << " microseconds" << std::endl;
    }

    return 0;
}
