#include <iostream>
#include <vector>
#include <chrono>
#include <immintrin.h> // For SSE intrinsics
#include <cmath> // For std::pow
#include <stdexcept>

// Function to compute the dot product using DPPS (SSE)
float computeDotProductSSE(const std::vector<float>& vecA, const std::vector<float>& vecB) {
    if (vecA.size() != vecB.size()) {
        throw std::invalid_argument("Vectors must have the same size");
    }

    const size_t vectorSize = vecA.size();
    const size_t simdWidth = 4; // DPPS processes 4 floats at a time
    __m128 sum = _mm_setzero_ps(); // Initialize the sum to zero

    // Process vectors in chunks of 4 floats
    size_t i = 0;
    for (; i + simdWidth <= vectorSize; i += simdWidth) {
        __m128 a = _mm_loadu_ps(&vecA[i]); // Load 4 floats from vecA
        __m128 b = _mm_loadu_ps(&vecB[i]); // Load 4 floats from vecB
        __m128 dp = _mm_dp_ps(a, b, 0xF1); // Dot product (mask: all 4 elements)
        sum = _mm_add_ps(sum, dp);         // Accumulate results
    }

    // Sum up the results from the SIMD registers
    float result[4];
    _mm_storeu_ps(result, sum);
    float dotProduct = result[0];

    // Handle remaining elements (tail processing)
    for (; i < vectorSize; ++i) {
        dotProduct += vecA[i] * vecB[i];
    }

    return dotProduct;
}

int main() {
    // Vector sizes as powers of 2
    std::vector<size_t> vectorSizes;
    for (int i = 4; i <= 16; ++i) { // 2^4 = 16 to 2^16 = 65536
        vectorSizes.push_back(static_cast<size_t>(std::pow(2, i)));
    }

    // High-resolution timer alias
    using namespace std::chrono;

    std::cout << "SSE DPPS Dot Product Performance:\n";
    std::cout << "Vector Size\tExecution Time (microseconds)\n";

    // Test for each vector size
    for (const size_t size : vectorSizes) {
        // Initialize vectors with arbitrary values
        std::vector<float> vecA(size, 1.0f);
        std::vector<float> vecB(size, 2.0f);

        // Measure execution time
        auto start = high_resolution_clock::now();
        float result = computeDotProductSSE(vecA, vecB);
        auto end = high_resolution_clock::now();

        // Calculate elapsed time in milliseconds
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        std::cout << "Vector size: " << size << ", Time taken: " << duration << " microseconds" << std::endl;

    }
    return 0;
}
