#include <immintrin.h> // For AVX Intrinsics
#include <vector>
#include <iostream>
#include <chrono>

float dotProductAVX(const std::vector<int>& a, const std::vector<int>& b)
{
    size_t size = a.size();
    size_t i = 0;
    __m256 sum = _mm256_setzero_ps(); // Initialize sum to 0

    // Process 8 floats at a time
    for (; i + 7 < size; i += 8)
    {
        __m256 vecA = _mm256_loadu_ps(reinterpret_cast<const float*>(&a[i])); // Load 8 floats from vector a
        __m256 vecB = _mm256_loadu_ps(reinterpret_cast<const float*>(&b[i])); // Load 8 floats from vector b
        __m256 prod = _mm256_mul_ps(vecA, vecB); // Element-wise multiplication
        sum = _mm256_add_ps(sum, prod); // Accumulate the sum
    }

    // Horizontal sum of the 8 packed floats in sum
    float result[8];
    _mm256_storeu_ps(result, sum);
    float dotProduct = result[0] + result[1] + result[2] + result[3] + result[4] + result[5] + result[6] + result[7];

    // Handle the remaining elements
    for (; i < size; i++)
    {
        dotProduct += static_cast<float>(a[i]) * static_cast<float>(b[i]);
    }

    return dotProduct;
}

std::vector<int> generateVector(size_t size)
{
    std::vector<int> vec(size);
    for (long i = 0; i < static_cast<long>(size); ++i)
    {
        vec[i] = rand() % 100;
    }
    return vec;
}

int main()
{
    std::cout << "vector Size, Execution Time (ms)\n";
    for (size_t size = 16; size <= 65536; size *= 2)
    {
        // Generate two random vectors of the current size
        std::vector<int> vec1 = generateVector(size);
        std::vector<int> vec2 = generateVector(size);

        // Measure the execution time of the dot product
        auto start = std::chrono::high_resolution_clock::now();
        double result = dotProductAVX(vec1, vec2);
        auto end = std::chrono::high_resolution_clock::now();

        // Calculate elapsed time in microseconds
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        std::cout << "Vector size: " << size << ", Time taken: " << duration << " microseconds" << std::endl;
    }
    return 0;
}