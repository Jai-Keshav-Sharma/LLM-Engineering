
#include <iostream>
#include <iomanip>
#include <vector>
#include <limits>
#include <chrono>

// Linear Congruential Generator
uint32_t lcg(uint32_t& seed, const uint32_t a = 1664525, const uint32_t c = 1013904223, const uint32_t m = 4294967296) {
    seed = (a * seed + c) % m;
    return seed;
}

// Function to find the maximum subarray sum
int64_t max_subarray_sum(const size_t n, uint32_t seed, const int min_val, const int max_val) {
    std::vector<int> random_numbers(n);
    for (size_t i = 0; i < n; ++i) {
        random_numbers[i] = lcg(seed) % (max_val - min_val + 1) + min_val;
    }
    int64_t max_sum = std::numeric_limits<int64_t>::min();
    for (size_t i = 0; i < n; ++i) {
        int64_t current_sum = 0;
        for (size_t j = i; j < n; ++j) {
            current_sum += random_numbers[j];
            if (current_sum > max_sum) {
                max_sum = current_sum;
            }
        }
    }
    return max_sum;
}

// Function to calculate total maximum subarray sum over 20 runs
int64_t total_max_subarray_sum(const size_t n, uint32_t initial_seed, const int min_val, const int max_val) {
    int64_t total_sum = 0;
    for (int run = 0; run < 20; ++run) {
        total_sum += max_subarray_sum(n, initial_seed, min_val, max_val);
        initial_seed = lcg(initial_seed); // Update seed for the next run
    }
    return total_sum;
}

int main() {
    const size_t n = 10000; // Number of random numbers
    uint32_t initial_seed = 42; // Initial seed for the LCG
    const int min_val = -10; // Minimum value of random numbers
    const int max_val = 10; // Maximum value of random numbers

    auto start_time = std::chrono::high_resolution_clock::now();
    int64_t result = total_max_subarray_sum(n, initial_seed, min_val, max_val);
    auto end_time = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end_time - start_time;

    std::cout << "Total Maximum Subarray Sum (20 runs): " << result << std::endl;
    std::cout << "Execution Time: " << std::fixed << std::setprecision(6) << elapsed.count() << " seconds" << std::endl;

    return 0;
}
