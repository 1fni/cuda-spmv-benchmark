#ifndef VISUALIZATION_H
#define VISUALIZATION_H

#include "io.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Generate sparsity pattern visualization
 * @param matrix Matrix data to visualize
 * @param output_filename Output PNG filename
 * @param max_display_size Maximum matrix size to display (larger matrices are subsampled)
 * @return 0 on success, non-zero on error
 */
int generate_sparsity_pattern(const MatrixData* matrix, const char* output_filename, int max_display_size);

/**
 * @brief Generate performance comparison chart
 * @param results Array of benchmark results
 * @param num_results Number of results
 * @param output_filename Output PNG filename
 * @param chart_title Chart title
 * @return 0 on success, non-zero on error
 */
int generate_performance_chart(const char** operator_names, const double* gflops_values, 
                              const double* bandwidth_values, int num_results,
                              const char* output_filename, const char* chart_title);

/**
 * @brief Generate scaling analysis chart
 * @param sizes Array of matrix sizes
 * @param performance_data Array of performance values for each size
 * @param num_points Number of data points
 * @param output_filename Output PNG filename
 * @param operator_name Operator name for chart title
 * @return 0 on success, non-zero on error
 */
int generate_scaling_chart(const int* sizes, const double* performance_data, int num_points,
                          const char* output_filename, const char* operator_name);

#ifdef __cplusplus
}
#endif

#endif // VISUALIZATION_H