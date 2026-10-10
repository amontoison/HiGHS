#pragma once

#if defined(CUPDLP_GPU) || defined(HIPDLP_GPU)
#include "gpu_backend.hpp"

// Layout of d_convergence_results_
#define IDX_PRIMAL_FEAS 0
#define IDX_DUAL_FEAS 1
#define IDX_PRIMAL_OBJ 2
#define IDX_DUAL_OBJ 3
#define IDX_FPE_PRIMAL_SQ 4
#define IDX_FPE_DUAL_SQ 5
#define IDX_FPE_CROSS 6
#define NUM_GPU_RESULTS 7

#ifdef __cplusplus
extern "C" {
#endif

void launchKernelUpdateX_wrapper(double* d_x_new, const double* d_x_old,
                                 const double* d_aty, const double* d_cost,
                                 const double* d_lower, const double* d_upper,
                                 double primal_step, int n_cols,
                                 gpuStream_t stream);

void launchKernelUpdateY_wrapper(double* d_y_new, const double* d_y_old,
                                 const double* d_ax_old, const double* d_ax_new,
                                 const double* d_row_lower,
                                 const bool* d_is_equality, double dual_step,
                                 int n_rows, gpuStream_t stream);

void launchKernelUpdateAverages_wrapper(double* d_x_sum, double* d_y_sum,
                                        const double* d_x_current,
                                        const double* d_y_current,
                                        double weight, int n_cols, int n_rows,
                                        gpuStream_t stream);

void launchKernelScaleVector_wrapper(double* d_out, const double* d_in,
                                     double scale, int n, gpuStream_t stream);

void launchCheckConvergenceKernels_wrapper(
    double* d_results, double* d_slack_pos, double* d_slack_neg,
    const double* d_x, const double* d_y, const double* d_ax,
    const double* d_aty, const double* d_col_cost, const double* d_row_lower,
    const double* d_col_lower, const double* d_col_upper,
    const bool* d_is_equality, const double* d_col_scale,
    const double* d_row_scale, int n_cols, int n_rows, bool use_halpern_slack,
    gpuStream_t stream);

void launchKernelHalpernPrimalMinor_wrapper(
    double* d_current_primal, double* d_reflected_primal,
    const double* d_dual_product, const double* d_objective,
    const double* d_var_lb, const double* d_var_ub, const double* d_step_size,
    int n,
    const double* d_anchor, const int* d_halpern_iteration, int k_offset,
    double reflection_coeff, gpuStream_t stream);

void launchKernelHalpernPrimalMajor_wrapper(
    double* d_current_primal, double* d_pdhg_primal,
    double* d_reflected_primal, const double* d_dual_product,
    const double* d_objective, const double* d_var_lb, const double* d_var_ub,
    const double* d_step_size, int n, double* d_dual_slack,
    const double* d_anchor, const int* d_halpern_iteration, int k_offset,
    double reflection_coeff, gpuStream_t stream);

void launchKernelHalpernDualMinor_wrapper(
    double* d_current_dual, double* d_reflected_dual,
    const double* d_primal_product, const double* d_rhs,
    const bool* d_is_equality, const double* d_step_size, int n,
    const double* d_anchor, const int* d_halpern_iteration, int k_offset,
    double reflection_coeff, gpuStream_t stream);

void launchKernelHalpernDualMajor_wrapper(
    double* d_current_dual, double* d_pdhg_dual, double* d_reflected_dual,
    const double* d_primal_product, const double* d_rhs,
    const bool* d_is_equality, const double* d_step_size, int n,
    const double* d_anchor, const int* d_halpern_iteration, int k_offset,
    double reflection_coeff, gpuStream_t stream);

void launchKernelFixedPointErrorDual_wrapper(double* d_delta_y,
                                             const double* d_pdhg_dual,
                                             const double* d_reflected_dual,
                                             double* d_results, int n_rows,
                                             gpuStream_t stream);

void launchKernelFixedPointErrorPrimal_wrapper(
    const double* d_pdhg_primal, const double* d_reflected_primal,
    const double* d_at_delta_y, double* d_results, int n_cols,
    gpuStream_t stream);

void launchKernelScaleByInverse_wrapper(double* d_x_out, const double* d_x_in,
                                        const double* d_norm, int n,
                                        gpuStream_t stream);

void launchKernelPowerResidual_wrapper(double* d_x, const double* d_y,
                                       const double* d_alpha, int n,
                                       gpuStream_t stream);

#ifdef __cplusplus
}
#endif
#endif