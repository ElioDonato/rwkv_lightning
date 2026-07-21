#undef __CUDA_NO_HALF_OPERATORS__
#undef __CUDA_NO_HALF_CONVERSIONS__

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include <cuda_fp16.h>
#include <mma.h>

#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/layout/matrix.h"

namespace wmma = nvcuda::wmma;

constexpr int BLOCK_N = 64;
constexpr int BLOCK_K = 64;

__global__ void w8_dequantize_kernel(
    const int8_t* __restrict__ qweight,
    const half* __restrict__ scale,
    half* __restrict__ weight,
    int N,
    int K) {
    const int n = blockIdx.x;
    const half row_scale = scale[n];
    for (int k = threadIdx.x; k < K; k += blockDim.x) {
        weight[n * K + k] = __hmul(
            __int2half_rn(static_cast<int>(qweight[n * K + k])), row_scale);
    }
}

void launch_w8_dequantize(
    const at::Tensor& qweight,
    const at::Tensor& scale,
    at::Tensor& weight) {
    const int N = static_cast<int>(qweight.size(0));
    const int K = static_cast<int>(qweight.size(1));
    constexpr int THREADS = 256;
    w8_dequantize_kernel<<<N, THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
        qweight.data_ptr<int8_t>(),
        reinterpret_cast<const half*>(scale.data_ptr<at::Half>()),
        reinterpret_cast<half*>(weight.data_ptr<at::Half>()), N, K);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

namespace {

using CutlassW8A16Gemm = cutlass::gemm::device::GemmUniversal<
    cutlass::half_t,
    cutlass::layout::RowMajor,
    int8_t,
    cutlass::layout::ColumnMajor,
    float,
    cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<32, 32, 64>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<
        float, 4, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    4,
    8,
    16,
    cutlass::arch::OpMultiplyAddMixedInputUpcast,
    cutlass::ComplexTransform::kNone,
    cutlass::ComplexTransform::kNone>;

template <typename Gemm>
cutlass::Status run_cutlass_w8a16(
    const at::Tensor& x,
    const at::Tensor& qweight,
    at::Tensor& accumulator,
    int M,
    int N,
    int K,
    cudaStream_t stream) {
    typename Gemm::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K},
        1,
        {1.0f, 0.0f},
        x.data_ptr<at::Half>(),
        qweight.data_ptr<int8_t>(),
        accumulator.data_ptr<float>(),
        accumulator.data_ptr<float>(),
        0, 0, 0, 0,
        K, K, N, N};
    Gemm gemm;
    return gemm(arguments, nullptr, stream);
}

__global__ void w8a16_scale_output_kernel(
    const float* __restrict__ accumulator,
    const half* __restrict__ scale,
    const half* __restrict__ bias,
    half* __restrict__ output,
    int elements,
    int N) {
    for (int index = blockIdx.x * blockDim.x + threadIdx.x;
         index < elements;
         index += blockDim.x * gridDim.x) {
        const int n = index % N;
        float value = accumulator[index] * __half2float(scale[n]);
        if (bias != nullptr) value += __half2float(bias[n]);
        output[index] = __float2half_rn(value);
    }
}

}  // namespace

void launch_w8a16_cutlass(
    const at::Tensor& x,
    const at::Tensor& qweight,
    const at::Tensor& scale,
    const c10::optional<at::Tensor>& bias,
    at::Tensor& accumulator,
    at::Tensor& output) {
    const int K = static_cast<int>(x.size(-1));
    const int N = static_cast<int>(qweight.size(0));
    const int M = static_cast<int>(x.numel() / K);
    const auto stream = at::cuda::getCurrentCUDAStream();

    const cutlass::Status status = run_cutlass_w8a16<CutlassW8A16Gemm>(
        x, qweight, accumulator, M, N, K, stream);
    TORCH_CHECK(
        status == cutlass::Status::kSuccess,
        "CUTLASS W8A16 GEMM failed: ", cutlassGetStatusString(status));

    const half* bias_ptr = bias.has_value()
        ? reinterpret_cast<const half*>(bias->data_ptr<at::Half>())
        : nullptr;
    const int elements = M * N;
    constexpr int THREADS = 256;
    const int blocks = (elements + THREADS - 1) / THREADS;
    w8a16_scale_output_kernel<<<blocks, THREADS, 0, stream>>>(
        accumulator.data_ptr<float>(),
        reinterpret_cast<const half*>(scale.data_ptr<at::Half>()),
        bias_ptr,
        reinterpret_cast<half*>(output.data_ptr<at::Half>()),
        elements,
        N);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int MAX_M>
__global__ void w8a16_gemv_kernel(
    const half* __restrict__ x,
    const int8_t* __restrict__ qweight,
    const half* __restrict__ scale,
    const half* __restrict__ bias,
    half* __restrict__ output,
    int M,
    int N,
    int K) {
    constexpr int THREADS = 256;
    constexpr int WARPS = THREADS / 32;
    __shared__ float warp_sums[MAX_M][WARPS];
    const int n = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    float sums[MAX_M];
    #pragma unroll
    for (int m = 0; m < MAX_M; ++m) sums[m] = 0.0f;

    for (int k = tid; k < K; k += THREADS) {
        const float weight = static_cast<float>(qweight[n * K + k]);
        #pragma unroll
        for (int m = 0; m < MAX_M; ++m) {
            if (m < M) sums[m] += __half2float(x[m * K + k]) * weight;
        }
    }
    #pragma unroll
    for (int m = 0; m < MAX_M; ++m) {
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            sums[m] += __shfl_down_sync(0xffffffff, sums[m], offset);
        if (lane == 0) warp_sums[m][warp] = sums[m];
    }
    __syncthreads();
    if (tid < M) {
        float value = 0.0f;
        #pragma unroll
        for (int w = 0; w < WARPS; ++w) value += warp_sums[tid][w];
        value *= __half2float(scale[n]);
        if (bias != nullptr) value += __half2float(bias[n]);
        output[tid * N + n] = __float2half_rn(value);
    }
}

template <int BLOCK_M>
__global__ void w8a16_gemm_kernel(
    const half* __restrict__ x,
    const int8_t* __restrict__ qweight,
    const half* __restrict__ scale,
    const half* __restrict__ bias,
    half* __restrict__ output,
    int M,
    int N,
    int K) {
    constexpr int WARP_M_TILES = BLOCK_M == 64 ? 2 : 1;
    constexpr int WARP_N_TILES = BLOCK_M >= 32 ? 2 : 1;
    constexpr int WARPS_M = BLOCK_M / (16 * WARP_M_TILES);
    constexpr int WARPS_N = BLOCK_N / (16 * WARP_N_TILES);
    constexpr int NUM_WARPS = WARPS_M * WARPS_N;
    constexpr int NUM_THREADS = NUM_WARPS * 32;

    __shared__ half x_tile[BLOCK_M][BLOCK_K];
    __shared__ half weight_tile[BLOCK_N][BLOCK_K];
    __shared__ float output_tile[BLOCK_M][BLOCK_N];

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int warp_m = warp_id / WARPS_N;
    const int warp_n = warp_id % WARPS_N;
    const int block_m = blockIdx.y * BLOCK_M;
    const int block_n = blockIdx.x * BLOCK_N;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float>
        accumulator[WARP_M_TILES][WARP_N_TILES];
    #pragma unroll
    for (int wm = 0; wm < WARP_M_TILES; ++wm)
        #pragma unroll
        for (int wn = 0; wn < WARP_N_TILES; ++wn)
            wmma::fill_fragment(accumulator[wm][wn], 0.0f);

    for (int block_k = 0; block_k < K; block_k += BLOCK_K) {
        for (int index = tid; index < BLOCK_M * BLOCK_K; index += NUM_THREADS) {
            const int row = index / BLOCK_K;
            const int col = index % BLOCK_K;
            const int global_m = block_m + row;
            const int global_k = block_k + col;
            x_tile[row][col] =
                (global_m < M && global_k < K) ? x[global_m * K + global_k] : __float2half(0.0f);
        }
        constexpr int VECTOR_BYTES = 16;
        constexpr int WEIGHT_VECTORS = BLOCK_K * BLOCK_N / VECTOR_BYTES;
        #pragma unroll
        for (int vector_index = tid; vector_index < WEIGHT_VECTORS;
             vector_index += NUM_THREADS) {
            const int scalar_index = vector_index * VECTOR_BYTES;
            const int col = scalar_index / BLOCK_K;
            const int row = scalar_index % BLOCK_K;
            const int global_k = block_k + row;
            const int global_n = block_n + col;
            int4 packed = make_int4(0, 0, 0, 0);
            const bool full_vector = global_n < N && global_k + VECTOR_BYTES <= K;
            if (full_vector) {
                packed = *reinterpret_cast<const int4*>(
                    qweight + global_n * K + global_k);
            }
            const int8_t* values = reinterpret_cast<const int8_t*>(&packed);
            #pragma unroll
            for (int i = 0; i < VECTOR_BYTES; ++i)
                weight_tile[col][row + i] = full_vector
                    ? __int2half_rn(static_cast<int>(values[i]))
                    : ((global_n < N && global_k + i < K)
                       ? __int2half_rn(static_cast<int>(
                             qweight[global_n * K + global_k + i]))
                       : __float2half(0.0f));
        }
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BLOCK_K; k += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major>
                a[WARP_M_TILES];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major>
                b[WARP_N_TILES];
            #pragma unroll
            for (int wm = 0; wm < WARP_M_TILES; ++wm)
                wmma::load_matrix_sync(
                    a[wm], &x_tile[(warp_m * WARP_M_TILES + wm) * 16][k], BLOCK_K);
            #pragma unroll
            for (int wn = 0; wn < WARP_N_TILES; ++wn) {
                wmma::load_matrix_sync(
                    b[wn],
                    &weight_tile[(warp_n * WARP_N_TILES + wn) * 16][k],
                    BLOCK_K);
            }
            #pragma unroll
            for (int wm = 0; wm < WARP_M_TILES; ++wm)
                #pragma unroll
                for (int wn = 0; wn < WARP_N_TILES; ++wn)
                    wmma::mma_sync(
                        accumulator[wm][wn], a[wm], b[wn], accumulator[wm][wn]);
        }
        __syncthreads();
    }

    #pragma unroll
    for (int wm = 0; wm < WARP_M_TILES; ++wm)
        #pragma unroll
        for (int wn = 0; wn < WARP_N_TILES; ++wn)
            wmma::store_matrix_sync(
                &output_tile[(warp_m * WARP_M_TILES + wm) * 16]
                            [(warp_n * WARP_N_TILES + wn) * 16],
                accumulator[wm][wn], BLOCK_N, wmma::mem_row_major);
    __syncthreads();

    for (int index = tid; index < BLOCK_M * BLOCK_N; index += NUM_THREADS) {
        const int row = index / BLOCK_N;
        const int col = index % BLOCK_N;
        const int global_m = block_m + row;
        const int global_n = block_n + col;
        if (global_m < M && global_n < N) {
            float value = output_tile[row][col] * __half2float(scale[global_n]);
            if (bias != nullptr) value += __half2float(bias[global_n]);
            output[global_m * N + global_n] = __float2half_rn(value);
        }
    }
}

void launch_w8a16_gemm(
    const at::Tensor& x,
    const at::Tensor& qweight,
    const at::Tensor& scale,
    const c10::optional<at::Tensor>& bias,
    at::Tensor& output) {
    const int64_t K64 = x.size(-1);
    const int64_t N64 = qweight.size(0);
    const int64_t M64 = x.numel() / K64;
    TORCH_CHECK(M64 <= INT_MAX && N64 <= INT_MAX && K64 <= INT_MAX,
                "W8A16 GEMM dimensions exceed int32 range");
    const int M = static_cast<int>(M64);
    const int N = static_cast<int>(N64);
    const int K = static_cast<int>(K64);
    const half* bias_ptr = bias.has_value()
        ? reinterpret_cast<const half*>(bias->data_ptr<at::Half>())
        : nullptr;
    const auto stream = at::cuda::getCurrentCUDAStream();

    if (M == 1) {
        const dim3 block(256);
        w8a16_gemv_kernel<1><<<N, block, 0, stream>>>(
            reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
            qweight.data_ptr<int8_t>(),
            reinterpret_cast<const half*>(scale.data_ptr<at::Half>()),
            bias_ptr,
            reinterpret_cast<half*>(output.data_ptr<at::Half>()), M, N, K);
    } else if (M <= 16) {
        const dim3 grid((N + BLOCK_N - 1) / BLOCK_N);
        const dim3 block(128);
        const dim3 launch_grid(grid.x, (M + 15) / 16);
        w8a16_gemm_kernel<16><<<launch_grid, block, 0, stream>>>(
            reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
            qweight.data_ptr<int8_t>(),
            reinterpret_cast<const half*>(scale.data_ptr<at::Half>()),
            bias_ptr,
            reinterpret_cast<half*>(output.data_ptr<at::Half>()), M, N, K);
    } else if (M <= 32) {
        const dim3 grid((N + BLOCK_N - 1) / BLOCK_N);
        const dim3 block(128);
        const dim3 launch_grid(grid.x, (M + 31) / 32);
        w8a16_gemm_kernel<32><<<launch_grid, block, 0, stream>>>(
            reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
            qweight.data_ptr<int8_t>(),
            reinterpret_cast<const half*>(scale.data_ptr<at::Half>()),
            bias_ptr,
            reinterpret_cast<half*>(output.data_ptr<at::Half>()), M, N, K);
    } else {
        const dim3 grid((N + BLOCK_N - 1) / BLOCK_N);
        const dim3 block(128);
        const dim3 launch_grid(grid.x, (M + 63) / 64);
        w8a16_gemm_kernel<64><<<launch_grid, block, 0, stream>>>(
            reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
            qweight.data_ptr<int8_t>(),
            reinterpret_cast<const half*>(scale.data_ptr<at::Half>()),
            bias_ptr,
            reinterpret_cast<half*>(output.data_ptr<at::Half>()), M, N, K);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
