#include <torch/extension.h>

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <unordered_map>

namespace {
using Workspaces = std::unordered_map<cudaStream_t, at::Tensor>;
using DeviceWorkspaces = std::unordered_map<int, Workspaces>;

thread_local DeviceWorkspaces dequant_workspaces;
thread_local DeviceWorkspaces accumulator_workspaces;

at::Tensor get_workspace(
    DeviceWorkspaces& workspaces,
    const at::Tensor& like,
    int64_t numel,
    at::ScalarType dtype) {
    const int device = like.get_device();
    const cudaStream_t stream = c10::cuda::getCurrentCUDAStream(device).stream();
    auto& workspace = workspaces[device][stream];
    if (!workspace.defined() || workspace.numel() < numel) {
        workspace = at::empty({numel}, like.options().dtype(dtype));
    }
    return workspace.narrow(0, 0, numel);
}

at::Tensor get_dequant_workspace(const at::Tensor& like, int64_t numel) {
    return get_workspace(dequant_workspaces, like, numel, at::kHalf);
}

at::Tensor get_accumulator_workspace(const at::Tensor& like, int64_t numel) {
    return get_workspace(accumulator_workspaces, like, numel, at::kFloat);
}
}  // namespace

void launch_w8a16_gemm(
    const at::Tensor& x,
    const at::Tensor& qweight,
    const at::Tensor& scale,
    const c10::optional<at::Tensor>& bias,
    at::Tensor& output);
void launch_w8_dequantize(
    const at::Tensor& qweight,
    const at::Tensor& scale,
    at::Tensor& weight);
void launch_w8a16_cutlass(
    const at::Tensor& x,
    const at::Tensor& qweight,
    const at::Tensor& scale,
    const c10::optional<at::Tensor>& bias,
    at::Tensor& accumulator,
    at::Tensor& output);

at::Tensor w8a16_gemm(
    const at::Tensor& x,
    const at::Tensor& qweight,
    const at::Tensor& scale,
    const c10::optional<at::Tensor>& bias) {
    TORCH_CHECK(x.is_cuda() && qweight.is_cuda() && scale.is_cuda(),
                "x, qweight, and scale must be CUDA tensors");
    TORCH_CHECK(x.device() == qweight.device() && x.device() == scale.device(),
                "x, qweight, and scale must be on the same CUDA device");
    TORCH_CHECK(x.scalar_type() == at::kHalf, "x must be float16");
    TORCH_CHECK(qweight.scalar_type() == at::kChar, "qweight must be int8");
    TORCH_CHECK(scale.scalar_type() == at::kHalf, "scale must be float16");
    TORCH_CHECK(qweight.dim() == 2, "qweight must be two-dimensional");
    TORCH_CHECK(scale.dim() == 1 && scale.size(0) == qweight.size(0),
                "scale must contain one value per output channel");
    TORCH_CHECK(x.dim() >= 1 && x.size(-1) == qweight.size(1),
                "x and qweight have incompatible inner dimensions");
    TORCH_CHECK(x.size(-1) > 0, "the inner dimension must be nonzero");
    TORCH_CHECK(qweight.is_contiguous() && scale.is_contiguous(),
                "qweight and scale must be contiguous");
    if (bias.has_value()) {
        TORCH_CHECK(bias->is_cuda() && bias->scalar_type() == at::kHalf,
                    "bias must be a CUDA float16 tensor");
        TORCH_CHECK(bias->device() == x.device(),
                    "bias must be on the same CUDA device as x");
        TORCH_CHECK(bias->is_contiguous() && bias->numel() == qweight.size(0),
                    "bias must be contiguous with one value per output channel");
    }

    const c10::cuda::CUDAGuard device_guard(x.device());
    auto x_contiguous = x.contiguous();
    auto output_sizes = x.sizes().vec();
    output_sizes.back() = qweight.size(0);
    const auto M = x.numel() / x.size(-1);
    if (M == 0) {
        return at::empty(output_sizes, x.options());
    }
    if ((x.size(-1) % 16) != 0 || (qweight.size(0) % 4) != 0) {
        // The Tensor Core kernel uses 128-bit vectorized input accesses. Keep
        // arbitrary shapes correct instead of imposing hidden alignment rules
        // on this public operator.
        auto weight = get_dequant_workspace(x, qweight.numel()).view(qweight.sizes());
        launch_w8_dequantize(qweight, scale, weight);
        auto x_2d = x_contiguous.view({M, x.size(-1)});
        auto output_2d = at::mm(x_2d, weight.transpose(0, 1));
        if (bias.has_value()) output_2d.add_(*bias);
        return output_2d.view(output_sizes);
    }
    auto output = at::empty(output_sizes, x.options());
    if (M == 1) {
        launch_w8a16_gemm(x_contiguous, qweight, scale, bias, output);
    } else {
        auto accumulator = get_accumulator_workspace(
            x, M * qweight.size(0));
        launch_w8a16_cutlass(
            x_contiguous, qweight, scale, bias, accumulator, output);
    }
    return output;
}

TORCH_LIBRARY(rwkv_w8a16, m) {
    m.def("gemm(Tensor x, Tensor qweight, Tensor scale, Tensor? bias=None) -> Tensor");
}

TORCH_LIBRARY_IMPL(rwkv_w8a16, CUDA, m) {
    m.impl("gemm", &w8a16_gemm);
}
