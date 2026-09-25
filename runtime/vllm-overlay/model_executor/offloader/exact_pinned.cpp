// Experimental exact-size UVA backing. No quantization or tensor math changes.
#include <torch/extension.h>
#include <cuda_runtime_api.h>
#include <atomic>
#include <cstring>
#include <memory>

static std::atomic<int64_t> live_bytes{0};

at::Tensor allocate_copy(const at::Tensor& source) {
  TORCH_CHECK(source.device().is_cpu() && source.is_contiguous(),
              "exact pinned backing needs a contiguous CPU tensor");
  const auto bytes = source.numel() * source.element_size();
  TORCH_CHECK(bytes > 0, "empty tensors use the standard allocator");
  int device = 0;
  auto error = cudaGetDevice(&device);
  TORCH_CHECK(error == cudaSuccess, cudaGetErrorString(error));
  void* pointer = nullptr;
  error = cudaHostAlloc(&pointer, bytes, cudaHostAllocPortable | cudaHostAllocMapped);
  TORCH_CHECK(error == cudaSuccess, cudaGetErrorString(error));
  live_bytes += bytes;
  auto destroy = [device, bytes](void* p) noexcept {
    // A UVA tensor's C++ deleter keeps this CPU tensor alive. Synchronize the
    // owning device before releasing backing storage, including load-time
    // repacking. Never throw during shutdown if CUDA has already unloaded.
    int previous = device;
    cudaGetDevice(&previous);
    if (cudaSetDevice(device) == cudaSuccess) cudaDeviceSynchronize();
    cudaFreeHost(p);
    cudaSetDevice(previous);
    live_bytes -= bytes;
  };
  auto owner = std::shared_ptr<void>(pointer, destroy);
  auto output = at::from_blob(pointer, source.sizes(),
                             [owner](void*) {}, source.options());
  std::memcpy(output.data_ptr(), source.data_ptr(), bytes);
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("allocate_copy", &allocate_copy);
  module.def("live_bytes", []() { return live_bytes.load(); });
}
