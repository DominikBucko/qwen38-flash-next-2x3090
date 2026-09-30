// GPU side of the cold-expert bridge: one polling thread, release/acquire at system scope.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>

namespace {
constexpr int H = 2560, TOPK = 10, MAXT = 4;

template <typename IdT>
__global__ void submit_kernel(const __nv_bfloat16* __restrict__ x, const IdT* __restrict__ ids,
                              const float* __restrict__ w, int T, char* req, long long* counter, int layer,
                              int bump, const signed char* __restrict__ mask) {
  __shared__ long long seq;
  if (threadIdx.x == 0) {
    long long c = *counter;
    if (bump) { c += 1; *counter = c; }
    seq = c * 64 + layer;
    *reinterpret_cast<int*>(req + 8) = T;
    *reinterpret_cast<int*>(req + 12) = layer;
  }
  int* ido = reinterpret_cast<int*>(req + 64);
  float* wo = reinterpret_cast<float*>(req + 224);
  for (int i = threadIdx.x; i < MAXT * TOPK; i += blockDim.x) {
    const bool v = i < T * TOPK && !(mask && mask[i]);
    ido[i] = v ? static_cast<int>(ids[i]) : -1;
    wo[i] = v ? w[i] : 0.f;
  }
  float* xo = reinterpret_cast<float*>(req + 448);
  for (int i = threadIdx.x; i < T * H; i += blockDim.x) xo[i] = __bfloat162float(x[i]);
  __threadfence_system();
  __syncthreads();
  if (threadIdx.x == 0)
    asm volatile("st.release.sys.global.b64 [%0], %1;" ::"l"(req), "l"(seq) : "memory");
}

__global__ void wait_add_kernel(__nv_bfloat16* out, int n, const char* resp, const long long* counter, int layer,
                                const float* __restrict__ extra) {
  // Every block polls once-per-128 ns with one thread, then reads its slice of the result with one
  // float4 per thread: PCIe reads of the 40 KB result proceed in parallel instead of 40 serial rounds.
  if (threadIdx.x == 0) {
    const long long expected = *reinterpret_cast<const volatile long long*>(counter) * 64 + layer;
    long long v;
    for (;;) {
      asm volatile("ld.acquire.sys.global.b64 %0, [%1];" : "=l"(v) : "l"(resp) : "memory");
      if (v == expected) break;
      __nanosleep(128);
    }
  }
  __syncthreads();
  const int i4 = blockIdx.x * blockDim.x + threadIdx.x;  // float4 index
  if (4 * i4 < n) {
    float4 y = __ldcv(reinterpret_cast<const float4*>(resp + 64) + i4);  // uncached: never a stale replay
    if (extra) {
      const float4 g = reinterpret_cast<const float4*>(extra)[i4];
      y.x += g.x; y.y += g.y; y.z += g.z; y.w += g.w;
    }
    __nv_bfloat16* o = out + 4 * i4;
    o[0] = __float2bfloat16(__bfloat162float(o[0]) + y.x);
    o[1] = __float2bfloat16(__bfloat162float(o[1]) + y.y);
    o[2] = __float2bfloat16(__bfloat162float(o[2]) + y.z);
    o[3] = __float2bfloat16(__bfloat162float(o[3]) + y.w);
  }
}
// ---- big slot (prefill chunks, eager): many blocks copy, one thread publishes; one thread waits, many add.
constexpr int MAXT_BIG = 512;
constexpr size_t BIG_IDS = 64, BIG_W = 64 + 4 * MAXT_BIG * TOPK, BIG_X = 64 + 8 * MAXT_BIG * TOPK;

template <typename IdT>
__global__ void submit_big_copy_kernel(const __nv_bfloat16* __restrict__ x, const IdT* __restrict__ ids,
                                       const float* __restrict__ w, int T, char* req) {
  const int stride = gridDim.x * blockDim.x;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int* ido = reinterpret_cast<int*>(req + BIG_IDS);
  float* wo = reinterpret_cast<float*>(req + BIG_W);
  for (int i = tid; i < T * TOPK; i += stride) {
    ido[i] = static_cast<int>(ids[i]);
    wo[i] = w[i];
  }
  float4* xo = reinterpret_cast<float4*>(req + BIG_X);
  const int n4 = T * H / 4;
  for (int i = tid; i < n4; i += stride) {
    const __nv_bfloat16* s = x + 4 * i;
    xo[i] = make_float4(__bfloat162float(s[0]), __bfloat162float(s[1]), __bfloat162float(s[2]),
                        __bfloat162float(s[3]));
  }
  __threadfence_system();
}

__global__ void publish_kernel(char* req, long long* counter, int T, int layer, int bump) {
  long long c = *counter;
  if (bump) { c += 1; *counter = c; }
  *reinterpret_cast<int*>(req + 8) = T;
  *reinterpret_cast<int*>(req + 12) = layer;
  __threadfence_system();
  const long long seq = c * 64 + layer;
  asm volatile("st.release.sys.global.b64 [%0], %1;" ::"l"(req), "l"(seq) : "memory");
}

__global__ void wait_kernel(const char* resp, const long long* counter, int layer) {
  const long long expected = *reinterpret_cast<const volatile long long*>(counter) * 64 + layer;
  long long v;
  for (;;) {
    asm volatile("ld.acquire.sys.global.b64 %0, [%1];" : "=l"(v) : "l"(resp) : "memory");
    if (v == expected) break;
    __nanosleep(1000);
  }
}

__global__ void add_f32_kernel(float* __restrict__ acc, int n4, const char* resp) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n4) {
    const float4 y = __ldcv(reinterpret_cast<const float4*>(resp + 64) + i);
    float4 a = reinterpret_cast<float4*>(acc)[i];
    a.x += y.x; a.y += y.y; a.z += y.z; a.w += y.w;
    reinterpret_cast<float4*>(acc)[i] = a;
  }
}
}  // namespace

bool event_done(int64_t event) {
  return cudaEventQuery(reinterpret_cast<cudaEvent_t>(event)) == cudaSuccess;
}

void bridge_submit_big(torch::Tensor x, torch::Tensor ids, torch::Tensor w, int64_t req_addr, torch::Tensor counter,
                       int64_t layer, bool bump) {
  TORCH_CHECK(x.dtype() == torch::kBFloat16 && x.is_contiguous() && ids.is_contiguous() && w.is_contiguous());
  TORCH_CHECK(ids.size(1) == TOPK && x.size(1) == H, "big submit: topk 10, hidden 2560");
  const int T = x.size(0);
  TORCH_CHECK(T >= 1 && T <= MAXT_BIG, "big submit: 1..512 tokens");
  auto stream = c10::cuda::getCurrentCUDAStream();
  char* req = reinterpret_cast<char*>(req_addr);
  auto* xp = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
  const int blocks = std::min(160, (T * H / 4 + 255) / 256);
  if (ids.dtype() == torch::kInt64)
    submit_big_copy_kernel<long long><<<blocks, 256, 0, stream>>>(
        xp, reinterpret_cast<const long long*>(ids.data_ptr<int64_t>()), w.data_ptr<float>(), T, req);
  else
    submit_big_copy_kernel<int><<<blocks, 256, 0, stream>>>(xp, ids.data_ptr<int32_t>(), w.data_ptr<float>(), T,
                                                            req);
  publish_kernel<<<1, 1, 0, stream>>>(req, reinterpret_cast<long long*>(counter.data_ptr<int64_t>()), T,
                                      static_cast<int>(layer), bump);
}

void bridge_wait_add_f32(torch::Tensor acc, int64_t resp_addr, torch::Tensor counter, int64_t layer) {
  TORCH_CHECK(acc.dtype() == torch::kFloat32 && acc.is_contiguous() && acc.numel() % 4 == 0);
  auto stream = c10::cuda::getCurrentCUDAStream();
  const char* resp = reinterpret_cast<const char*>(resp_addr);
  wait_kernel<<<1, 1, 0, stream>>>(resp, reinterpret_cast<const long long*>(counter.data_ptr<int64_t>()),
                                   static_cast<int>(layer));
  const int n4 = acc.numel() / 4;
  add_f32_kernel<<<(n4 + 255) / 256, 256, 0, stream>>>(acc.data_ptr<float>(), n4, resp);
}

void bridge_submit(torch::Tensor x, torch::Tensor ids, torch::Tensor w, int64_t req_addr, torch::Tensor counter,
                   int64_t layer, bool bump, int64_t mask_addr) {
  const signed char* mask = reinterpret_cast<const signed char*>(mask_addr);
  TORCH_CHECK(x.dtype() == torch::kBFloat16 && x.is_contiguous() && ids.is_contiguous() && w.is_contiguous());
  const int T = x.size(0);
  auto stream = c10::cuda::getCurrentCUDAStream();
  char* req = reinterpret_cast<char*>(req_addr);
  auto* xp = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
  auto* cp = reinterpret_cast<long long*>(counter.data_ptr<int64_t>());
  if (ids.dtype() == torch::kInt64)
    submit_kernel<long long><<<1, 256, 0, stream>>>(xp, reinterpret_cast<const long long*>(ids.data_ptr<int64_t>()),
                                                    w.data_ptr<float>(), T, req, cp, static_cast<int>(layer), bump,
                                                    mask);
  else
    submit_kernel<int><<<1, 256, 0, stream>>>(xp, ids.data_ptr<int32_t>(), w.data_ptr<float>(), T, req, cp,
                                              static_cast<int>(layer), bump, mask);
}

void bridge_wait_add(torch::Tensor out, int64_t resp_addr, torch::Tensor counter, int64_t layer,
                     int64_t extra_addr) {
  TORCH_CHECK(out.dtype() == torch::kBFloat16 && out.is_contiguous());
  auto stream = c10::cuda::getCurrentCUDAStream();
  const int blocks = (out.numel() / 4 + 255) / 256;
  wait_add_kernel<<<blocks, 256, 0, stream>>>(reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), out.numel(),
                                         reinterpret_cast<const char*>(resp_addr),
                                         reinterpret_cast<const long long*>(counter.data_ptr<int64_t>()),
                                         static_cast<int>(layer), reinterpret_cast<const float*>(extra_addr));
}
