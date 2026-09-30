// CPU compute for "cold" MoE experts of Qwen3.8-Flash-Next (decode, 1..4 tokens).
// Weights are the checkpoint's AutoRound GPTQ INT4 tensors, symmetric (zero = 8), group 128:
//   qweight int32 [K/8, N] (8 consecutive k per int32, low nibble first), scales fp16 [K/128, N].
//   gate/up: K = 2560 -> N = 640;  down: K = 640 -> N = 2560.
// Activations stay FP32 (no activation quantization); accumulation is FP32.
#include <torch/extension.h>
#include <immintrin.h>
#include <atomic>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <thread>
#include <pthread.h>
#include <sched.h>
#include <sys/mman.h>
#include <cerrno>
#include <fcntl.h>
#include <unistd.h>
#include <vector>

namespace {

constexpr int H = 2560;   // hidden
constexpr int I = 640;    // expert intermediate
constexpr int GS = 128;   // quant group
constexpr int MAXT = 4;   // tokens per decode verify step handled here

static inline __m256 load_h8(const uint16_t* p) {
  return _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p)));
}

#define Q_NIB(q, j) _mm256_cvtepi32_ps(_mm256_and_si256(_mm256_srli_epi32(q, 4 * (j)), mask))

// Row-major streaming GEMV over columns [n0, n1) (n1 - n0 <= MAXC, multiple of 32):
//   y[t][n] = sum_k x[t][k] * (q[k][n] - 8) * s[k / 128][n]
// Each packed row is read once, left to right; per-group partial sums live in L1.
constexpr int MAXC = 640;
#ifndef PF_ROWS
#define PF_ROWS 6
#endif
constexpr int PF = PF_ROWS;  // software prefetch distance in packed rows
template <int T>
static void gemv_rows(const int32_t* __restrict qw, const uint16_t* __restrict sc, int K, int N, int n0, int n1,
                      const float* __restrict x, const float* __restrict xg, float* __restrict y, int ldy) {
  alignas(32) float tot[T][MAXC];
  alignas(32) float grp[T][MAXC];
  const int W = n1 - n0;
  for (int t = 0; t < T; ++t) std::memset(tot[t], 0, sizeof(float) * W);
  const __m256i mask = _mm256_set1_epi32(0xF);
  const int groups = K / GS;
  for (int g = 0; g < groups; ++g) {
    for (int t = 0; t < T; ++t) std::memset(grp[t], 0, sizeof(float) * W);
    const int r_end = (g + 1) * (GS / 8);
    for (int r = g * (GS / 8); r < r_end; ++r) {
      const int32_t* row = qw + static_cast<size_t>(r) * N + n0;
      const float* xr = x + r * 8;
      if (PF > 0 && r + PF < K / 8) {
        const char* ahead = reinterpret_cast<const char*>(row + static_cast<size_t>(PF) * N);
        for (int b = 0; b < W * 4; b += 64) _mm_prefetch(ahead + b, _MM_HINT_T0);
      }
      for (int c = 0; c < W; c += 32) {
        const __m256i q0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(row + c));
        const __m256i q1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(row + c + 8));
        const __m256i q2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(row + c + 16));
        const __m256i q3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(row + c + 24));
        __m256 a[T][4];
        for (int t = 0; t < T; ++t)
          for (int u = 0; u < 4; ++u) a[t][u] = _mm256_load_ps(&grp[t][c + 8 * u]);
#define STEP(j)                                                          \
  {                                                                      \
    const __m256 v0 = Q_NIB(q0, j), v1 = Q_NIB(q1, j);                   \
    const __m256 v2 = Q_NIB(q2, j), v3 = Q_NIB(q3, j);                   \
    for (int t = 0; t < T; ++t) {                                        \
      const __m256 xb = _mm256_broadcast_ss(xr + t * K + (j));           \
      a[t][0] = _mm256_fmadd_ps(v0, xb, a[t][0]);                        \
      a[t][1] = _mm256_fmadd_ps(v1, xb, a[t][1]);                        \
      a[t][2] = _mm256_fmadd_ps(v2, xb, a[t][2]);                        \
      a[t][3] = _mm256_fmadd_ps(v3, xb, a[t][3]);                        \
    }                                                                    \
  }
        STEP(0) STEP(1) STEP(2) STEP(3) STEP(4) STEP(5) STEP(6) STEP(7)
#undef STEP
        for (int t = 0; t < T; ++t)
          for (int u = 0; u < 4; ++u) _mm256_store_ps(&grp[t][c + 8 * u], a[t][u]);
      }
    }
    const uint16_t* srow = sc + static_cast<size_t>(g) * N + n0;
    for (int c = 0; c < W; c += 8) {
      const __m256 sv = load_h8(srow + c);
      for (int t = 0; t < T; ++t) {
        const __m256 z = _mm256_set1_ps(8.0f * xg[t * groups + g]);
        _mm256_store_ps(&tot[t][c], _mm256_fmadd_ps(sv, _mm256_sub_ps(_mm256_load_ps(&grp[t][c]), z),
                                                    _mm256_load_ps(&tot[t][c])));
      }
    }
  }
  for (int t = 0; t < T; ++t) std::memcpy(y + static_cast<size_t>(t) * ldy, tot[t], sizeof(float) * W);
}

static void gemv_cols(const int32_t* qw, const uint16_t* sc, int K, int N, int n0, int n1, int Tn,
                      const float* x, const float* xg, float* y, int ldy) {
  for (int a = n0; a < n1; a += MAXC) {
    const int b = std::min(n1, a + MAXC);
    switch (Tn) {
      case 1: gemv_rows<1>(qw, sc, K, N, a, b, x, xg, y + (a - n0), ldy); break;
      case 2: gemv_rows<2>(qw, sc, K, N, a, b, x, xg, y + (a - n0), ldy); break;
      case 3: gemv_rows<3>(qw, sc, K, N, a, b, x, xg, y + (a - n0), ldy); break;
      default: gemv_rows<4>(qw, sc, K, N, a, b, x, xg, y + (a - n0), ldy); break;
    }
  }
}

static void group_sums(const float* x, int T, int K, float* xg) {
  const int groups = K / GS;
  for (int t = 0; t < T; ++t)
    for (int g = 0; g < groups; ++g) {
      float s = 0.f;
      for (int k = 0; k < GS; ++k) s += x[t * K + g * GS + k];
      xg[t * groups + g] = s;
    }
}

// ---------------------------------------------------------------- thread pool
class Pool {
 public:
  // ~1 ms of pause-spinning covers the GPU work between two layers of one decode step.
  static constexpr int kSpin = 1 << 16;
  explicit Pool(int n, const std::vector<int>& cpus) : n_(n) {
    for (int i = 1; i < n_; ++i) threads_.emplace_back([this, i, cpus] {
      if (static_cast<int>(cpus.size()) > i) {
        cpu_set_t set; CPU_ZERO(&set); CPU_SET(cpus[i], &set);
        pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
      }
      loop(i);
    });
    if (!cpus.empty()) {
      cpu_set_t set; CPU_ZERO(&set); CPU_SET(cpus[0], &set);
      pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
    }
  }
  int id_of_caller() const { return 0; }
  ~Pool() {
    stop_.store(true, std::memory_order_release);
    epoch_.fetch_add(1, std::memory_order_acq_rel);
    epoch_.notify_all();
    for (auto& t : threads_) t.join();
  }
  int size() const { return n_; }
  // Run fn(item) for item in [0, count) on all threads; returns when done.
  template <class F>
  void run(int count, F&& fn) {
    fn_ = [&fn](int i, int w) { fn(i, w); };
    count_ = count;
    next_.store(0, std::memory_order_relaxed);
    done_.store(0, std::memory_order_relaxed);
    epoch_.fetch_add(1, std::memory_order_acq_rel);
    epoch_.notify_all();
    work();
    while (done_.load(std::memory_order_acquire) != n_ - 1) _mm_pause();
  }

 private:
  void work(int w = 0) {
    for (;;) {
      const int i = next_.fetch_add(1, std::memory_order_relaxed);
      if (i >= count_) break;
      fn_(i, w);
    }
  }
  void loop(int id) {
    // Start from the construction epoch (0), not the current one: a worker that starts
    // late must still serve a run() posted before it began spinning.
    uint64_t seen = 0;
    for (;;) {
      uint64_t e;
      int spins = 0;
      while ((e = epoch_.load(std::memory_order_acquire)) == seen) {
        if (++spins < kSpin) { _mm_pause(); continue; }
        epoch_.wait(seen, std::memory_order_acquire);  // idle: sleep instead of burning a core
      }
      seen = e;
      if (stop_.load(std::memory_order_acquire)) return;
      work(id);
      done_.fetch_add(1, std::memory_order_acq_rel);
    }
  }
  int n_;
  std::vector<std::thread> threads_;
  alignas(64) std::atomic<uint64_t> epoch_{0};
  alignas(64) std::atomic<int> next_{0};
  alignas(64) std::atomic<int> done_{0};
  alignas(64) std::atomic<bool> stop_{false};
  int count_ = 0;
  std::function<void(int, int)> fn_;
};

Pool* g_pool = nullptr;

}  // namespace

// Tail experts (not in the arena) are read through a file mapping: ask the kernel to start reading all of
// a layer's tail experts at once (MADV_WILLNEED), then compute the resident experts first.
const uint8_t* g_tail_mask = nullptr;  // [L, E], 1 = tail expert (set_tail_mask)
std::atomic<int64_t> g_tail_jobs{0}, g_cold_jobs{0};  // decode: expert jobs served from the tail / in total
void set_tail_mask(torch::Tensor mask) {
  static torch::Tensor keep;
  keep = mask.contiguous();
  g_tail_mask = keep.data_ptr<uint8_t>();
}
static void prefetch_expert(const int64_t* e) {
  static const size_t sizes[6] = {819200, 25600, 819200, 25600, 819200, 25600};
  for (int i = 0; i < 6; ++i) {
    const uintptr_t a = static_cast<uintptr_t>(e[i]), lo = a & ~uintptr_t(4095);
    madvise(reinterpret_cast<void*>(lo), sizes[i] + (a - lo), MADV_WILLNEED);
  }
}

// Raw-pointer core: out[T][H] = sum over non-skipped (t, k) of w[t,k] * expert(ids[t,k], x[t]).
struct Job { int e, n; int tok[MAXT]; float wt[MAXT]; };
// One work item per expert by default (fastest in the trace-driven bench); set_slices() splits further.
int g_slices[I / GS + 1] = {0, 640};
int g_nslices = 1;
void set_slices(std::vector<int64_t> b) {
  TORCH_CHECK(b.size() >= 2 && b.front() == 0 && b.back() == I, "slices must span 0..640");
  for (size_t i = 0; i < b.size(); ++i) {
    TORCH_CHECK(b[i] % GS == 0, "slice bounds must be multiples of 128");
    g_slices[i] = static_cast<int>(b[i]);
  }
  g_nslices = static_cast<int>(b.size()) - 1;
}

static void compute_layer(const int64_t* tab, int E, const uint8_t* sk, int T, int topk, const int32_t* idp,
                          const float* wp, const float* xp, float* op, float* sp, size_t scratch_floats,
                          const uint8_t* tm = nullptr) {
  Job jobs[MAXT * 16];
  int J = 0;
  for (int t = 0; t < T; ++t)
    for (int k = 0; k < topk; ++k) {
      const int e = idp[t * topk + k];
      if (e < 0 || e >= E || sk[e]) continue;
      int j = 0;
      while (j < J && jobs[j].e != e) ++j;
      if (j == J) { jobs[J].e = e; jobs[J].n = 0; ++J; }
      jobs[j].tok[jobs[j].n] = t; jobs[j].wt[jobs[j].n] = wp[t * topk + k]; jobs[j].n++;
    }
  std::memset(op, 0, sizeof(float) * T * H);
  if (J == 0) return;
  g_cold_jobs.fetch_add(J, std::memory_order_relaxed);
  if (tm) {
    int tail = 0;
    for (int j = 0; j < J; ++j)
      if (tm[jobs[j].e]) { prefetch_expert(tab + static_cast<int64_t>(jobs[j].e) * 6); ++tail; }
    g_tail_jobs.fetch_add(tail, std::memory_order_relaxed);
    std::stable_partition(jobs, jobs + J, [&](const Job& jb) { return !tm[jb.e]; });
  }
  const int nw = g_pool->size();
  const size_t per_job = MAXT * H + MAXT * (H / GS);
  const size_t per_worker = 3 * MAXT * I + MAXT * H + MAXT * (I / GS);
  TORCH_CHECK(per_job * J + per_worker * nw <= scratch_floats, "scratch too small");
  float* wbase = sp + per_job * J;
  for (int j = 0; j < J; ++j) {
    float* xs = sp + per_job * j;
    for (int i = 0; i < jobs[j].n; ++i) std::memcpy(xs + i * H, xp + jobs[j].tok[i] * H, sizeof(float) * H);
    group_sums(xs, jobs[j].n, H, xs + MAXT * H);
  }
  for (int w = 0; w < nw; ++w) std::memset(wbase + per_worker * w + 3 * MAXT * I, 0, sizeof(float) * MAXT * H);
  const int SL = g_nslices;
  g_pool->run(J * SL, [&](int item, int w) {
    const Job& jb = jobs[item / SL];
    const int c0 = g_slices[item % SL], c1 = g_slices[item % SL + 1], cw = c1 - c0;
    const float* xs = sp + per_job * (item / SL);
    const float* xg = xs + MAXT * H;
    float* wb = wbase + per_worker * w;
    float* gu = wb;
    float* act = wb + 2 * MAXT * I;
    float* yacc = wb + 3 * MAXT * I;
    float* ag = yacc + MAXT * H;
    const int64_t* e = tab + static_cast<int64_t>(jb.e) * 6;
    gemv_cols(reinterpret_cast<const int32_t*>(e[0]), reinterpret_cast<const uint16_t*>(e[1]), H, I, c0, c1, jb.n,
              xs, xg, gu, cw);
    gemv_cols(reinterpret_cast<const int32_t*>(e[2]), reinterpret_cast<const uint16_t*>(e[3]), H, I, c0, c1, jb.n,
              xs, xg, gu + MAXT * I, cw);
    const int ng = cw / GS;
    for (int t = 0; t < jb.n; ++t)
      for (int g = 0; g < ng; ++g) {
        float sum = 0.f;
        for (int n = g * GS; n < (g + 1) * GS; ++n) {
          const float gv = gu[t * cw + n], u = gu[MAXT * I + t * cw + n];
          const float a = gv / (1.0f + std::exp(-gv)) * u;
          act[t * cw + n] = a;
          sum += a;
        }
        ag[t * ng + g] = sum;
      }
    const int32_t* qd = reinterpret_cast<const int32_t*>(e[4]) + static_cast<size_t>(c0 / 8) * H;
    const uint16_t* sd = reinterpret_cast<const uint16_t*>(e[5]) + static_cast<size_t>(c0 / GS) * H;
    alignas(32) float y[MAXT * 320];
    for (int a0 = 0; a0 < H; a0 += 320) {
      switch (jb.n) {
        case 1: gemv_rows<1>(qd, sd, cw, H, a0, a0 + 320, act, ag, y, 320); break;
        case 2: gemv_rows<2>(qd, sd, cw, H, a0, a0 + 320, act, ag, y, 320); break;
        case 3: gemv_rows<3>(qd, sd, cw, H, a0, a0 + 320, act, ag, y, 320); break;
        default: gemv_rows<4>(qd, sd, cw, H, a0, a0 + 320, act, ag, y, 320); break;
      }
      for (int t = 0; t < jb.n; ++t) {
        float* o = yacc + static_cast<size_t>(jb.tok[t]) * H + a0;
        const __m256 wv = _mm256_set1_ps(jb.wt[t]);
        for (int n = 0; n < 320; n += 8)
          _mm256_storeu_ps(o + n, _mm256_fmadd_ps(wv, _mm256_loadu_ps(y + t * 320 + n), _mm256_loadu_ps(o + n)));
      }
    }
  });
  for (int w = 0; w < nw; ++w) {
    const float* yacc = wbase + per_worker * w + 3 * MAXT * I;
    for (int i = 0; i < T * H; i += 8)
      _mm256_storeu_ps(op + i, _mm256_add_ps(_mm256_loadu_ps(op + i), _mm256_loadu_ps(yacc + i)));
  }
}

// Prefill chunks (T up to MAXT_BIG): one work item per cold expert, largest first, each processing its tokens
// 4 at a time so the expert's 2.5 MB stay in L2/L3 between passes. Every (token, k) pair's unweighted output
// gets its own row of a pair buffer; the per-token reduction then runs in (t, k) order, so the result does not
// depend on thread scheduling.
constexpr int MAXT_BIG = 512;
size_t big_scratch_floats(int nw) {
  return static_cast<size_t>(MAXT_BIG) * (H / GS) + static_cast<size_t>(MAXT_BIG) * 10 * H +
         static_cast<size_t>(nw) * (4 * H + 4 * (H / GS) + 3 * 4 * I + 4 * (I / GS) + 4 * 320);
}

static void compute_layer_big(const int64_t* tab, int E, const uint8_t* sk, int T, int topk, const int32_t* idp,
                              const float* wp, const float* xp, float* op, float* sp, size_t scratch_floats,
                              const uint8_t* tm = nullptr) {
  const int P = T * topk;
  std::vector<int> cnt(E + 1, 0);
  for (int p = 0; p < P; ++p) {
    const int e = idp[p];
    if (e >= 0 && e < E && !sk[e]) cnt[e + 1]++;
  }
  std::vector<int> off(E + 1, 0);
  for (int e = 0; e < E; ++e) off[e + 1] = off[e] + cnt[e + 1];
  const int NP = off[E];
  std::vector<int> pair_of(P, -1), tok(NP > 0 ? NP : 1), fill(off.begin(), off.end() - 1);
  for (int p = 0; p < P; ++p) {
    const int e = idp[p];
    if (e < 0 || e >= E || sk[e]) continue;
    const int slot = fill[e]++;
    tok[slot] = p / topk;
    pair_of[p] = slot;
  }
  std::memset(op, 0, sizeof(float) * T * H);
  if (NP == 0) return;
  std::vector<int> ex;
  for (int e = 0; e < E; ++e)
    if (cnt[e + 1]) ex.push_back(e);
  std::stable_sort(ex.begin(), ex.end(), [&](int a, int b) { return cnt[a + 1] > cnt[b + 1]; });
  if (tm) {
    for (int e : ex)
      if (tm[e]) prefetch_expert(tab + static_cast<int64_t>(e) * 6);
    std::stable_partition(ex.begin(), ex.end(), [&](int e) { return !tm[e]; });
  }
  const int nw = g_pool->size();
  const size_t per_worker = 4 * H + 4 * (H / GS) + 3 * 4 * I + 4 * (I / GS) + 4 * 320;
  float* xg_all = sp;
  float* ypair = xg_all + static_cast<size_t>(T) * (H / GS);
  float* wbase = ypair + static_cast<size_t>(NP) * H;
  TORCH_CHECK(static_cast<size_t>(wbase - sp) + per_worker * nw <= scratch_floats, "big scratch too small");
  g_pool->run((T + 15) / 16, [&](int item, int) {
    const int t0 = item * 16, n = std::min(16, T - t0);
    group_sums(xp + static_cast<size_t>(t0) * H, n, H, xg_all + static_cast<size_t>(t0) * (H / GS));
  });
  g_pool->run(static_cast<int>(ex.size()), [&](int item, int w) {
    const int e = ex[item];
    const int base = off[e], n = cnt[e + 1];
    float* wb = wbase + per_worker * w;
    float* xs = wb;
    float* xg = xs + 4 * H;
    float* gu = xg + 4 * (H / GS);
    float* act = gu + 2 * 4 * I;
    float* ag = act + 4 * I;
    float* y = ag + 4 * (I / GS);
    const int64_t* ep = tab + static_cast<int64_t>(e) * 6;
    const int32_t* qd = reinterpret_cast<const int32_t*>(ep[4]);
    const uint16_t* sd = reinterpret_cast<const uint16_t*>(ep[5]);
    for (int c = 0; c < n; c += 4) {
      const int m = std::min(4, n - c);
      for (int i = 0; i < m; ++i) {
        const int t = tok[base + c + i];
        std::memcpy(xs + i * H, xp + static_cast<size_t>(t) * H, sizeof(float) * H);
        std::memcpy(xg + i * (H / GS), xg_all + static_cast<size_t>(t) * (H / GS), sizeof(float) * (H / GS));
      }
      gemv_cols(reinterpret_cast<const int32_t*>(ep[0]), reinterpret_cast<const uint16_t*>(ep[1]), H, I, 0, I, m,
                xs, xg, gu, I);
      gemv_cols(reinterpret_cast<const int32_t*>(ep[2]), reinterpret_cast<const uint16_t*>(ep[3]), H, I, 0, I, m,
                xs, xg, gu + 4 * I, I);
      for (int t = 0; t < m; ++t)
        for (int g = 0; g < I / GS; ++g) {
          float sum = 0.f;
          for (int k = g * GS; k < (g + 1) * GS; ++k) {
            const float gv = gu[t * I + k], u = gu[4 * I + t * I + k];
            const float a = gv / (1.0f + std::exp(-gv)) * u;
            act[t * I + k] = a;
            sum += a;
          }
          ag[t * (I / GS) + g] = sum;
        }
      for (int a0 = 0; a0 < H; a0 += 320) {
        switch (m) {
          case 1: gemv_rows<1>(qd, sd, I, H, a0, a0 + 320, act, ag, y, 320); break;
          case 2: gemv_rows<2>(qd, sd, I, H, a0, a0 + 320, act, ag, y, 320); break;
          case 3: gemv_rows<3>(qd, sd, I, H, a0, a0 + 320, act, ag, y, 320); break;
          default: gemv_rows<4>(qd, sd, I, H, a0, a0 + 320, act, ag, y, 320); break;
        }
        for (int i = 0; i < m; ++i)
          std::memcpy(ypair + static_cast<size_t>(base + c + i) * H + a0, y + i * 320, sizeof(float) * 320);
      }
    }
  });
  g_pool->run((T + 7) / 8, [&](int item, int) {
    for (int t = item * 8; t < std::min(T, item * 8 + 8); ++t) {
      float* o = op + static_cast<size_t>(t) * H;
      for (int k = 0; k < topk; ++k) {
        const int slot = pair_of[t * topk + k];
        if (slot < 0) continue;
        const __m256 wv = _mm256_set1_ps(wp[t * topk + k]);
        const float* yr = ypair + static_cast<size_t>(slot) * H;
        for (int nn = 0; nn < H; nn += 8)
          _mm256_storeu_ps(o + nn, _mm256_fmadd_ps(wv, _mm256_loadu_ps(yr + nn), _mm256_loadu_ps(o + nn)));
      }
    }
  });
}

// Big version of layer_forward for tests: T up to MAXT_BIG.
void layer_forward_big(torch::Tensor table, int64_t layer, torch::Tensor x, torch::Tensor ids, torch::Tensor w,
                       torch::Tensor skip, torch::Tensor out) {
  TORCH_CHECK(g_pool, "call init(threads) first");
  const int T = x.size(0), topk = ids.size(1), E = table.size(1);
  TORCH_CHECK(T >= 1 && T <= MAXT_BIG && topk <= 10, "1..512 tokens, topk <= 10");
  static std::vector<float> scratch;  // allocated (and zeroed) once, not per call
  if (scratch.size() < big_scratch_floats(g_pool->size())) scratch.resize(big_scratch_floats(g_pool->size()));
  compute_layer_big(table.data_ptr<int64_t>() + layer * E * 6, E, skip.data_ptr<uint8_t>(), T, topk,
                    ids.data_ptr<int32_t>(), w.data_ptr<float>(), x.data_ptr<float>(), out.data_ptr<float>(),
                    scratch.data(), scratch.size());
}

// table: int64 [L, E, 6]; x fp32 [T, H]; ids int32 [T, topk]; w fp32 [T, topk]; skip uint8 [E]; out fp32 [T, H]
void layer_forward(torch::Tensor table, int64_t layer, torch::Tensor x, torch::Tensor ids, torch::Tensor w,
                   torch::Tensor skip, torch::Tensor out, torch::Tensor scratch) {
  TORCH_CHECK(g_pool, "call init(threads) first");
  const int T = x.size(0), topk = ids.size(1), E = table.size(1);
  TORCH_CHECK(T >= 1 && T <= MAXT, "1..4 tokens");
  compute_layer(table.data_ptr<int64_t>() + layer * E * 6, E, skip.data_ptr<uint8_t>(), T, topk,
                ids.data_ptr<int32_t>(), w.data_ptr<float>(), x.data_ptr<float>(), out.data_ptr<float>(),
                scratch.data_ptr<float>(), scratch.numel());
}

// ---------------------------------------------------------------- dynamic arena (decode-driven LRU)
// Arena slots belong to fixed layers. Decode stamps the arena experts it uses (last_use = step) and collects the
// tail experts it had to read from the file mapping; after a step's last layer, up to g_dyn.per_step of those
// are copied into their layer's least-recently-used slot and the evicted expert becomes a tail expert. Capacity
// never changes. Promotions run only while no prefill step can be DMAing from the arena: stream_v2 brackets each
// prefill step with arena_begin_prefill()/arena_end_prefill(event) and the server checks that event first.
#ifdef WITH_BRIDGE
bool event_done(int64_t event);
#endif
struct DynArena {
  bool on = false;
  int L = 0, E = 0, last_layer = -1;
  int64_t* table = nullptr;
  uint8_t* tail = nullptr;
  int32_t* slot_of = nullptr;
  int32_t* owner = nullptr;
  int64_t* last_use = nullptr;
  const int64_t* file_addr = nullptr;
  int64_t base = 0, slot_bytes = 0;
  std::vector<std::vector<int>> layer_slots;
  int per_step = 8;
  std::vector<std::pair<int, int>> cand;
  int64_t step = -1, promoted = 0, skipped = 0;
  std::vector<torch::Tensor> keep;
};
DynArena g_dyn;
std::atomic<int> g_prefill_active{0}, g_promoting{0};
std::atomic<int64_t> g_dma_event{0};
// Prefill routing counts ([L, E] float32 in pinned host memory, decayed per prefill step, copied by the GPU at the
// end of each prefill step). After a prompt, the most-used tail experts are promoted first, with extra budget.
std::atomic<int64_t> g_usage_event{0};
const float* g_usage = nullptr;
std::vector<std::pair<int, int>> g_hints;
int g_hint_per_layer = 8, g_boost = 24;
void usage_ready(int64_t event, int64_t host_ptr, int64_t per_layer, int64_t boost) {
  g_usage = reinterpret_cast<const float*>(host_ptr);
  g_hint_per_layer = static_cast<int>(per_layer);
  g_boost = static_cast<int>(boost);
  g_usage_event.store(event, std::memory_order_seq_cst);
}

void dyn_init(torch::Tensor table, torch::Tensor tail, torch::Tensor slot_of, torch::Tensor owner,
              torch::Tensor slot_layer, torch::Tensor last_use, torch::Tensor file_addr, int64_t base,
              int64_t slot_bytes, int64_t per_step, int64_t last_layer, torch::Tensor protected_slots) {
  g_dyn = DynArena{};
  g_dyn.keep = {table, tail, slot_of, owner, slot_layer, last_use, file_addr, protected_slots};
  g_dyn.L = table.size(0);
  g_dyn.E = table.size(1);
  g_dyn.table = table.data_ptr<int64_t>();
  g_dyn.tail = tail.data_ptr<uint8_t>();
  g_dyn.slot_of = slot_of.data_ptr<int32_t>();
  g_dyn.owner = owner.data_ptr<int32_t>();
  g_dyn.last_use = last_use.data_ptr<int64_t>();
  g_dyn.file_addr = file_addr.data_ptr<int64_t>();
  g_dyn.base = base;
  g_dyn.slot_bytes = slot_bytes;
  g_dyn.per_step = static_cast<int>(per_step);
  g_dyn.last_layer = static_cast<int>(last_layer);
  g_dyn.layer_slots.assign(g_dyn.L, {});
  const int32_t* sl = slot_layer.data_ptr<int32_t>();
  const uint8_t* prot = protected_slots.data_ptr<uint8_t>();
  for (int64_t i = 0; i < slot_layer.numel(); ++i)
    if (!prot[i]) g_dyn.layer_slots[sl[i]].push_back(static_cast<int>(i));  // protected slots never move
  g_dyn.on = per_step > 0;
}

void arena_begin_prefill() {
  g_prefill_active.store(1, std::memory_order_seq_cst);
  while (g_promoting.load(std::memory_order_seq_cst)) std::this_thread::sleep_for(std::chrono::microseconds(20));
}
void arena_end_prefill(int64_t event) {
  g_dma_event.store(event, std::memory_order_seq_cst);
  g_prefill_active.store(0, std::memory_order_seq_cst);
}
std::vector<int64_t> dyn_stats() { return {g_dyn.promoted, g_dyn.skipped}; }

// Decode request of `layer` finished: stamp arena experts, remember tail experts.
static void dyn_note(int layer, int64_t step, int T, const int32_t* ids, const uint8_t* sk) {
  DynArena& d = g_dyn;
  d.step = step;
  for (int i = 0; i < T * 10; ++i) {
    const int e = ids[i];
    if (e < 0 || e >= d.E || sk[e]) continue;
    const int s = d.slot_of[layer * d.E + e];
    if (s >= 0) { d.last_use[s] = step; continue; }
    bool seen = false;
    for (auto& c : d.cand) if (c.first == layer && c.second == e) { seen = true; break; }
    if (!seen) d.cand.emplace_back(layer, e);
  }
}

static void dyn_take_hints() {
  // Called by the server between decode requests: turn completed prefill counts into promotion hints.
  DynArena& d = g_dyn;
  const int64_t ev = g_usage_event.load(std::memory_order_seq_cst);
  if (!ev || !g_usage) return;
#ifdef WITH_BRIDGE
  if (!event_done(ev)) return;
#endif
  g_usage_event.store(0, std::memory_order_seq_cst);
  g_hints.clear();
  std::vector<std::pair<float, int>> best;
  for (int l = 0; l < d.L; ++l) {
    best.clear();
    for (int e = 0; e < d.E; ++e) {
      const float c = g_usage[static_cast<int64_t>(l) * d.E + e];
      if (c > 0.f && d.tail[static_cast<int64_t>(l) * d.E + e]) best.emplace_back(c, e);
    }
    const int n = std::min<int>(g_hint_per_layer, static_cast<int>(best.size()));
    std::partial_sort(best.begin(), best.begin() + n, best.end(), [](auto& a, auto& b) { return a.first > b.first; });
    for (int i = 0; i < n; ++i) g_hints.emplace_back(l, best[i].second);
  }
}

static void dyn_promote() {
  DynArena& d = g_dyn;
  dyn_take_hints();
  if (d.cand.empty() && g_hints.empty()) return;
  g_promoting.store(1, std::memory_order_seq_cst);
  bool ok = !g_prefill_active.load(std::memory_order_seq_cst);
#ifdef WITH_BRIDGE
  const int64_t ev = g_dma_event.load(std::memory_order_seq_cst);
  if (ok && ev) ok = event_done(ev);
#endif
  if (!ok) {  // hints stay queued for a later step
    d.skipped += static_cast<int64_t>(d.cand.size());
    d.cand.clear();
    g_promoting.store(0, std::memory_order_seq_cst);
    return;
  }
  // Hints first (the last prompt's most-used tail experts, with their own budget), then this step's tail reads.
  size_t nh = 0;
  if (!g_hints.empty()) {
    nh = std::min<size_t>(g_hints.size(), static_cast<size_t>(g_boost));
    std::vector<std::pair<int, int>> merged(g_hints.begin(), g_hints.begin() + nh);
    g_hints.erase(g_hints.begin(), g_hints.begin() + nh);
    merged.insert(merged.end(), d.cand.begin(), d.cand.end());
    d.cand.swap(merged);
  }
  static const size_t sizes[6] = {819200, 25600, 819200, 25600, 819200, 25600};       // table (CPU) order
  static const size_t offs[6] = {0, 2457600, 819200, 2483200, 1638400, 2508800};      // offset in the slot
  struct Move { int l, e, slot, victim; };
  std::vector<Move> moves;
  const int budget = d.per_step + static_cast<int>(nh);
  for (auto& c : d.cand) {
    if (static_cast<int>(moves.size()) >= budget) break;
    const int l = c.first, e = c.second;
    if (d.slot_of[l * d.E + e] >= 0) continue;
    bool dup = false;  // hints and decode reads can name the same expert: promote it once
    for (const Move& m : moves) if (m.l == l && m.e == e) { dup = true; break; }
    if (dup) continue;
    int best = -1;
    int64_t best_use = INT64_MAX;
    for (int s : d.layer_slots[l])
      if (d.last_use[s] < best_use && d.last_use[s] < d.step) { best_use = d.last_use[s]; best = s; }
    if (best < 0) continue;
    d.last_use[best] = d.step;
    moves.push_back({l, e, best, d.owner[best]});
  }
  if (!moves.empty()) {
    g_pool->run(static_cast<int>(moves.size()) * 6, [&](int item, int) {
      const Move& m = moves[item / 6];
      const int i = item % 6;
      std::memcpy(reinterpret_cast<char*>(d.base + static_cast<int64_t>(m.slot) * d.slot_bytes) + offs[i],
                  reinterpret_cast<const char*>(d.file_addr[(static_cast<int64_t>(m.l) * d.E + m.e) * 6 + i]),
                  sizes[i]);
    });
    for (const Move& m : moves) {
      const int64_t ve = static_cast<int64_t>(m.l) * d.E + m.victim, ne = static_cast<int64_t>(m.l) * d.E + m.e;
      for (int i = 0; i < 6; ++i) {
        d.table[ve * 6 + i] = d.file_addr[ve * 6 + i];
        d.table[ne * 6 + i] = d.base + static_cast<int64_t>(m.slot) * d.slot_bytes + static_cast<int64_t>(offs[i]);
      }
      d.tail[ve] = 1;
      d.slot_of[ve] = -1;
      d.tail[ne] = 0;
      d.slot_of[ne] = m.slot;
      d.owner[m.slot] = m.e;
    }
  }
  d.promoted += static_cast<int64_t>(moves.size());
  d.skipped += static_cast<int64_t>(d.cand.size()) - static_cast<int64_t>(moves.size());
  d.cand.clear();
  g_promoting.store(0, std::memory_order_seq_cst);
}

// ---------------------------------------------------------------- GPU request server
// Request slot (bytes): seq int64 @0, T int32 @8, layer int32 @12, ids int32[40] @64, w f32[40] @224, x f32[4*H] @448.
// Response slot: seq int64 @0, y f32[4*H] @64. The GPU publishes req.seq last (st.release.sys) and waits for
// resp.seq == req.seq (ld.acquire.sys).
constexpr size_t REQ_BYTES = 41472, RESP_BYTES = 41088;
namespace {
std::thread g_server;
std::atomic<bool> g_server_stop{false};
std::atomic<int64_t> g_served{0};
std::atomic<int64_t> g_compute_ns{0};
// Decode-step breakdown (server thread only): per step, CPU compute over all layers, the gaps between one
// layer's response and the next layer's request (GPU work between MoE layers + handoff), and the gap from the
// last layer's response to the next step's first request (lm_head, drafts, sampling, scheduling).
struct StepStats {
  int64_t steps = 0, compute_ns = 0, layer_gap_ns = 0, step_gap_ns = 0, requests = 0;
};
StepStats g_stats;
std::atomic<int64_t> g_stats_every{0};
}

void set_stats_every(int64_t steps) { g_stats_every.store(steps); }

// Big slot (prefill chunks, eager only): seq @0, T @8, layer @12, ids int32[512*10] @64, w f32[512*10] @20544,
// x f32[512*H] @41024; response seq @0, y f32[512*H] @64. Same seq protocol, one slot shared by all layers.
constexpr size_t BIG_IDS = 64, BIG_W = 64 + 4 * MAXT_BIG * 10, BIG_X = 64 + 8 * MAXT_BIG * 10;
constexpr size_t REQ_BIG_BYTES = BIG_X + 4ull * MAXT_BIG * H, RESP_BIG_BYTES = 64 + 4ull * MAXT_BIG * H;
std::vector<int64_t> big_layout() {
  return {static_cast<int64_t>(REQ_BIG_BYTES), static_cast<int64_t>(RESP_BIG_BYTES), MAXT_BIG};
}

void start_server(torch::Tensor table, torch::Tensor skip, int64_t req_addr, int64_t resp_addr, int64_t nslots,
                  int64_t threads, std::vector<int64_t> cpus, int64_t req_big_addr, int64_t resp_big_addr) {
  TORCH_CHECK(!g_server.joinable(), "server already running");
  g_server_stop.store(false);
  const int E = table.size(1);
  const int64_t* tab = table.data_ptr<int64_t>();
  const uint8_t* sk = skip.data_ptr<uint8_t>();
  std::vector<int> c(cpus.begin(), cpus.end());
  g_server = std::thread([=] {
    delete g_pool;
    g_pool = new Pool(static_cast<int>(threads), c);   // server thread becomes worker 0
    std::vector<float> scratch(4 << 20);
    std::vector<float> scratch_big(req_big_addr ? big_scratch_floats(static_cast<int>(threads)) : 0);
    std::vector<int64_t> seen(nslots, 0);
    int64_t seen_big = 0;
    char* req = reinterpret_cast<char*>(req_addr);
    char* resp = reinterpret_cast<char*>(resp_addr);
    int idle = 0;
    using clk = std::chrono::steady_clock;
    clk::time_point last_done{};
    int last_layer = -1;
    StepStats window;
    while (!g_server_stop.load(std::memory_order_relaxed)) {
      bool any = false;
      for (int s = 0; s < nslots; ++s) {
        char* rq = req + REQ_BYTES * s;
        const int64_t q = __atomic_load_n(reinterpret_cast<int64_t*>(rq), __ATOMIC_ACQUIRE);
        if (q == seen[s]) continue;
        seen[s] = q;
        any = true;
        const int T = *reinterpret_cast<int32_t*>(rq + 8);
        const int layer = *reinterpret_cast<int32_t*>(rq + 12);
        char* rs = resp + RESP_BYTES * s;
        const auto tc0 = clk::now();
        if (last_layer >= 0) {
          const int64_t gap = std::chrono::duration_cast<std::chrono::nanoseconds>(tc0 - last_done).count();
          if (layer == last_layer + 1) window.layer_gap_ns += gap;
          else if (layer == 0 && gap < 1000000000LL) { window.step_gap_ns += gap; window.steps += 1; }
        }
        if (T >= 1 && T <= MAXT && layer >= 0) {
          compute_layer(tab + static_cast<int64_t>(layer) * E * 6, E, sk + static_cast<int64_t>(layer) * E, T, 10,
                        reinterpret_cast<int32_t*>(rq + 64), reinterpret_cast<float*>(rq + 224),
                        reinterpret_cast<float*>(rq + 448), reinterpret_cast<float*>(rs + 64), scratch.data(),
                        scratch.size(), g_tail_mask ? g_tail_mask + static_cast<int64_t>(layer) * E : nullptr);
        } else {
          std::memset(rs + 64, 0, sizeof(float) * MAXT * H);
        }
        __atomic_store_n(reinterpret_cast<int64_t*>(rs), q, __ATOMIC_RELEASE);
        last_done = clk::now();
        const int64_t cns = std::chrono::duration_cast<std::chrono::nanoseconds>(last_done - tc0).count();
        g_compute_ns.fetch_add(cns, std::memory_order_relaxed);
        g_served.fetch_add(1, std::memory_order_relaxed);
        window.compute_ns += cns;
        window.requests += 1;
        last_layer = layer;
        if (g_dyn.on && T >= 1 && T <= MAXT && layer >= 0) {
          dyn_note(layer, q / 64, T, reinterpret_cast<int32_t*>(rq + 64), sk + static_cast<int64_t>(layer) * E);
          if (layer == g_dyn.last_layer) dyn_promote();
        }
        const int64_t every = g_stats_every.load(std::memory_order_relaxed);
        if (every > 0 && window.steps >= every) {
          const double n = static_cast<double>(window.steps);
          const int64_t tj = g_tail_jobs.exchange(0), cj = g_cold_jobs.exchange(0);
          std::fprintf(stderr, "cpu_moe decode steps=%lld: per step cpu %.2f ms, between layers %.2f ms, "
                       "step tail %.2f ms (%.1f requests/step), cold experts %.0f/step of which tail %.1f; "
                       "promoted %lld, skipped %lld (total)\n",
                       static_cast<long long>(window.steps), window.compute_ns / n / 1e6,
                       window.layer_gap_ns / n / 1e6, window.step_gap_ns / n / 1e6, window.requests / n,
                       cj / n, tj / n, static_cast<long long>(g_dyn.promoted),
                       static_cast<long long>(g_dyn.skipped));
          window = StepStats{};
        }
      }
      if (req_big_addr) {
        char* rq = reinterpret_cast<char*>(req_big_addr);
        const int64_t q = __atomic_load_n(reinterpret_cast<int64_t*>(rq), __ATOMIC_ACQUIRE);
        if (q != seen_big) {
          seen_big = q;
          any = true;
          const int T = *reinterpret_cast<int32_t*>(rq + 8);
          const int layer = *reinterpret_cast<int32_t*>(rq + 12);
          char* rs = reinterpret_cast<char*>(resp_big_addr);
          if (T >= 1 && T <= MAXT_BIG && layer >= 0) {
            compute_layer_big(tab + static_cast<int64_t>(layer) * E * 6, E, sk + static_cast<int64_t>(layer) * E, T,
                              10, reinterpret_cast<int32_t*>(rq + BIG_IDS), reinterpret_cast<float*>(rq + BIG_W),
                              reinterpret_cast<float*>(rq + BIG_X), reinterpret_cast<float*>(rs + 64),
                              scratch_big.data(), scratch_big.size(),
                              g_tail_mask ? g_tail_mask + static_cast<int64_t>(layer) * E : nullptr);
          }
          __atomic_store_n(reinterpret_cast<int64_t*>(rs), q, __ATOMIC_RELEASE);
          last_layer = -1;  // prefill requests are not part of the decode-step breakdown
        }
      }
      if (any) { idle = 0; continue; }
      if (++idle < (1 << 16)) { _mm_pause(); continue; }
      std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
    delete g_pool;
    g_pool = nullptr;
  });
}

std::vector<int64_t> stop_server() {
  g_server_stop.store(true);
  if (g_server.joinable()) g_server.join();
  std::vector<int64_t> r{g_served.exchange(0), g_compute_ns.exchange(0)};
  return r;
}

#ifdef WITH_BRIDGE
void bridge_submit(torch::Tensor x, torch::Tensor ids, torch::Tensor w, int64_t req_addr, torch::Tensor counter,
                   int64_t layer, bool bump, int64_t mask_addr);
void bridge_wait_add(torch::Tensor out, int64_t resp_addr, torch::Tensor counter, int64_t layer,
                     int64_t extra_addr);
void bridge_submit_big(torch::Tensor x, torch::Tensor ids, torch::Tensor w, int64_t req_addr, torch::Tensor counter,
                       int64_t layer, bool bump);
void bridge_wait_add_f32(torch::Tensor acc, int64_t resp_addr, torch::Tensor counter, int64_t layer);
#endif

double bench_expert(torch::Tensor table, int64_t layer, int64_t expert, int64_t T, int64_t reps) {
  const int E = table.size(1);
  const int64_t* e = table.data_ptr<int64_t>() + (layer * E + expert) * 6;
  std::vector<float> x(MAXT * H, 0.01f), xg(MAXT * (H / GS)), g(MAXT * I), a(MAXT * I, 0.01f), ag(MAXT * (I / GS)), y(MAXT * H);
  group_sums(x.data(), T, H, xg.data());
  group_sums(a.data(), T, I, ag.data());
  auto once = [&] {
    gemv_cols(reinterpret_cast<const int32_t*>(e[0]), reinterpret_cast<const uint16_t*>(e[1]), H, I, 0, I, T, x.data(), xg.data(), g.data(), I);
    gemv_cols(reinterpret_cast<const int32_t*>(e[2]), reinterpret_cast<const uint16_t*>(e[3]), H, I, 0, I, T, x.data(), xg.data(), g.data(), I);
    gemv_cols(reinterpret_cast<const int32_t*>(e[4]), reinterpret_cast<const uint16_t*>(e[5]), I, H, 0, H, T, a.data(), ag.data(), y.data(), H);
  };
  once();
  const auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < reps; ++i) once();
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / reps;
}

// Prefill tail reads (stream_v2._TailReader): copy parts [fd, file offset, bytes, destination offset, drop] into
// memory at dst_base with `threads` threads, and drop the page cache of parts flagged for it. Bound without the
// GIL: the Python reader made ~900 syscalls per layer from 8 threads, each re-taking the GIL from the thread that
// launches the prefill kernels.
int64_t tail_read(int64_t dst_base, torch::Tensor parts, int64_t threads) {
  TORCH_CHECK(parts.scalar_type() == torch::kInt64 && parts.dim() == 2 && parts.size(1) == 5,
              "tail_read: parts must be int64 [n, 5]");
  const torch::Tensor p = parts.contiguous();
  const int64_t n = p.size(0);
  const int64_t* a = p.data_ptr<int64_t>();
  std::atomic<int64_t> next{0}, total{0};
  std::atomic<int> failed{0};
  auto work = [&]() {
    for (int64_t i = next.fetch_add(1, std::memory_order_relaxed); i < n;
         i = next.fetch_add(1, std::memory_order_relaxed)) {
      const int64_t* r = a + 5 * i;
      const int fd = static_cast<int>(r[0]);
      char* dst = reinterpret_cast<char*>(dst_base + r[3]);
      int64_t done = 0;
      while (done < r[2]) {
        const ssize_t got = pread(fd, dst + done, static_cast<size_t>(r[2] - done), static_cast<off_t>(r[1] + done));
        if (got < 0 && errno == EINTR) continue;
        if (got <= 0) { failed.store(1); break; }
        done += got;
      }
      if (r[4]) posix_fadvise(fd, static_cast<off_t>(r[1]), static_cast<off_t>(r[2]), POSIX_FADV_DONTNEED);
      total.fetch_add(done, std::memory_order_relaxed);
    }
  };
  const int64_t nt = std::max<int64_t>(1, std::min<int64_t>(threads, n));
  std::vector<std::thread> pool;
  pool.reserve(static_cast<size_t>(nt - 1));
  for (int64_t t = 1; t < nt; ++t) pool.emplace_back(work);
  work();
  for (auto& t : pool) t.join();
  TORCH_CHECK(!failed.load(), "tail_read: short read");
  return total.load();
}

void init(int64_t threads, std::vector<int64_t> cpus) {
  delete g_pool;
  std::vector<int> c(cpus.begin(), cpus.end());
  g_pool = new Pool(static_cast<int>(threads), c);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("init", &init);
  m.def("layer_forward", &layer_forward);
  m.def("bench_expert", &bench_expert);
  m.def("start_server", &start_server);
  m.def("set_slices", &set_slices);
  m.def("stop_server", &stop_server);
  m.def("set_stats_every", &set_stats_every);
  m.def("layer_forward_big", &layer_forward_big);
  m.def("set_tail_mask", &set_tail_mask);
  m.def("dyn_init", &dyn_init);
  m.def("arena_begin_prefill", &arena_begin_prefill);
  m.def("arena_end_prefill", &arena_end_prefill);
  m.def("dyn_stats", &dyn_stats);
  m.def("usage_ready", &usage_ready);
  m.def("big_layout", &big_layout);
  m.def("tail_read", &tail_read, py::call_guard<py::gil_scoped_release>());
#ifdef WITH_BRIDGE
  m.def("bridge_submit", &bridge_submit);
  m.def("bridge_wait_add", &bridge_wait_add);
  m.def("bridge_submit_big", &bridge_submit_big);
  m.def("bridge_wait_add_f32", &bridge_wait_add_f32);
#endif
}
