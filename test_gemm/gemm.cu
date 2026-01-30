// run_sm90_bf16_tma_coop_gemm.cu
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#include "cutlass/cutlass.h"
#include "cutlass/bfloat16.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/util/packed_stride.hpp"

#include "cutlass/gemm/kernel/gemm_universal_decl.h"
#include "cutlass/gemm/kernel/tile_scheduler.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"

#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cute/numeric/integral_constant.hpp"
// #include "cutlass/epilogue/fusion/linear_combination.hpp"

#include <cute/layout.hpp>
#include <cute/numeric/integral_constant.hpp>

#define CHECK_CUDA(call) do {                                  \
  cudaError_t err = call;                                      \
  if (err != cudaSuccess) {                                    \
    fprintf(stderr, "CUDA error %s:%d: %s\n",                  \
            __FILE__, __LINE__, cudaGetErrorString(err));      \
    std::exit(1);                                              \
  }                                                            \
} while(0)

static inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

using ElementA = cutlass::bfloat16_t;
using ElementB = cutlass::bfloat16_t;
using ElementC = void;                 // align Flux
using ElementD = cutlass::bfloat16_t;
using ElementAccumulator = float;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;
using LayoutD = cutlass::layout::RowMajor;

// 128-bit alignment in elements (bf16 -> 16 bytes / 2 bytes = 8 elems)
static constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value; // 8
static constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value; // 8
static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementD>::value; // ElementC=void -> use D
static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value; // 8

using TileShape    = cute::Shape<cute::Int<128>, cute::Int<256>, cute::Int<64>>;
using ClusterShape = cute::Shape<cute::Int<2>,   cute::Int<1>,   cute::Int<1>>;

// Flux mainloop_stage = 0 -> StageCountAutoCarveout<epi_smem>
using StageCount = cutlass::gemm::collective::StageCountAutoCarveout<0>;

// BF16 + Cooperative schedule
using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecializedCooperative;

using CollectiveMma =
  typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90,
    cutlass::arch::OpClassTensorOp,
    ElementA, LayoutA, AlignmentA,
    ElementB, LayoutB, AlignmentB,
    ElementAccumulator,
    TileShape,
    ClusterShape,
    StageCount,
    KernelSchedule
  >::CollectiveOp;

// Non-fp8/non-blockscale: ElementCompute = ElementD
using ElementCompute = ElementD;

// Cooperative epilogue schedule
using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecializedCooperative;

using EpilogueFusion =
  cutlass::epilogue::fusion::LinearCombination<
    ElementD, ElementAccumulator, ElementC, ElementAccumulator
  >;

using CollectiveEpilogue =
  typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90,
    cutlass::arch::OpClassTensorOp,
    TileShape,
    ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator,
    ElementCompute,
    ElementC, LayoutC, AlignmentC,
    ElementD, LayoutD, AlignmentD,
    EpilogueSchedule,
    EpilogueFusion
  >::CollectiveOp;

using Scheduler = cutlass::gemm::PersistentScheduler;

// ProblemShape: (m,n,k)
using ProblemShape = cute::tuple<int,int,int>;

using GemmKernel =
  cutlass::gemm::kernel::GemmUniversal<
    ProblemShape,
    CollectiveMma,
    CollectiveEpilogue,
    Scheduler
  >;

using GemmDevice = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

int main(int argc, char** argv) {
  int iters = 10;
  int warmup = 3;
  if (argc >= 2) iters = std::atoi(argv[1]);
  if (argc >= 3) warmup = std::atoi(argv[2]);

  // Problem
  int m = 32768, n = 32768, k = 4096;

  // quick device check
  int dev = 0;
  CHECK_CUDA(cudaGetDevice(&dev));
  cudaDeviceProp prop{};
  CHECK_CUDA(cudaGetDeviceProperties(&prop, dev));
  if (prop.major < 9) {
    fprintf(stderr, "This example requires SM90+ (Hopper). Current: sm_%d%d\n", prop.major, prop.minor);
    return 1;
  }

  // Allocate A[M,K], B[K,N] logically.
  // BUT LayoutB is ColumnMajor for (K,N), i.e. leading dimension = K.
  size_t bytesA = size_t(m) * size_t(k) * sizeof(ElementA);
  size_t bytesB = size_t(k) * size_t(n) * sizeof(ElementB);
  size_t bytesD = size_t(m) * size_t(n) * sizeof(ElementD);

  ElementA* A = nullptr;
  ElementB* B = nullptr;
  ElementD* D = nullptr;
  CHECK_CUDA(cudaMalloc(&A, bytesA));
  CHECK_CUDA(cudaMalloc(&B, bytesB));
  CHECK_CUDA(cudaMalloc(&D, bytesD));

  // (optional) initialize with something (here: just memset for simplicity)
  CHECK_CUDA(cudaMemset(A, 0, bytesA));
  CHECK_CUDA(cudaMemset(B, 0, bytesB));
  CHECK_CUDA(cudaMemset(D, 0, bytesD));

  // Packed strides (Flux style)
  // A: RowMajor [m,k] => stride (lda = k)
  auto strideA = cutlass::make_cute_packed_stride(cute::Stride<cute::_2, cute::_1>{}, cute::make_shape(m, k));
  // B: ColumnMajor [k,n] => in CUTLASS layout terms, B is [k,n] with stride (ldb = k) in column-major
  auto strideB = cutlass::make_cute_packed_stride(cute::Stride<cute::_1, cute::_2>{}, cute::make_shape(k, n));
  // C is void, but still pass a pointer/stride placeholder if API requires it
  auto strideC = cutlass::make_cute_packed_stride(cute::Stride<cute::_2, cute::_1>{}, cute::make_shape(m, n));
  auto strideD = cutlass::make_cute_packed_stride(cute::Stride<cute::_2, cute::_1>{}, cute::make_shape(m, n));

  // Scheduler swizzle size (Flux)
  int swizzle_m = ceil_div(m, 128);
  int swizzle_n = ceil_div(n, 256);
  int max_swizzle_size = (swizzle_m < swizzle_n) ? swizzle_m : swizzle_n;

  typename GemmDevice::Arguments args{
    cutlass::gemm::GemmUniversalMode::kGemm,
    ProblemShape{m, n, k},
    // Mainloop arguments
    typename GemmKernel::CollectiveMainloop::Arguments{
      A, strideA,
      B, strideB
    },
    // Epilogue arguments
    typename GemmKernel::CollectiveEpilogue::Arguments{
      typename EpilogueFusion::Params{
        ElementAccumulator(1.0f),  // alpha
        ElementAccumulator(0.0f)   // beta
      },
      /* C_ptr */ nullptr, strideC,
      /* D_ptr */ D,       strideD
    },
    // Scheduler arguments
    typename GemmKernel::TileScheduler::Arguments{
      /* max_swizzle_size = */ max_swizzle_size
    }
  };

  // Check implementability
  auto status = GemmDevice::can_implement(args);
  if (status != cutlass::Status::kSuccess) {
    fprintf(stderr, "GemmDevice::can_implement failed: %d\n", int(status));
    return 1;
  }

  // Workspace
  size_t workspace_bytes = GemmDevice::get_workspace_size(args);
  void* workspace = nullptr;
  if (workspace_bytes) CHECK_CUDA(cudaMalloc(&workspace, workspace_bytes));

  GemmDevice gemm;
  status = gemm.initialize(args, workspace);
  if (status != cutlass::Status::kSuccess) {
    fprintf(stderr, "gemm.initialize failed: %d\n", int(status));
    return 1;
  }

  // Warmup
  for (int i = 0; i < warmup; ++i) {
    status = gemm.run();
    if (status != cutlass::Status::kSuccess) {
      fprintf(stderr, "gemm.run warmup failed: %d\n", int(status));
      return 1;
    }
  }
  CHECK_CUDA(cudaDeviceSynchronize());

  // Timing
  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  CHECK_CUDA(cudaEventRecord(start));
  for (int i = 0; i < iters; ++i) {
    status = gemm.run();
    if (status != cutlass::Status::kSuccess) {
      fprintf(stderr, "gemm.run failed: %d\n", int(status));
      return 1;
    }
  }
  CHECK_CUDA(cudaEventRecord(stop));
  CHECK_CUDA(cudaEventSynchronize(stop));

  float ms = 0.0f;
  CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
  float avg_ms = ms / float(iters);

  double tflops = (2.0 * double(m) * double(n) * double(k)) / (avg_ms * 1e-3) / 1e12;
  printf("M=N=%d K=%d, avg %.4f ms, %.2f TFLOPS\n", m, k, avg_ms, tflops);

  // Cleanup
  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));
  if (workspace) CHECK_CUDA(cudaFree(workspace));
  CHECK_CUDA(cudaFree(A));
  CHECK_CUDA(cudaFree(B));
  CHECK_CUDA(cudaFree(D));
  return 0;
}
