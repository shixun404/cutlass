#include <iostream>

#include "cutlass/cutlass.h"

#include "cute/tensor.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler_params.h"

#include "cutlass/util/command_line.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/gett.hpp"
#include "cutlass/util/reference/host/tensor_norm.h"
#include "cutlass/util/reference/host/tensor_compare.h"


#include <iostream>

#include "helper.h"

using namespace cute;

using nvfp4 = cutlass::nv_float4_t<cutlass::float_e2m1_t>;

/////////////////////////////////////////////////////////////////////////////////////////////////
/// GEMM kernel configurations
/////////////////////////////////////////////////////////////////////////////////////////////////

// A matrix configuration
using         ElementA    = cutlass::nv_float4_t<cutlass::float_e2m1_t>;    // Element type for A matrix operand
using         LayoutATag  = cutlass::layout::RowMajor;                      // Layout type for A matrix operand
constexpr int AlignmentA  = 32;                                             // Memory access granularity/alignment of A matrix in units of elements (up to 16 bytes)

// B matrix configuration
using         ElementB    = cutlass::nv_float4_t<cutlass::float_e2m1_t>;    // Element type for B matrix operand
using         LayoutBTag  = cutlass::layout::ColumnMajor;                   // Layout type for B matrix operand
constexpr int AlignmentB  = 32;                                             // Memory access granularity/alignment of B matrix in units of elements (up to 16 bytes)

// C/D matrix configuration
using         ElementD    = cutlass::float_e2m1_t;                          // Element type for D matrix operand
using         ElementSFD  = cutlass::float_ue8m0_t;                         // Element type for SFD matrix operand
using         ElementC    = cutlass::bfloat16_t;                            // Element type for C matrix operand
using         LayoutCTag  = cutlass::layout::RowMajor;                      // Layout type for C matrix operand
using         LayoutDTag  = cutlass::layout::RowMajor;                      // Layout type for D matrix operand
using         LayoutSFDTag = LayoutDTag;                                    // Layout type for SFD should be same as D matrix operand

constexpr int AlignmentD  = 128 / cutlass::sizeof_bits<ElementD>::value;    // Memory access granularity/alignment of C matrix in units of elements (up to 16 bytes)
constexpr int AlignmentC  = 128 / cutlass::sizeof_bits<ElementC>::value;    // Memory access granularity/alignment of C matrix in units of elements (up to 16 bytes)
// Kernel functional config
using ElementAccumulator  = float;                                          // Element type for internal accumulation
using ElementCompute      = float;                                          // Element type for internal accumulation
using ArchTag             = cutlass::arch::Sm120;                           // Tag indicating the minimum SM that supports the intended feature
using OperatorClass       = cutlass::arch::OpClassBlockScaledTensorOp;      // Operator class tag

// Kernel Perf config
using ThreadBlockShape    = Shape<_128,_128,_128>;                          // Threadblock's tile size
using ClusterShape        = Shape<_1,_1,_1>;                                // Shape of the threadblocks in a cluster

constexpr int InputSFVectorSize  = 16;
constexpr int OutputSFVectorSize = InputSFVectorSize;

// D = alpha * acc + beta * C
//      With BlockScaleFactor generation.
using FusionOperation = cutlass::epilogue::fusion::LinCombBlockScaleFactor<
    OutputSFVectorSize,
    ElementD,
    ElementCompute,
    ElementSFD, LayoutSFDTag,
    ElementC>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ThreadBlockShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutCTag, AlignmentC,
    ElementD, LayoutDTag, AlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAuto,                      // Epilogue schedule policy
    FusionOperation
  >::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutATag, AlignmentA,
    ElementB, LayoutBTag, AlignmentB,
    ElementAccumulator,
    ThreadBlockShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::KernelTmaWarpSpecializedPingpong                           // Ping-pong kernel schedule policy.
  >::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int,int,int,int>,                                                   // Indicates ProblemShape
    CollectiveMainloop,
    CollectiveEpilogue,
    void>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

// Reference device GEMM implementation type
using StrideA   = typename Gemm::GemmKernel::StrideA;
using LayoutA   = decltype(cute::make_layout(make_shape(0,0,0), StrideA{}));
using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFA;      // Scale Factor tensors have an interleaved layout. Bring Layout instead of stride.
using StrideB   = typename Gemm::GemmKernel::StrideB;
using LayoutB   = decltype(cute::make_layout(make_shape(0,0,0), StrideB{}));
using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFB;      // Scale Factor tensors have an interleaved layout. Bring Layout instead of stride.
using StrideC   = typename Gemm::GemmKernel::StrideC;
using LayoutC   = decltype(cute::make_layout(make_shape(0,0,0), StrideC{}));
using StrideD   = typename Gemm::GemmKernel::StrideD;
using LayoutD   = decltype(cute::make_layout(make_shape(0,0,0), StrideD{}));

using FusionOp = typename Gemm::EpilogueOutputOp;
constexpr bool IsBlockScaleSupported = FusionOp::IsBlockScaleSupported;
using SfdOutputCfg = cutlass::detail::Sm1xxBlockScaledOutputConfig<OutputSFVectorSize>;
using LayoutSFD = typename SfdOutputCfg::LayoutSF;

//
// Data members
//

/// Initialization
StrideA stride_A;
LayoutA layout_A;
LayoutSFA layout_SFA;
StrideB stride_B;
LayoutB layout_B;
LayoutSFB layout_SFB;
StrideC stride_C;
LayoutC layout_C;
StrideD stride_D;
LayoutD layout_D;
LayoutSFD layout_SFD;

uint64_t seed;

cutlass::HostTensor<nvfp4::DataType, cutlass::layout::PackedVectorLayout> block_A;
cutlass::HostTensor<nvfp4::DataType, cutlass::layout::PackedVectorLayout> block_B;
cutlass::HostTensor<nvfp4::DataType, cutlass::layout::PackedVectorLayout> block_D;
cutlass::HostTensor<nvfp4::DataType, cutlass::layout::PackedVectorLayout> block_reference_D;

cutlass::HostTensor<nvfp4::ScaleFactorType, cutlass::layout::PackedVectorLayout> block_SFA;
cutlass::HostTensor<nvfp4::ScaleFactorType, cutlass::layout::PackedVectorLayout> block_SFB;
cutlass::HostTensor<cutlass::float_ue8m0_t, cutlass::layout::PackedVectorLayout> block_SFD;
cutlass::HostTensor<cutlass::float_ue8m0_t, cutlass::layout::PackedVectorLayout> block_reference_SFD;

cutlass::HostTensor<ElementCompute, cutlass::layout::PackedVectorLayout> block_Normconst;

cutlass::HostTensor<cutlass::bfloat16_t, cutlass::layout::PackedVectorLayout> block_C;




/// Helper to initialize a block of device data
template <typename Element, typename Layout>
bool initialize_block(
  cutlass::TensorView<Element, Layout> view,
  uint64_t seed) {

  double scope_max, scope_min;
  constexpr int bits_input = cutlass::sizeof_bits<Element>::value;

  if constexpr (bits_input == 1) {
    scope_max = 2;
    scope_min = 0;
  }
  else if constexpr (bits_input <= 6) {
    scope_max = 2;
    scope_min = -2;
  }
  else if constexpr (bits_input <= 8) {
    if constexpr (cute::is_same_v<Element, cutlass::float_ue8m0_t>) {
      scope_max = 4;
      scope_min = 1;
    }
    else {
      scope_max = 1;
      scope_min = -1;
    }
  }
  else{
    scope_max = 4;
    scope_min = -4;
  }
  cutlass::reference::host::TensorFillRandomUniform(
    view, seed, scope_max, scope_min, 0);

  return true;
}

static inline float e2m1_to_float(uint8_t x) {
  uint8_t s = x >> 3;
  uint8_t e = (x >> 1) & 0x3;
  uint8_t m = x & 1;
  float sign = s ? -1.f : 1.f;
  if (e == 0) return sign * (m ? 0.25f : 0.0f);
  if (e == 3) return sign * INFINITY;
  return sign * (1.0f + 0.5f * m) * (1 << (e - 1));
}

static inline float ue4m3_to_float(uint8_t x) {
  uint8_t s = x >> 7;          // sign
  uint8_t e = (x >> 3) & 0xF;  // 4-bit exponent
  uint8_t m = x & 0x7;         // 3-bit mantissa

  float sign = s ? -1.f : 1.f;

  if (e == 0) {                // subnormal
      return sign * (m ? m * powf(2.0f, -10.0f) : 0.0f);
  }
  if (e == 0xF) {              // Inf / NaN
      return sign * INFINITY;
  }
  // normal numbers
  return sign * (1.0f + m / 8.0f) * powf(2.0f, (float)e - 7.0f);
}

void nvfp4_gemm_host_row_major(int M, int N, int K, uint8_t* hA, uint8_t* hB, uint8_t* hD, uint8_t* hSFA, uint8_t* hSFB, uint8_t* hSFD, cutlass::bfloat16_t* hC){
    float* accumulator = (float*)malloc(M * N * sizeof(float));
    for(int i = 0; i < M; i++){
        for(int j = 0; j < N; j++){
            accumulator[i * N + j] += hC[i * N + j];
            for(int k = 0; k < K; k++){
              uint8_t packedA = hA[(i * K + k) / 2];
              uint8_t nibA = ((i * K + k) % 2 == 0) ? (packedA & 0xF) : ((packedA >> 4) & 0xF);
              uint8_t packedB = hB[(k * N + j) / 2];
              uint8_t nibB = ((k * N + j) % 2 == 0) ? (packedB & 0xF) : ((packedB >> 4) & 0xF);

              // ---- convert to float ----
              float a = e2m1_to_float(nibA) / ue4m3_to_float(hSFA[i * K + k]);
              float b = e2m1_to_float(nibB) / ue4m3_to_float(hSFB[k * N + j]);

              accumulator[i * N + j] += a * b;
            }
        }
    }
    // for(int i = 0; i < M; i += SCALE_BLOCK_SIZE){
    //     for(int j = 0; j < N; j += SCALE_BLOCK_SIZE){
    //         float block_scale_factor = 0;
    //         for(int ii = i; ii < i + SCALE_BLOCK_SIZE; ii++){
    //             for(int jj = j; jj < j + SCALE_BLOCK_SIZE; jj++){
    //                 block_scale_factor = max(block_scale_factor, abs(accumulator[ii * N + jj]));
    //             }
    //         }
    //         for(int ii = i; ii < i + SCALE_BLOCK_SIZE; ii++){
    //             for(int jj = j; jj < j + SCALE_BLOCK_SIZE; jj++){
    //                 hSFD[ii * N + jj] = block_scale_factor;
    //             }
    //         }
    //     }
    // }
    // for(int i = 0; i < M; i++){
    //     for(int j = 0; j < N; j++){
    //         hD[i * N + j] = accumulator[i * N + j] / hSFD[i * N + j];
    //     }
    // }
    // free(accumulator);
}

template <typename T>
auto make_iterator(T* ptr) {
  return cute::recast_ptr<T>(ptr);
}


// Command line options parsing
struct Options {

  bool help;

  float alpha, beta;
  int iterations;
  int m, n, k;

  Options():
    help(false),
    m(1024), n(1024), k(1024),
    alpha(1.f), beta(0.f),
    iterations(10)
  { }

  // Parses the command line
  void parse(int argc, char const **args) {
    cutlass::CommandLine cmd(argc, args);

    if (cmd.check_cmd_line_flag("help")) {
      help = true;
      return;
    }

    cmd.get_cmd_line_argument("m", m);
    cmd.get_cmd_line_argument("n", n);
    cmd.get_cmd_line_argument("k", k);
    cmd.get_cmd_line_argument("alpha", alpha, 1.f);
    cmd.get_cmd_line_argument("beta", beta, 0.f);
    cmd.get_cmd_line_argument("iterations", iterations);
  }

  /// Prints the usage statement.
  std::ostream & print_usage(std::ostream &out) const {

    out << "79b_blackwell_geforce_nvfp4_nvfp4_gemm\n\n"
      << "  Blackwell NVFP4 GEMM using a Warp Specialized kernel.\n\n"
      << "Options:\n\n"
      << "  --help                      If specified, displays this usage statement\n\n"
      << "  --m=<int>                   Sets the M extent of the GEMM\n"
      << "  --n=<int>                   Sets the N extent of the GEMM\n"
      << "  --k=<int>                   Sets the K extent of the GEMM\n"
      << "  --alpha=<f32>               Epilogue scalar alpha\n"
      << "  --beta=<f32>                Epilogue scalar beta\n\n"
      << "  --iterations=<int>          Number of profiling iterations to perform.\n\n";

    out << "\n\nExamples:\n\n"
      << "$ " << "./examples/79_blackwell_geforce_gemm/79b_blackwell_geforce_nvfp4_nvfp4_gemm" << " --m=1024 --n=512 --k=1024 --alpha=2 --beta=0.707 \n\n";

    return out;
  }

  /// Compute performance in GFLOP/s
  double gflops(double runtime_s) const
  {
    // Two flops per multiply-add
    uint64_t flop = uint64_t(2) * m * n * k;
    double gflop = double(flop) / double(1.0e9);
    return gflop / runtime_s;
  }
};

bool verify(const Options &options) {
  using namespace cute;
  // Create the arguments for host reference implementation
  Tensor tensor_A = make_tensor(make_iterator(block_A.host_data()), layout_A);
  Tensor tensor_SFA = make_tensor(block_SFA.host_data(), layout_SFA);
  Tensor tensor_B = make_tensor(make_iterator(block_B.host_data()), layout_B);
  Tensor tensor_SFB = make_tensor(block_SFB.host_data(), layout_SFB);

  cutlass::reference::host::GettBlockScalingMainloopParams<
      ElementAccumulator,                 // ElementAccumulator
      decltype(tensor_A),                 // TensorA
      decltype(tensor_SFA),               // TensorSfA
      decltype(tensor_B),                 // TensorB
      decltype(tensor_SFB)                // TensorSfB
    > mainloop_params{tensor_A, tensor_SFA, tensor_B, tensor_SFB};

  auto tensor_C = cute::make_tensor(make_iterator(block_C.host_data()), layout_C);
  auto tensor_D = cute::make_tensor(make_iterator(block_reference_D.host_data()), layout_D);
  auto tensor_SFD = make_tensor(block_reference_SFD.host_data(), layout_SFD);

  cutlass::reference::host::GettBlockScalingEpilogueParams<
      ElementAccumulator,                   // ElementScalar
      ElementAccumulator,                   // ElementAccumulator
      ElementAccumulator,                   // ElementCompute
      decltype(tensor_C),                   // TensorC
      decltype(tensor_D),                   // TensorD
      decltype(tensor_SFD),                 // TensorSfD
      cute::Int<OutputSFVectorSize>,
      cutlass::reference::host::SfStrategy::SfDGen
    > epilogue_params{options.alpha, options.beta, tensor_C, tensor_D, tensor_SFD, block_Normconst.at(cutlass::make_Coord(0))};

  cutlass::reference::host::Gemm3x(mainloop_params, epilogue_params);

  // Comparison
  block_D.sync_host();
  bool passed = cutlass::reference::host::TensorEquals(block_reference_D.host_view(), block_D.host_view());
  passed &= (cutlass::reference::host::TensorNorm(block_reference_D.host_view()) > 0);
  passed &= (cutlass::reference::host::TensorNorm(block_D.host_view()) > 0);

  return passed;
}







int main(int argc, const char **args){
  
  Options options;

  options.parse(argc, args);

  if (options.help) {
    options.print_usage(std::cout) << std::endl;
    return 0;
  }

  int m = options.m;
  int n = options.n;
  int k = options.k;

  printf("m: %d, n: %d, k: %d\n", m, n, k);

    using namespace cute;
    // For SFA and SFB tensors layouts
    using Sm1xxBlkScaledConfig =  typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
    // For SFD tensor layout
    using Sm1xxBlockScaledOutputConfig=  typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

    stride_A = cutlass::make_cute_packed_stride(StrideA{}, {m, k, 1});
    stride_B = cutlass::make_cute_packed_stride(StrideB{}, {n, k, 1});
    stride_C = cutlass::make_cute_packed_stride(StrideC{}, {m, n, 1});
    stride_D = cutlass::make_cute_packed_stride(StrideD{}, {m, n, 1});

    layout_A = make_layout(make_shape(m, k, 1), stride_A);
    layout_B = make_layout(make_shape(n, k, 1), stride_B);
    layout_C = make_layout(make_shape(m, n, 1), stride_C);
    layout_D = make_layout(make_shape(m, n, 1), stride_D);
    layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(m, n, k, 1));
    layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(m, n, k, 1));
    layout_SFD = SfdOutputCfg::tile_atom_to_shape_SFD(cute::make_shape(m, n, k, 1));

    printf("Layout A: ");
    print(layout_A);
    printf("\n");
    
    printf("Layout B: ");
    print(layout_B);
    printf("\n");

    printf("Layout C: ");
    print(layout_C);
    printf("\n");
    
    printf("Layout D: ");
    print(layout_D);
    printf("\n");
    
    printf("Layout SFA: ");
    print(layout_SFA);
    printf("\n");
    
    printf("Layout SFB: ");
    print(layout_SFB);
    printf("\n");
    
    printf("Layout SFD: ");
    print(layout_SFD);
    printf("\n");

    // 在第298行之后添加以下代码：

printf("\n=== Layout Size Analysis ===\n");

// 1. Layout A的size分析
printf("--- Layout A Analysis ---\n");
printf("Layout A shape: ");
print(layout_A.shape());
printf("\n");
printf("Layout A stride: ");
print(layout_A.stride());
printf("\n");
printf("Layout A total elements: %lld\n", (long long)size(layout_A));
printf("Layout A memory size: %lld bytes\n", (long long)size(layout_A) * sizeof(nvfp4::DataType));

// 2. Layout SFA的size分析
printf("\n--- Layout SFA Analysis ---\n");
printf("Layout SFA shape: ");
print(layout_SFA.shape());
printf("\n");
printf("Layout SFA stride: ");
print(layout_SFA.stride());
printf("\n");
printf("Layout SFA total elements: %lld\n", (long long)size(layout_SFA));
printf("Layout SFA filtered elements: %lld\n", (long long)size(filter_zeros(layout_SFA)));
printf("Layout SFA memory size: %lld bytes\n", (long long)size(filter_zeros(layout_SFA)) * sizeof(nvfp4::ScaleFactorType));

printf("==========================================\n\n");


    block_A.reset(cutlass::make_Coord(size(layout_A)));
    block_B.reset(cutlass::make_Coord(size(layout_B)));
    block_C.reset(cutlass::make_Coord(size(layout_C)));
    block_D.reset(cutlass::make_Coord(size(layout_D)));
    block_reference_D.reset(cutlass::make_Coord(size(layout_D)));
    block_reference_SFD.reset(cutlass::make_Coord(size(filter_zeros(layout_SFD))));
    block_Normconst.reset(cutlass::make_Coord(1));

    block_SFA.reset(cutlass::make_Coord(size(filter_zeros(layout_SFA))));
    block_SFB.reset(cutlass::make_Coord(size(filter_zeros(layout_SFB))));
    block_SFD.reset(cutlass::make_Coord(size(filter_zeros(layout_SFD))));

    printf("block_A size: %zu\n", block_A.size());
    printf("block_B size: %zu\n", block_B.size());
    printf("block_C size: %zu\n", block_C.size());
    printf("block_D size: %zu\n", block_D.size());
    printf("block_D_reference size: %zu\n", block_reference_D.size());
    printf("block_SFD_reference size: %zu\n", block_reference_SFD.size());
    printf("block_SFA size: %zu\n", block_SFA.size());
    printf("block_SFB size: %zu\n", block_SFB.size());
    
    // 初始化 host 端数据
    int seed = 2025;
    initialize_block(block_A.host_view(), seed + 2021);
    initialize_block(block_B.host_view(), seed + 2022);
    initialize_block(block_C.host_view(), seed + 2023);
    initialize_block(block_SFA.host_view(), seed + 2024);
    initialize_block(block_SFB.host_view(), seed + 2025);
    block_Normconst.at(cutlass::make_Coord(0)) = 2;
    
    // // 将数据复制到 device
    // block_A.sync_device();
    // block_B.sync_device();
    // block_C.sync_device();
    // block_SFA.sync_device();
    // block_SFB.sync_device();
    // block_SFD.sync_device();
    // block_Normconst.sync_device();
    
    // // 获取 device 指针
    // nvfp4::DataType* dA = block_A.device_data();
    // nvfp4::DataType* dB = block_B.device_data();
    // nvfp4::DataType* dD = block_D.device_data();
    
    // nvfp4::ScaleFactorType* dSFA = block_SFA.device_data();
    // nvfp4::ScaleFactorType* dSFB = block_SFB.device_data();
    // cutlass::float_ue8m0_t* dSFD = block_SFD.device_data();
    
    // cutlass::bfloat16_t* dC = block_C.device_data();
    
    return 0;
    
}




