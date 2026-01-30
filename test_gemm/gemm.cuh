
#include <iostream>

#include "cutlass/cutlass.h"

#include "cute/tensor.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
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
#include "cutlass/util/reference/device/tensor_fill.h"

#include "helper.h"

using namespace cute;

using RasterOrderOptions = typename cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90Params::RasterOrderOptions;

using ElementA = cutlass::bfloat16_t;
using ElementB = cutlass::bfloat16_t;
using ElementC = void;                 
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

// BF16 + Cooperative schedule
using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecializedCooperative;



// Non-fp8/non-blockscale: ElementCompute = ElementD
using ElementCompute = ElementD;

// Cooperative epilogue schedule
using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecializedCooperative;

using EpilogueFusion =
  cutlass::epilogue::fusion::LinearCombination<
    ElementD, ElementAccumulator, ElementD, ElementAccumulator
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
    ElementD, LayoutC, AlignmentC,
    ElementD, LayoutD, AlignmentD,
    EpilogueSchedule,
    EpilogueFusion
  >::CollectiveOp;

using StageCount = cutlass::gemm::collective::StageCountAutoCarveout<
      static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>;

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

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;


using DeviceGemmReference = cutlass::reference::device::Gemm<
  ElementA,
  LayoutA,
  ElementB,
  LayoutB,
  ElementD,
  LayoutD,
  ElementAccumulator,
  ElementAccumulator>;


// // A matrix configuration
// using         ElementA    = float;                                          // Element type for A matrix operand
// using         LayoutA     = cutlass::layout::RowMajor;                      // Layout type for A matrix operand
// constexpr int AlignmentA  = 128 / cutlass::sizeof_bits<ElementA>::value;    // Memory access granularity/alignment of A matrix in units of elements (up to 16 bytes)

// // B matrix configuration
// using         ElementB    = float;                                          // Element type for B matrix operand
// using         LayoutB     = cutlass::layout::ColumnMajor;                   // Layout type for B matrix operand
// constexpr int AlignmentB  = 128 / cutlass::sizeof_bits<ElementB>::value;    // Memory access granularity/alignment of B matrix in units of elements (up to 16 bytes)

// // C/D matrix configuration
// using         ElementC    = float;                                          // Element type for C and D matrix operands
// using         LayoutC     = cutlass::layout::ColumnMajor;                   // Layout type for C and D matrix operands
// constexpr int AlignmentC  = 128 / cutlass::sizeof_bits<ElementC>::value;    // Memory access granularity/alignment of C matrix in units of elements (up to 16 bytes)

// // Core kernel configurations
// using ElementAccumulator  = float;                                          // Element type for internal accumulation
// using ArchTag             = cutlass::arch::Sm90;                            // Tag indicating the minimum SM that supports the intended feature
// using OperatorClass       = cutlass::arch::OpClassTensorOp;                 // Operator class tag
// using TileShape           = Shape<_128,_128,_32>;                           // Threadblock-level tile size
// using ClusterShape        = Shape<_4,_2,_1>;                                // Shape of the threadblocks in a cluster
// using StageCountType = cutlass::gemm::collective::StageCountAuto;           // Stage count maximized based on the tile size
// using KernelSchedule = cutlass::gemm::collective::KernelScheduleAuto;       // Kernel to launch based on the default setting in the Collective Builder

// using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
//     cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
//     TileShape, ClusterShape,
//     cutlass::epilogue::collective::EpilogueTileAuto,
//     ElementAccumulator, ElementAccumulator,
//     ElementC, LayoutC, AlignmentC,
//     ElementC, LayoutC, AlignmentC,
//     cutlass::epilogue::collective::EpilogueScheduleAuto
//   >::CollectiveOp;

// using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
//     ArchTag, OperatorClass,
//     ElementA, LayoutA, AlignmentA,
//     ElementB, LayoutB, AlignmentB,
//     ElementAccumulator,
//     TileShape, ClusterShape,
//     cutlass::gemm::collective::StageCountAutoCarveout<
//       static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
//     cutlass::gemm::collective::KernelScheduleAuto
//   >::CollectiveOp;

// using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
//     Shape<int,int,int>, // Indicates ProblemShape
//     CollectiveMainloop,
//     CollectiveEpilogue
// >;

// using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

// // Reference device GEMM implementation type
// using DeviceGemmReference = cutlass::reference::device::Gemm<
//   ElementA,
//   LayoutA,
//   ElementB,
//   LayoutB,
//   ElementC,
//   LayoutC,
//   ElementAccumulator,
//   ElementAccumulator>;

using StrideA = typename Gemm::GemmKernel::StrideA;
using StrideB = typename Gemm::GemmKernel::StrideB;
using StrideC = typename Gemm::GemmKernel::StrideC;
using StrideD = typename Gemm::GemmKernel::StrideD;

cutlass::DeviceAllocation<typename Gemm::ElementA> block_A;
cutlass::DeviceAllocation<typename Gemm::ElementB> block_B;
cutlass::DeviceAllocation<typename Gemm::ElementC> block_C;
cutlass::DeviceAllocation<typename Gemm::EpilogueOutputOp::ElementOutput> block_D;
cutlass::DeviceAllocation<typename Gemm::EpilogueOutputOp::ElementOutput> block_ref_D;

StrideA stride_A;
StrideB stride_B;
StrideC stride_C;
StrideD stride_D;
