# SPDX-FileCopyrightText: Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""
2SM Dense GEMM example using cute_ext decorators.

Supports batched / group-GEMM via the L dimension in --mnkl (M,N,K,L).
For L > 1, the kernel is invoked once per batch element sequentially.

Tile constraints (hardware):
  2-CTA mode: M must be divisible by 256, N by 256, K by 64
  1-CTA mode: M must be divisible by 128, N by 256, K by 64
"""

import argparse
import torch
import math
import cutlass
from cutlass import cute
from cutlass.cute import experimental as cute_ext
from cutlass.cute.runtime import from_dlpack
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils as utils
from cutlass.base_dsl.typing import Numeric
from typing import Tuple, Type


def create_gemm_tensors_torch(
    M,
    N,
    K,
    majors: tuple[
        cute.nvgpu.tcgen05.OperandMajorMode,
        cute.nvgpu.tcgen05.OperandMajorMode,
        cute.nvgpu.tcgen05.OperandMajorMode,
    ],
    dtypes: tuple[torch.dtype, torch.dtype, torch.dtype],
):
    A = None
    B = None
    D = None

    if majors[0] == cute.nvgpu.tcgen05.OperandMajorMode.MN:
        A = torch.empty(K, M).random_(-4, 4).permute(1, 0).to(dtypes[0]).cuda()
    elif majors[0] == cute.nvgpu.tcgen05.OperandMajorMode.K:
        A = torch.empty(M, K).random_(-4, 4).permute(0, 1).to(dtypes[0]).cuda()
    if majors[1] == cute.nvgpu.tcgen05.OperandMajorMode.MN:
        B = torch.empty(K, N).random_(-4, 4).permute(1, 0).to(dtypes[1]).cuda()
    elif majors[1] == cute.nvgpu.tcgen05.OperandMajorMode.K:
        B = torch.empty(N, K).random_(-4, 4).permute(0, 1).to(dtypes[1]).cuda()
    if majors[2] == cute.nvgpu.tcgen05.OperandMajorMode.MN:
        D = torch.empty(N, M).random_(-4, 4).permute(1, 0).to(dtypes[2]).cuda()
    elif majors[2] == cute.nvgpu.tcgen05.OperandMajorMode.K:
        D = torch.empty(M, N).random_(-4, 4).permute(0, 1).to(dtypes[2]).cuda()

    return A, B, D


def get_gemm_tensors(
    M,
    N,
    K,
    majors: tuple[
        cute.nvgpu.tcgen05.OperandMajorMode,
        cute.nvgpu.tcgen05.OperandMajorMode,
        cute.nvgpu.tcgen05.OperandMajorMode,
    ],
    dtypes: tuple[torch.dtype, torch.dtype, torch.dtype],
):
    A, B, D = create_gemm_tensors_torch(M, N, K, majors, dtypes)

    A_cute = from_dlpack(A, assumed_align=16).mark_layout_dynamic(
        leading_dim=1 if majors[0] == cute.nvgpu.tcgen05.OperandMajorMode.K else 0
    )
    B_cute = from_dlpack(B, assumed_align=16).mark_layout_dynamic(
        leading_dim=1 if majors[1] == cute.nvgpu.tcgen05.OperandMajorMode.K else 0
    )
    D_cute = from_dlpack(D, assumed_align=16).mark_layout_dynamic(
        leading_dim=1 if majors[2] == cute.nvgpu.tcgen05.OperandMajorMode.K else 0
    )

    return A, B, D, A_cute, B_cute, D_cute


def sm100_4x4x1_kernel_builder(
    use_tma_multicast: bool,
    use_2cta_instrs: bool,
    acc_dtype: Type[Numeric],
    M: int,
    N: int,
):
    CLUSTER_SHAPE = (2, 1, 1)
    GRID_SHAPE = (
        math.ceil(M / 128),
        math.ceil(N / 256),
        1,
    )  # TODO (xpbowler): remove hard-code
    NUM_WARPS_PER_CTA = 6
    TMA_STORE_PIPE_DEPTH = 4
    MAINLOOP_STAGE_DEPTH = 4  # pipeline depth of TMA->MMA
    # pipeline depth of mainloop->epilogue. only useful if using persistent CTA
    EPILOGUE_STAGE_DEPTH = 1

    # m256n256k16 2SM MMA / m128n256k16 1SM MMA
    mma_inst_shape_mnk = (256, 256, 16) if use_2cta_instrs else (128, 256, 16)

    @cute_ext.kernel
    def kernel(
        mA: cute.Tensor,
        mB: cute.Tensor,
        mD: cute.Tensor,
    ):
        d_layout = utils.LayoutEnum.from_tensor(mD)
        d_dtype = mD.element_type
        ab_dtype = mA.element_type

        mma_inst_shape_m, mma_inst_shape_n, mma_inst_shape_k = mma_inst_shape_mnk
        if cutlass.const_expr(use_2cta_instrs):
            cta_group = cute.nvgpu.tcgen05.CtaGroup.TWO
        else:
            cta_group = cute.nvgpu.tcgen05.CtaGroup.ONE

        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            ab_dtype,
            utils.LayoutEnum.from_tensor(mA).mma_major_mode(),
            utils.LayoutEnum.from_tensor(mB).mma_major_mode(),
            acc_dtype,
            cta_group,
            (mma_inst_shape_m, mma_inst_shape_n),
        )

        mma_inst_tile_k = (
            4  # 4 MMAs per MMA tile K. For 16b types, tcgen05.mma has K=16.
        )
        mma_inst_tile_m = mma_inst_tile_n = 1  # 1 MMAs per MMA tile M/N
        bM = mma_inst_shape_m * mma_inst_tile_m
        bN = mma_inst_shape_n * mma_inst_tile_n
        bK = mma_inst_shape_k * mma_inst_tile_k
        mnk_tiler = (bM, bN, bK)

        cta_m, cta_n, _ = cute.arch.block_idx()
        tid_x, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout(CLUSTER_SHAPE),
            cute.core._pack_shape((cute.size(tiled_mma.thr_id.shape),)),
        )
        cluster_layout_v_size = cute.size(cluster_layout_vmnk.shape[0])
        mma_coord_vmnk = (
            cta_m % cluster_layout_v_size,
            cta_m // cluster_layout_v_size,
            cta_n,
        )

        gA = cute.zipped_divide(mA, (bM, bK))  # ((bM, bK), (M/bM, K/bK))
        gA_tma = cute.zipped_divide(
            mA, (bM // cluster_layout_v_size, bK)
        )  # ((bM/2, bK), (2*M/bM, K/bK))
        tAgA = gA_tma[(None, None), (cta_m, None)]  # ((bM/2, bK), (1, K/bK))

        gB_tma = cute.zipped_divide(
            mB, (bN // cluster_layout_v_size, bK)
        )  # ((bN/2, bK), (2*M/bM, K/bK))
        # ((bN/2, bK), (1, K/bK))
        tBgB = gB_tma[
            (None, None),
            (cluster_layout_v_size * cta_n + cta_m % cluster_layout_v_size, None),
        ]

        gD_tma = cute.zipped_divide(
            mD, (bM // cluster_layout_v_size, bN)
        )  # ((bM/2, bN), (2*M/bM, N/bN))
        tDgD = gD_tma[(None, None), (cta_m, cta_n)]  # ((bM/2, bN), (1, 1))

        a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma,
            mnk_tiler,
            ab_dtype,
            MAINLOOP_STAGE_DEPTH,
        )
        b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma,
            mnk_tiler,
            ab_dtype,
            MAINLOOP_STAGE_DEPTH,
        )

        cta_tile_shape_mnk = cute.shape_div(mnk_tiler, (cluster_layout_v_size, 1, 1))
        epi_tile = sm100_utils.compute_epilogue_tile_shape(
            cta_tile_shape_mnk,
            use_2cta_instrs,
            d_layout,
            d_dtype,
        )
        sc_smem_layout_staged = sm100_utils.make_smem_layout_epi(
            d_dtype,
            d_layout,
            epi_tile,
            TMA_STORE_PIPE_DEPTH,
        )

        tmem_layout = cute_ext.make_tmem_layout_acc(
            tiled_mma, mnk_tiler, EPILOGUE_STAGE_DEPTH
        )

        bufferA = cute_ext.allocate(
            ab_dtype,
            cute.AddressSpace.smem,
            a_smem_layout_staged,
            alignment=1024,
        )

        bufferB = cute_ext.allocate(
            ab_dtype,
            cute.AddressSpace.smem,
            b_smem_layout_staged,
            alignment=1024,
        )

        bufferAcc = cute_ext.allocate(
            acc_dtype,
            cute.AddressSpace.tmem,
            tmem_layout,
            alignment=16,
            is2cta=use_2cta_instrs,
        )

        bufferC = cute_ext.allocate(
            d_dtype,
            cute.AddressSpace.smem,
            sc_smem_layout_staged,
            alignment=1024,
        )

        copy_atom_t2r = sm100_utils.get_tmem_load_op(
            cta_tile_shape_mnk,
            d_layout,
            d_dtype,
            acc_dtype,
            epi_tile,
            use_2cta_instrs,
        )

        # Take only one stage of the TMEM buffer for the epilogue
        accumulators = cute.zipped_divide(bufferAcc, ((epi_tile), 1))
        acc_epi_div = accumulators[((None, None), 0), 0]

        # Create the TMEM copy atom based on the size of transfer within one iteration of epilogue
        tiled_copy_t2r = cute.nvgpu.tcgen05.make_tmem_copy(copy_atom_t2r, acc_epi_div)

        # Calculate the per thread destination size per iteration for output of TMEM and input of SMEM
        gC_mnl_epi = cute.flat_divide(tDgD, epi_tile)
        acc_d_rmem_layout = cute_ext.make_t2r_rmem_layout(
            tiled_copy_t2r, gC_mnl_epi, tid_x
        )

        bufferRAcc = cute_ext.allocate(
            acc_dtype,
            cute.AddressSpace.rmem,
            acc_d_rmem_layout,
            alignment=32,
        )
        bufferRD = cute_ext.allocate(
            d_dtype,
            cute.AddressSpace.rmem,
            acc_d_rmem_layout,
            alignment=32,
        )

        tma_mcast_proj_A = 2
        tma_mcast_proj_B = 1

        mma_operation_type = tma_operation_type = None
        acc_pipe = mainloop_pipe = None
        if cutlass.const_expr(use_2cta_instrs):
            mma_operation_type = cute_ext.OperationTypeEnum.SM100_MMA_2SM_SS
            if cutlass.const_expr(use_tma_multicast):
                tma_operation_type = (
                    cute_ext.OperationTypeEnum.SM100_TMA_LOAD_2SM_MULTICAST
                )
            else:
                tma_operation_type = cute_ext.OperationTypeEnum.SM100_TMA_LOAD_2SM

        else:
            mma_operation_type = cute_ext.OperationTypeEnum.SM100_MMA_1SM_SS
            if cutlass.const_expr(use_tma_multicast):
                tma_operation_type = cute_ext.OperationTypeEnum.SM90_TMA_LOAD_MULTICAST
            else:
                tma_operation_type = cute_ext.OperationTypeEnum.SM90_TMA_LOAD

        # MMA <-> TMEM load pipeline
        # if 2CTA MMA, warpgroup from both peer and leader CTA consumer.release
        acc_pipe_consumer_arv_count = 256 if use_2cta_instrs else 128
        acc_pipe = cute_ext.UMMAtoAsyncPipeline.create(
            num_stages=EPILOGUE_STAGE_DEPTH,
            mma_operation_type=mma_operation_type,
            consumer=cute_ext.OperationTypeEnum.SM100_COPY_T2R,
            consumer_arv_count=acc_pipe_consumer_arv_count,
            cluster_layout_vmnk=cluster_layout_vmnk,
        )

        if cutlass.const_expr(use_tma_multicast):
            # TMA load <-> MMA pipeline
            mainloop_pipe = cute_ext.TMAToUMMAPipeline.create_with_mask(
                num_stages=MAINLOOP_STAGE_DEPTH,
                tma_operation_type=tma_operation_type,
                mma_operation_type=mma_operation_type,
                cluster_layout_vmnk=cluster_layout_vmnk,
            )
        else:
            mainloop_pipe = cute_ext.TMAToUMMAPipeline.create(
                num_stages=MAINLOOP_STAGE_DEPTH,
                mma_operation_type=mma_operation_type,
                tma_operation_type=tma_operation_type,
                cluster_layout_vmnk=cluster_layout_vmnk,
            )

        tma_store_warp_id = 0
        mma_warp_id = 4
        tma_load_warp_id = 5
        is_tma_thr = warp_idx == tma_load_warp_id
        is_mma_thr = warp_idx == mma_warp_id
        is_epi_thr = warp_idx < 4
        is_leader_cta = mma_coord_vmnk[0] == 0

        # SMEM -> GMEM
        tma_store_pipe = cute_ext.TMAStorePipeline(
            stages=TMA_STORE_PIPE_DEPTH,
            arv_count=128,
            barrier_id=1,
            tma_warp_id=tma_store_warp_id,
        )

        k_tile_count = cute.size(gA, mode=[1, 1])
        if is_tma_thr:
            for k_tile in cutlass.range(0, k_tile_count, 1, unroll=1):
                gA_k = tAgA[None, None, k_tile]
                gB_k = tBgB[None, None, k_tile]

                producer_stage_token, idx = (
                    mainloop_pipe.producer_acquire_and_get_stage()
                )
                mbar = cute_ext.get_mbarrier(producer_stage_token)
                bufferA_sliced = bufferA[None, None, None, idx]
                bufferB_sliced = bufferB[None, None, None, idx]
                a_cta_v_map = cute_ext.get_cta_v_map_ab(mA, mnk_tiler, tiled_mma, "A")
                b_cta_v_map = cute_ext.get_cta_v_map_ab(mB, mnk_tiler, tiled_mma, "B")

                if cutlass.const_expr(use_tma_multicast):
                    cute_ext.tma_load_multicast(
                        gA_k,
                        bufferA_sliced,
                        mbar,
                        vmnk_layout=cluster_layout_vmnk,
                        cta_v_map=a_cta_v_map,
                        tma_operation_type=tma_operation_type,
                        multicast_mode=tma_mcast_proj_A,
                    )
                    cute_ext.tma_load_multicast(
                        gB_k,
                        bufferB_sliced,
                        mbar,
                        vmnk_layout=cluster_layout_vmnk,
                        cta_v_map=b_cta_v_map,
                        tma_operation_type=tma_operation_type,
                        multicast_mode=tma_mcast_proj_B,
                    )
                else:
                    cute_ext.tma_load(
                        gA_k,
                        bufferA_sliced,
                        mbar,
                        cta_v_map=a_cta_v_map,
                        tma_operation_type=tma_operation_type,
                    )
                    cute_ext.tma_load(
                        gB_k,
                        bufferB_sliced,
                        mbar,
                        cta_v_map=b_cta_v_map,
                        tma_operation_type=tma_operation_type,
                    )

                if is_leader_cta:
                    mainloop_pipe.producer_commit()
                mainloop_pipe.producer_state = cute_ext.pipeline_advance_iterator(
                    mainloop_pipe.raw_pipeline, mainloop_pipe.producer_state
                )

        if is_mma_thr and is_leader_cta:
            producer_stage_token, idx = acc_pipe.producer_acquire_and_get_stage()
            accumulators_sliced = bufferAcc[None, None, None, idx]

            mma_atom = cute.make_mma_atom(tiled_mma.op)
            mma_atom.set(cute.nvgpu.tcgen05.Field.ACCUMULATE, False)
            for k_tile in cutlass.range(0, k_tile_count, 1, unroll=1):
                _, mainloop_idx = mainloop_pipe.consumer_wait_and_get_stage()
                bufferA_sliced_stage = cute.core.slice_(
                    bufferA, (None, None, None, mainloop_idx)
                )
                bufferB_sliced_stage = cute.core.slice_(
                    bufferB, (None, None, None, mainloop_idx)
                )

                for k_block in cutlass.range(mma_inst_tile_k, unroll_full=True):
                    cute_ext.dot(
                        mma_atom,
                        cute.append_ones(
                            bufferA_sliced_stage[None, None, k_block], up_to_rank=3
                        ),
                        cute.append_ones(
                            bufferB_sliced_stage[None, None, k_block], up_to_rank=3
                        ),
                        accumulators_sliced,
                    )
                    mma_atom.set(cute.nvgpu.tcgen05.Field.ACCUMULATE, True)

                mainloop_pipe.consumer_release_and_advance()

            acc_pipe.producer_commit_and_advance()

        if is_epi_thr:
            _, idx = acc_pipe.consumer_wait_and_get_stage()
            accumulators_sliced = bufferAcc[(None, None), 0, 0, idx]
            acc_epi_div_tiled = cute.flat_divide(accumulators_sliced, epi_tile)

            tiled_copy_r2s = cute.make_tiled_copy_D(
                cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), d_dtype),
                tiled_copy_t2r,
            )
            c_cta_v_map = cute_ext.get_cta_v_map_c(mD, epi_tile)

            subtile_cnt = cute.size(acc_epi_div_tiled.shape, mode=[3])
            for mn in range(subtile_cnt):
                # TMEM -> RMEM
                cute_ext.partition_and_copy(
                    tiled_copy_t2r.get_slice(tid_x),
                    acc_epi_div_tiled[None, None, 0, mn],
                    bufferRAcc,
                )

                # RMEM -> RMEM
                bufferRD.store(bufferRAcc.load().to(d_dtype))

                tma_store_pipe.acquire_sync()
                store_idx = tma_store_pipe.get_index()

                # RMEM -> SMEM
                cute_ext.partition_and_copy(
                    tiled_copy_r2s.get_slice(tid_x),
                    bufferRD,
                    bufferC[None, None, store_idx],
                )

                tma_store_pipe.commit_sync()

                if warp_idx == tma_store_warp_id:
                    cute_ext.tma_store(
                        bufferC[None, None, store_idx],
                        gC_mnl_epi[None, None, 0, mn],
                        cta_v_map=c_cta_v_map,
                    )

                tma_store_pipe.release_advance()

            tma_store_pipe.tail()
            acc_pipe.consumer_release_and_advance()

    # Return a callable that launches the kernel with proper grid/block/cluster
    @cute_ext.jit
    def launch_kernel(mA: cute.Tensor, mB: cute.Tensor, mD: cute.Tensor):
        kernel(mA, mB, mD).launch(
            grid=GRID_SHAPE,
            block=(32 * NUM_WARPS_PER_CTA, 1, 1),
            cluster=CLUSTER_SHAPE,
            smem=cute.Int64(utils.get_smem_capacity_in_bytes("sm_100")),
        )

    return launch_kernel


# ── dtype helpers ──────────────────────────────────────────────────────────────

_CUTLASS_TO_TORCH = {
    cutlass.Float16:  torch.float16,
    cutlass.BFloat16: torch.bfloat16,
    cutlass.Float32:  torch.float32,
    cutlass.TFloat32: torch.float32,
}


def _to_torch_dtype(dtype):
    if dtype not in _CUTLASS_TO_TORCH:
        raise ValueError(f"2SM kernel: unsupported dtype {dtype}")
    return _CUTLASS_TO_TORCH[dtype]


def _to_major_mode(major_str, operand):
    """Map CLI major string to OperandMajorMode.

    Conventions (matching dense_gemm.py / dense_gemm_ptr_array.py):
      A: "k" → K-major (row-major),  "m" → MN-major (col-major)
      B: "k" → K-major (row-major),  "n" → MN-major (col-major)
      D: "n" → K-major  (row-major), "m" → MN-major (col-major)
    """
    K_mode  = cute.nvgpu.tcgen05.OperandMajorMode.K
    MN_mode = cute.nvgpu.tcgen05.OperandMajorMode.MN
    if operand in ("A", "B"):
        return K_mode if major_str == "k" else MN_mode
    else:  # D
        return K_mode if major_str == "n" else MN_mode


# ── run() ──────────────────────────────────────────────────────────────────────

def run(
    mnkl: Tuple[int, int, int, int],
    ab_dtype: Type[Numeric] = cutlass.Float16,
    d_dtype: Type[Numeric] = cutlass.Float16,
    acc_dtype: Type[Numeric] = cutlass.Float32,
    a_major: str = "k",
    b_major: str = "k",
    d_major: str = "n",
    use_tma_multicast: bool = True,
    use_2cta_instrs: bool = True,
    warmup_iterations: int = 0,
    iterations: int = 1,
    skip_ref_check: bool = False,
    **kwargs,
) -> float:
    """Execute a (batched / group) 2SM dense GEMM on Blackwell with benchmarking.

    For L > 1 the kernel is invoked L times per timed iteration, once per
    batch element (group-GEMM semantics: each group has shape M×N×K).

    :param mnkl: Problem size (M, N, K, L).  L is the batch/group count.
    :param ab_dtype: Data type for A and B (Float16, BFloat16, …).
    :param d_dtype:  Data type for output D.
    :param acc_dtype: Accumulator type (Float32 recommended).
    :param a_major: 'k' = K-major (row-major), 'm' = M-major (col-major).
    :param b_major: 'k' = K-major (row-major), 'n' = N-major (col-major).
    :param d_major: 'n' = N-major (row-major), 'm' = M-major (col-major).
    :param use_tma_multicast: Enable TMA multicast across cluster CTAs.
    :param use_2cta_instrs: Use m256n256k16 2-CTA MMA (False → m128n256k16 1-CTA).
    :param warmup_iterations: Warmup runs before timing.
    :param iterations: Timed iterations.
    :param skip_ref_check: Skip torch.mm reference validation.
    :returns: Execution time in **microseconds** per iteration.
    """
    M, N, K, L = mnkl

    # ── constraint check ──────────────────────────────────────────────────────
    m_align = 256 if use_2cta_instrs else 128
    if M % m_align != 0:
        raise ValueError(
            f"M={M} must be divisible by {m_align} "
            f"({'2-CTA' if use_2cta_instrs else '1-CTA'} tile constraint)"
        )
    if N % 256 != 0:
        raise ValueError(f"N={N} must be divisible by 256 (N-tile = 256)")
    if K % 64 != 0:
        raise ValueError(f"K={K} must be divisible by 64 (K-tile = 64)")

    torch_ab = _to_torch_dtype(ab_dtype)
    torch_d  = _to_torch_dtype(d_dtype)
    majors   = (
        _to_major_mode(a_major, "A"),
        _to_major_mode(b_major, "B"),
        _to_major_mode(d_major, "D"),
    )
    dtypes = (torch_ab, torch_ab, torch_d)

    print(f"Running 2SM Dense GEMM on Blackwell:")
    print(f"  mnkl={mnkl}, 2cta={use_2cta_instrs}, tma_mcast={use_tma_multicast}")
    print(f"  ab_dtype={ab_dtype}, d_dtype={d_dtype}, acc_dtype={acc_dtype}")
    print(f"  majors: A={a_major}, B={b_major}, D={d_major}")

    # ── create L batches of tensors ───────────────────────────────────────────
    torch.manual_seed(42)
    batches = []
    for _ in range(L):
        A_t, B_t, D_t, A_c, B_c, D_c = get_gemm_tensors(M, N, K, majors, dtypes)
        batches.append((A_t, B_t, D_t, A_c, B_c, D_c))

    # ── compile kernel (once, using first batch for shape inference) ──────────
    kernel_launcher = sm100_4x4x1_kernel_builder(
        use_tma_multicast, use_2cta_instrs, acc_dtype, M, N
    )
    A0_c, B0_c, D0_c = batches[0][3], batches[0][4], batches[0][5]
    compiled_kernel = cute_ext.compile(kernel_launcher, A0_c, B0_c, D0_c)

    def _run_all():
        for _, _, _, A_c, B_c, D_c in batches:
            compiled_kernel(A_c, B_c, D_c)

    # ── warmup ────────────────────────────────────────────────────────────────
    for _ in range(warmup_iterations):
        _run_all()

    # ── CUDA event timing ─────────────────────────────────────────────────────
    torch.cuda.synchronize()
    ev_start = torch.cuda.Event(enable_timing=True)
    ev_end   = torch.cuda.Event(enable_timing=True)
    ev_start.record()
    for _ in range(iterations):
        _run_all()
    ev_end.record()
    torch.cuda.synchronize()
    exec_time_us = ev_start.elapsed_time(ev_end) / iterations * 1000.0

    # ── reference check (first batch only) ───────────────────────────────────
    if not skip_ref_check:
        A_t, B_t, D_t = batches[0][0], batches[0][1], batches[0][2]
        try:
            ref = torch.mm(A_t.float(), B_t.float().T)
            torch.testing.assert_close(D_t.float(), ref, atol=1e-2, rtol=1e-2)
            print("PASS")
        except RuntimeError as e:
            if "no kernel image is available" in str(e):
                print("SKIP: Reference check skipped - GPU not supported by PyTorch")
            else:
                raise

    return exec_time_us


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    def _parse_ints(s):
        try:
            return tuple(int(x.strip()) for x in s.split(","))
        except ValueError:
            raise argparse.ArgumentTypeError(
                "Invalid format. Expected comma-separated integers."
            )

    parser = argparse.ArgumentParser(
        description="2SM Dense GEMM on Blackwell (SM100).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mnkl",
        type=_parse_ints,
        default=(256, 256, 64, 1),
        help="Problem size M,N,K,L (default: 256,256,64,1).  L>1 = group-GEMM.",
    )
    parser.add_argument("--ab_dtype",  type=cutlass.dtype, default=cutlass.Float16)
    parser.add_argument("--d_dtype",   type=cutlass.dtype, default=cutlass.Float16)
    parser.add_argument("--acc_dtype", type=cutlass.dtype, default=cutlass.Float32)
    parser.add_argument("--a_major", choices=["k", "m"], default="k")
    parser.add_argument("--b_major", choices=["k", "n"], default="k")
    parser.add_argument("--d_major", choices=["n", "m"], default="n")

    # Boolean flags with explicit enable/disable
    tma_grp = parser.add_mutually_exclusive_group()
    tma_grp.add_argument("--use_tma_multicast",
                         dest="use_tma_multicast", action="store_true")
    tma_grp.add_argument("--no_tma_multicast",
                         dest="use_tma_multicast", action="store_false")
    parser.set_defaults(use_tma_multicast=True)

    cta_grp = parser.add_mutually_exclusive_group()
    cta_grp.add_argument("--use_2cta_instrs",
                         dest="use_2cta_instrs", action="store_true")
    cta_grp.add_argument("--no_2cta_instrs",
                         dest="use_2cta_instrs", action="store_false")
    parser.set_defaults(use_2cta_instrs=True)

    parser.add_argument("--warmup_iterations", type=int, default=0)
    parser.add_argument("--iterations",        type=int, default=1)
    parser.add_argument("--skip_ref_check",    action="store_true")

    args = parser.parse_args()
    if len(args.mnkl) != 4:
        parser.error("--mnkl must have exactly 4 values: M,N,K,L")

    exec_time_us = run(
        args.mnkl,
        args.ab_dtype,
        args.d_dtype,
        args.acc_dtype,
        args.a_major,
        args.b_major,
        args.d_major,
        args.use_tma_multicast,
        args.use_2cta_instrs,
        args.warmup_iterations,
        args.iterations,
        args.skip_ref_check,
    )
    print(f"Execution time: {exec_time_us} microseconds per iteration")
