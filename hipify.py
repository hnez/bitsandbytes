#!/usr/bin/env python3

import collections
import os.path
import re

import torch.utils.hipify.hipify_python as hp

CSRC = os.path.abspath("csrc")

# PyTorch's hipify script only contains cuda -> hip mappins that they need.
# hipify-clang --python can generate a more complete list of mappings,
# but that one is too fat to ship here.
# As a compromise I've hand-picked the primitives used in this project,
# which hipify-clang knows about but which are not included in PyTorch 1.12.1.
# I've also added some hacks to make this particular project work.
MISSING_MAPPINGS = collections.OrderedDict([
    ("cub/block/block_discontinuity.cuh", "hipcub/block/block_discontinuity.hpp"),
    ("cub/block/block_radix_sort.cuh", "hipcub/block/block_radix_sort.hpp"),
    ("cub/block/block_store.cuh", "hipcub/block/block_store.hpp"),
    ("cub/warp/warp_reduce.cuh", "hipcub/warp/warp_reduce.hpp"),
    ("cuda_runtime_api.h", "hip/hip_runtime_api.h"),
    ("#include <math_constants.h>", ""),
    ("kernels.cuh", "kernels_hip.cuh"),
    ("ops.cuh", "ops_hip.cuh"),
    ("cub::BlockExchange", "hipcub::BlockExchange"),
    ("cub::BLOCK_LOAD_DIRECT", "hipcub::BLOCK_LOAD_DIRECT"),
    ("cub::BlockLoad", "hipcub::BlockLoad"),
    ("cub::BLOCK_LOAD_VECTORIZE", "hipcub::BLOCK_LOAD_VECTORIZE"),
    ("cub::BLOCK_LOAD_WARP_TRANSPOSE", "hipcub::BLOCK_LOAD_WARP_TRANSPOSE"),
    ("cub::BlockRadixSort", "hipcub::BlockRadixSort"),
    ("cub::BLOCK_SCAN_RAKING", "hipcub::BLOCK_SCAN_RAKING"),
    ("cub::BlockStore", "hipcub::BlockStore"),
    ("cub::BLOCK_STORE_VECTORIZE", "hipcub::BLOCK_STORE_VECTORIZE"),
    ("cub::BLOCK_STORE_WARP_TRANSPOSE", "hipcub::BLOCK_STORE_WARP_TRANSPOSE"),
    ("CUBLAS_GEMM_DEFAULT", "rocblas_gemm_algo_standard"),
    ("cublasCreate_v2", "rocblas_create_handle"),
    ("cublasGemmStridedBatchedEx", "rocblas_gemm_strided_batched_ex"),
    ("cub::NullType", "hipcub::NullType"),
    ("cub::ShuffleIndex", "hipcub::ShuffleIndex"),
    ("cusparseDnMatDescr_t", "hipsparseDnMatDescr_t"),
    ("CUSPARSE_INDEX_32I", "HIPSPARSE_INDEX_32I"),
    ("CUSPARSE_ORDER_ROW", "HIPSPARSE_ORDER_ROW"),
    ("cusparseSpMatDescr_t", "hipsparseSpMatDescr_t"),
    ("CUSPARSE_SPMM_ALG_DEFAULT", "HIPSPARSE_SPMM_ALG_DEFAULT"),
    ("cusparseSpMM_bufferSize", "hipsparseSpMM_bufferSize"),
    ("__syncwarp", "__syncthreads"),
])

for src, dst in MISSING_MAPPINGS.items():
    hp.PYTORCH_TRIE.add(src)
    hp.PYTORCH_MAP[src] = dst

hp.RE_PYTORCH_PREPROCESSOR = re.compile(
    r'(?<=\W)({0})(?=\W)'.format(hp.PYTORCH_TRIE.pattern())
)

hp.hipify(
    project_directory=CSRC,
    is_pytorch_extension=True,
)
