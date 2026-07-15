#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build}"

LOG_FILE="${LOG_FILE:-${BUILD_DIR}/run_op_s1.log}"
: > "${LOG_FILE}"

M=128
N=128
K=128
S=1

KERNEL=sgemm_tcu_op
APP_DIR="${BUILD_DIR}/tests/regression/${KERNEL}"
CONFIGS="-DNUM_THREADS=32 -DSGEMM_CONST_M=${M} -DSGEMM_CONST_N=${N} -DSGEMM_CONST_K=${K} -DSGEMM_CONST_SPARSITY=${S} -DVX_CFG_NUM_THREADS=32 -DVX_CFG_NUM_WARPS=4 -DVX_CFG_ISSUE_WIDTH=1 -DVX_CFG_TCU_TYPE=DPI -DVX_CFG_TCU_TYPE_DPI -DVX_CFG_EXT_TCU_ENABLE -DVX_CFG_EXT_DXA_ENABLE -DVX_CFG_TCU_WGMMA_ENABLE -DVX_CFG_TCU_SPARSE_ENABLE -DWGMMA_SS -DITYPE=fp16 -DOTYPE=fp32 -DPERF_ENABLE"

# make -C "${APP_DIR}" clean

build_start=$(date +%s)
env -u DEBUG CCACHE_DISABLE=1 CONFIGS="$CONFIGS" make -C "${APP_DIR}" # >> "${LOG_FILE}" 2>&1
build_status=$?
build_end=$(date +%s)

if [ ${build_status} -ne 0 ]; then
    echo "Build time (s): $((build_end - build_start))" | tee -a "${LOG_FILE}"
    exit ${build_status}
fi

run_start=$(date +%s)
env -u DEBUG CCACHE_DISABLE=1 VORTEX_PROFILING=1 CONFIGS="$CONFIGS" OPTS="-m ${M} -n ${N} -k ${K} -s ${S} -a 0.5 -b 0.5" make -C "${APP_DIR}" run-rtlsim >> "${LOG_FILE}" 2>&1
run_status=$?
run_end=$(date +%s)

echo "Build time (s): $((build_end - build_start))" | tee -a "${LOG_FILE}"
echo "Run time (s): $((run_end - run_start))" | tee -a "${LOG_FILE}"

exit ${run_status}
