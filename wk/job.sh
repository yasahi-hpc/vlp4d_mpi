#!/bin/sh
# Tsubame p100
if ls *.tsubame3.0_p100_kokkos > /dev/null 2>&1; then
  qsub -g jh200053 batch_scripts/sub_tsubame3.0_p100_kokkos.sh
elif ls *.tsubame3.0_p100_openacc > /dev/null 2>&1; then
  qsub -g jh200053 batch_scripts/sub_tsubame3.0_p100_acc.sh
# Tsubame broadwell
elif ls *.tsubame3.0_bdw_kokkos > /dev/null 2>&1; then
  qsub -g jh200053 batch_scripts/sub_tsubame3.0_bdw_kokkos.sh
elif ls *.tsubame3.0_bdw_openmp > /dev/null 2>&1; then
  qsub -g jh200053 batch_scripts/sub_tsubame3.0_bdw_omp.sh
# Oakforest pacs
elif ls *.pacs_knl_openmp > /dev/null 2>&1; then
  pjsub batch_scripts/sub_pacs_knl_omp.sh
# JFRS1 skx
elif ls *.jfrs1_skx_openmp > /dev/null 2>&1; then
  sbatch batch_scripts/sub_jfrs1_skx_omp.sh
elif ls *.jfrs1_skx_kokkos > /dev/null 2>&1; then
  sbatch batch_scripts/sub_jfrs1_skx_kokkos.sh
# Flow machine
elif ls *.flow_a64fx_kokkos > /dev/null 2>&1; then
  pjsub batch_scripts/sub_flow_a64fx_kokkos.sh
elif ls *.flow_a64fx_openmp > /dev/null 2>&1; then
  pjsub batch_scripts/sub_flow_a64fx_omp.sh
# Fugaku
elif ls *.fugaku_a64fx_kokkos > /dev/null 2>&1; then
  pjsub batch_scripts/sub_fugaku_a64fx_kokkos.sh
elif ls *.fugaku_a64fx_openmp > /dev/null 2>&1; then
  pjsub batch_scripts/sub_fugaku_a64fx_omp.sh
else
  echo "No executable!"
fi
