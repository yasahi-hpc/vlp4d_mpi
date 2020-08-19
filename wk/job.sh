#!/bin/sh
if ls *.p100_kokkos > /dev/null 2>&1; then
  qsub -g jh200053 batch_scripts/sub_p100_kokkos.sh
elif ls *.bdw_kokkos > /dev/null 2>&1; then
  qsub -g jh200053 batch_scripts/sub_bdw_kokkos.sh
elif ls *.pacs_knl_openmp > /dev/null 2>&1; then
  pjsub batch_scripts/sub_pacs_knl_omp.sh
else
  echo "No executable!"
fi
