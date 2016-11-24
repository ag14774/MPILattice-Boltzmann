# Add any `module load` or `export` commands that your code needs to
# compile and run to this file.

module load languages/intel-compiler-16-u2
module unload languages/gcc-4.8.4
source /cm/shared/languages/Intel-Compiler-XE-16-U2/itac/9.1.2.024/bin/itacvars.sh
export I_MPI_ASYNC_PROGRESS=1
export I_MPI_PIN_DOMAIN=core
export I_MPI_PIN_ORDER=compact
