# gpu-ecm-64b
fast factorization of 64-bit integers using the elliptic curve method (ECM) on GPU

# CUDA build instructions
To build the CUDA code for Nvidia GPUs, use the standard Makefile as "make all"

The standard CUDA_ROOT is searched for toolkit libraries.  If you have a 
custom install location, uncomment line 31 of the Makefile and fill in your path.

The default toolkit version is assumed to be 13+.  If you are using prior versions
(12 or below), change line 29 of the Makefile to read TOOLKIT_VERSION=12

If you have a custom GMP install location, uncomment and fill in lines 33 and 34 of the Makefile.

Has been built and tested on linux and on windows (msys2/mingw64).

# OpenCL build instructions
To build the OpenCL code for Nvidia or AMD GPUs, use the alternate Makefile.cl as "make -f Makefile.cl all"

If you have nonstandard OpenCL install location, on the make line you can add OCL_INCLUDE and OCL_LIB to 
point to these locations.  For example:

make -f Makefile.cl all OCL_INCLUDE=/home/me/custom_ocl/include OCL_LIB=/home/me/custom_ocl/lib

If you have a custom GMP location, do the same on the make line for that location, for example:

make -f Makefile.cl all OCL_INCLUDE=/home/me/custom_ocl/include OCL_LIB=/home/me/custom_ocl/lib GMP_INCLUDE=/home/me/custom_gmp/include GMP_LIB=/home/me/custom_gmp/lib

Has been built and tested on linux and on windows (msys2/mingw64).

# Usage
supply a filename with 64-bit or less integers to factor, one per line, using the -f flag

an output file is created with comma-delimted input,factor lines

If a factor is not found, the factor will duplicate the input.

command line help is available with -h flag

The code is optimized for semiprime integers, i.e., it does not expect the presence of small factors in the integers.  Composite factors may be returned in that case.

example:

./gpuecm64 -f inputs.txt 



