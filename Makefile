# 
# MIT License
# 
# Copyright (c) 2026 Ben Buhrow
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# 

# ============================================================================
# Makefile
# ============================================================================
CC = gcc
TOOLKIT_VERSION=13
CUDA_PATH = $(CUDA_ROOT)
CUDA_PATH = /usr/local/cuda-13
NVCC = $(CUDA_PATH)/bin/nvcc
CUSTOM_GMP_INC = ../gmp-install/6.2.0-gcc/include 
CUSTOM_GMP_LIB = ../gmp-install/6.2.0-gcc/lib 
SM = 90

CFLAGS = -I$(CUDA_PATH)/include  -I$(CUSTOM_GMP_INC) \
	-I. -Iytools -Iysieve -Iaprcl -O2 -DUSE_BMI2 -DUSE_AVX2 -DHAVE_CUDA -DTOOLKIT_VERSION=$(TOOLKIT_VERSION) \
	-fno-common -mbmi -mbmi2 -mavx2
LDFLAGS = -L$(CUDA_PATH)/lib64 -L$(CUSTOM_GMP_LIB) -Lytools \
	-lcudart -lgmp -lm -ldl -lcuda -pthread

ifeq ($(ICELAKE),1)
	CFLAGS += -DUSE_AVX512F -DUSE_AVX512BW -DSKYLAKEX -DIFMA -march=icelake-client
	SKYLAKEX = 1
else

ifeq ($(SKYLAKEX),1)
	CFLAGS += -DUSE_AVX512F -DUSE_AVX512BW -DSKYLAKEX -march=skylake-avx512 
endif
	
endif

MAIN_SRC = \
	main.c \
	gpu_cofactorization.c \
	cuda_xface.c \
	cmdOptions.c

YTOOLS_SRC = \
	ytools/threadpool.c \
	ytools/ytools.c

	
HEADERS = \
	cuda_xface.h \
	gpu_cofactorization.h \
	ytools/threadpool.h \
	ytools/ytools.h \
	common.h \
	cmdOptions.h

YTOOLS_OBJS = $(YTOOLS_SRC:.c=.o)
MAIN_OBJS = $(MAIN_SRC:.c=.o)

all: gpuecm64

gpuecm64: $(MAIN_OBJS) $(YTOOLS_OBJS) cuda_ecm$(SM).ptx
	$(CC) $(CFLAGS) -o gpuecm64 $(MAIN_OBJS) $(YTOOLS_OBJS) $(LDFLAGS) 

# build rules

cuda_ecm$(SM).ptx: cuda_ecm64.cu cuda_intrinsics.h
	$(NVCC) -arch sm_$(SM) -ptx -o $@ $<

%.o: %.c $(HEADERS)
	$(CC) $(CFLAGS) -c -o $@ $<
	
clean:
	rm -f *.o ytools/*.o gpuecm64