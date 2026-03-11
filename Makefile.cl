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

# NOTE: much of this makefile was developed with the assistance of Claude
# =============================================================================
# Makefile -- OpenCL ECM / cofactorization project
#
# Targets
# -------
#   make              -- build gpu_ecm64 (default)
#   make clean        -- remove build artefacts
#
# Platform detection
# ------------------
# Primary target: MSYS2 MinGW64 on Windows with AMD Adrenalin OpenCL.
#   pacman packages needed:
#     mingw-w64-x86_64-gcc
#     mingw-w64-x86_64-opencl-headers
#     mingw-w64-x86_64-opencl-icd
#     mingw-w64-x86_64-gmp  (or build GMP yourself)
#
# The Makefile also detects Linux automatically so the same file works
# on a Linux dev machine (e.g. with ROCm or Mesa Rusticl OpenCL).
#
# Usage on MSYS2:
#   cd <project_dir>
#   make
#
# Usage on Linux:
#   make
# =============================================================================

# ---------------------------------------------------------------------------
# Platform detection
# ---------------------------------------------------------------------------
ifeq ($(OS),Windows_NT)
    PLATFORM   := windows
    EXE_SUFFIX := .exe
else
    UNAME := $(shell uname -s)
    ifeq ($(UNAME),Linux)
        PLATFORM   := linux
        EXE_SUFFIX :=
    else
        PLATFORM   := unknown
        EXE_SUFFIX :=
    endif
endif

# ---------------------------------------------------------------------------
# Toolchain
# ---------------------------------------------------------------------------
CC      := gcc
AR      := ar
ARFLAGS := rcs

# ---------------------------------------------------------------------------
# OpenCL paths
#
# MSYS2/MinGW64:  opencl-icd installs headers to /mingw64/include/CL and
#                 the import library to /mingw64/lib/libOpenCL.a
# Linux (system): /usr/include/CL  +  -lOpenCL  (from ocl-icd or vendor)
# Linux (ROCm):   /opt/rocm/include + /opt/rocm/lib
#
# Override on the command line if your layout differs:
#   make OCL_INCLUDE=/my/path/include OCL_LIB=/my/path/lib
# ---------------------------------------------------------------------------
ifeq ($(PLATFORM),windows)
    # MSYS2 MinGW64 default layout
    OCL_INCLUDE ?= /mingw64/include
    OCL_LIB     ?= /mingw64/lib
    OCL_LDFLAGS  = -L$(OCL_LIB) -lOpenCL
else
    # Linux: try ROCm first, fall back to system
    ifneq ($(wildcard /opt/rocm/include/CL/cl.h),)
        OCL_INCLUDE ?= /opt/rocm/include
        OCL_LIB     ?= /opt/rocm/lib
    else
        OCL_INCLUDE ?= /usr/include
        OCL_LIB     ?= /usr/lib
    endif
    OCL_LDFLAGS = -L$(OCL_LIB) -lOpenCL
endif

# ---------------------------------------------------------------------------
# GMP paths
#
# MSYS2: mingw-w64-x86_64-gmp puts gmp.h in /mingw64/include
# Linux: usually /usr/include with -lgmp
# Override: make GMP_INCLUDE=/path GMP_LIB=/path
# ---------------------------------------------------------------------------
ifeq ($(PLATFORM),windows)
    GMP_INCLUDE ?= /mingw64/include
    GMP_LIB     ?= /mingw64/lib
else
    GMP_INCLUDE ?= /usr/include
    GMP_LIB     ?= /usr/lib
endif
GMP_LDFLAGS = -L$(GMP_LIB) -lgmp

# ---------------------------------------------------------------------------
# Project source layout
#
# Assumption: all translated .c files live in the same directory as this
# Makefile.  Headers for stubs (batch_factor.h, microecm.h) are expected
# here too.  Adjust SRC_DIR / INC_DIR if your tree differs.
# ---------------------------------------------------------------------------
SRC_DIR := .
OBJ_DIR := obj

# Source files we own
SRCS := \
    $(SRC_DIR)/ocl_xface.c          \
    $(SRC_DIR)/gpu_cofactorization_cl.c \
    $(SRC_DIR)/ytools/ytools.c \
    $(SRC_DIR)/ytools/threadpool.c \
    $(SRC_DIR)/cmdOptions.c \
	$(SRC_DIR)/mpz-ull.c \
    $(SRC_DIR)/main_cl.c

# Object files
OBJS := $(patsubst $(SRC_DIR)/%.c, $(OBJ_DIR)/%.o, $(SRCS))

# ---------------------------------------------------------------------------
# Compiler flags
# ---------------------------------------------------------------------------
CFLAGS := \
    -std=c11                        \
    -O2                             \
    -Wall -Wextra                   \
    -Wno-unused-parameter           \
    -DHAVE_CUDA_BATCH_FACTOR        \
    -I$(SRC_DIR)                    \
	-I$(SRC_DIR)/ytools/            \
    -I$(OCL_INCLUDE)                \
    -I$(GMP_INCLUDE)
	
# include these for mingw builds:
ifeq ($(PLATFORM),windows)
    CFLAGS += -DULL_NO_UL -DBITS_PER_GMP_ULONG=32
endif

# Debug build: make DEBUG=1
ifeq ($(DEBUG),1)
    CFLAGS += -g -O0 -DDEBUG
endif

# ---------------------------------------------------------------------------
# Linker flags
# ---------------------------------------------------------------------------

# note: liblasieve must be listed before lgmp
LDFLAGS := \
    $(OCL_LDFLAGS)  \
    $(GMP_LDFLAGS)  \
    -lm

# On Windows, also link the math and C runtime explicitly when using MinGW
ifeq ($(PLATFORM),windows)
    LDFLAGS += -static-libgcc
endif

# ---------------------------------------------------------------------------
# Primary target
# ---------------------------------------------------------------------------
TARGET := gpuecm64_cl$(EXE_SUFFIX)

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) -o $@ $^ $(LDFLAGS)
	@echo "Built $@"

# ---------------------------------------------------------------------------
# Compilation rules
# ---------------------------------------------------------------------------
$(OBJ_DIR):
	mkdir -p $(OBJ_DIR) \
	mkdir -p $(OBJ_DIR)/ytools

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c | $(OBJ_DIR)
	$(CC) $(CFLAGS) -c -o $@ $<

# ---------------------------------------------------------------------------
# Dependency tracking (auto-generated .d files)
# ---------------------------------------------------------------------------
DEPS := $(OBJS:.o=.d)
-include $(DEPS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c | $(OBJ_DIR)
	$(CC) $(CFLAGS) -MMD -MP -c -o $@ $<

# ---------------------------------------------------------------------------
# Clean
# ---------------------------------------------------------------------------
clean:
	rm -rf $(OBJ_DIR)
	rm -f $(TARGET) 
	@echo "Cleaned"

# ---------------------------------------------------------------------------
# Help / diagnostics
# ---------------------------------------------------------------------------
.PHONY: info
info:
	@echo "PLATFORM    = $(PLATFORM)"
	@echo "CC          = $(CC)"
	@echo "OCL_INCLUDE = $(OCL_INCLUDE)"
	@echo "OCL_LIB     = $(OCL_LIB)"
	@echo "GMP_INCLUDE = $(GMP_INCLUDE)"
	@echo "GMP_LIB     = $(GMP_LIB)"
	@echo "CFLAGS      = $(CFLAGS)"
	@echo "LDFLAGS     = $(LDFLAGS)"
	@echo "SRCS        = $(SRCS)"
	@echo "OBJS        = $(OBJS)"
