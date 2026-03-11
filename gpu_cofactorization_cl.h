/*
MIT License

Copyright (c) 2026 Ben Buhrow

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

/*--------------------------------------------------------------------
 * gpu_cofactorization.h  --  OpenCL translation
 *
 * All CUxxx types in device_thread_ctx_t are replaced with their
 * OpenCL equivalents:
 *
 *   CUcontext    -> cl_context
 *   CUmodule     -> cl_program
 *   CUstream     -> cl_command_queue  (queues are the stream equivalent)
 *   CUevent      -> cl_event          (timing events)
 *   CUdeviceptr  -> cl_mem            (opaque device buffer handle)
 *
 * The include of cuda_xface.h is replaced with ocl_xface.h.
 *--------------------------------------------------------------------*/

#pragma once
#include <stdint.h>
#include "ocl_xface.h"

#ifdef HAVE_CUDA_BATCH_FACTOR

typedef struct {
    int         gpunum;
    gpu_info_t *gpu_info;
} device_ctx_t;

typedef struct {
    /* reference to the configured device */
    device_ctx_t *dev;

    // config from calling code
    int verbose;

    // internal config
    uint32_t b1_2lp;
    uint32_t b2_2lp;
    uint32_t curves_2lp;
    uint32_t stop_nofactor;

    /* ----------------------------------------------------------------
     * OpenCL runtime objects
     * (replaces: CUcontext, CUmodule, CUstream, CUevent x2)
     * -------------------------------------------------------------- */
    cl_context       gpu_context;   /* was: CUcontext gpu_context    */
    cl_program       gpu_program;   /* was: CUmodule  gpu_module     */
    cl_command_queue queue;         /* was: CUstream  stream         */
    gpu_launch_t    *launch;
    cl_event         start_event;   /* was: CUevent   start_event    */
    cl_event         end_event;     /* was: CUevent   end_event      */

    /* ----------------------------------------------------------------
     * Device (GPU) buffers
     * (replaces CUdeviceptr -- now cl_mem opaque handles)
     * -------------------------------------------------------------- */
    cl_mem  gpu_factor_array;       /* was: CUdeviceptr gpu_a_array      */
    cl_mem  gpu_n_array;            /* was: CUdeviceptr gpu_n_array      */
    cl_mem  gpu_one_array;          /* was: CUdeviceptr gpu_one_array    */
    cl_mem  gpu_rsq_array;          /* was: CUdeviceptr gpu_rsq_array    */
    cl_mem  gpu_sigma_array;        /* was: CUdeviceptr gpu_u32_array    */
    cl_mem  gpu_rho_array;          /* was: CUdeviceptr gpu_rho_array    */

    uint32_t array_sz;
    uint32_t flags;

    // host data
    uint64_t* factors;
    uint64_t* rsq;
    uint64_t* one;
    uint64_t* modulus_in;
    uint32_t* sigma;
    uint32_t* rho;
    uint64_t* factors_out;

    // tracking factors to input residues
    uint32_t* rb_idx_r;
    uint32_t num_factors_2lp;

} device_thread_ctx_t;

/* create and destroy gpu context info */
device_ctx_t        *gpu_device_init(int which_gpu);
device_thread_ctx_t *gpu_ctx_init(device_ctx_t *d);
void                 gpu_dev_free(device_ctx_t *d);
void                 gpu_ctx_free(device_thread_ctx_t *t);

/* do gpu cofactorization work */
uint32_t do_gpu_cofactorization(device_thread_ctx_t* t, uint64_t* factors, uint64_t* lcg,
    uint64_t* inputs, uint32_t numinput, int b1_2lp_ovr, int b2_2lp_ovr, int curves_2lp_ovr);

#endif /* HAVE_CUDA_BATCH_FACTOR */
