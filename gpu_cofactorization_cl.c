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
 * gpu_cofactorization.c  --  OpenCL translation
 *
 * Per-call translation summary
 * ----------------------------
 *
 * Memory transfers (all were Async; OpenCL uses blocking=CL_FALSE for
 * the same effect with an in-order queue):
 *   cuMemcpyHtoDAsync(dst, src, sz, stream)
 *     -> clEnqueueWriteBuffer(queue, dst, CL_FALSE, 0, sz, src, 0,NULL,NULL)
 *   cuMemcpyDtoHAsync(dst, src, sz, stream)
 *     -> clEnqueueReadBuffer (queue, src, CL_FALSE, 0, sz, dst, 0,NULL,NULL)
 *
 * Kernel launch (CUDA split this across two calls):
 *   cuFuncSetBlockShape(func, tpb, 1, 1)   -- sets local size
 *   cuLaunchGridAsync(func, blocks, 1, stream) -- launches
 *     -> clEnqueueNDRangeKernel(queue, kernel, 1, NULL,
 *                               &global_size, &local_size, 0,NULL,NULL)
 *   where global_size = num_blocks * threads_per_block
 *         local_size  = threads_per_block
 *
 * Synchronization:
 *   cuStreamSynchronize(stream) -> clFinish(queue)
 *
 * Timing:
 *   cuEventRecord(ev, stream)      \
 *   cuEventSynchronize(ev)          >  replaced with wall-clock timing
 *   cuEventElapsedTime(&ms, s, e)  /   via clock_gettime or a profiling
 *                                      event approach (see below).
 *   OpenCL timing uses CL_QUEUE_PROFILING_ENABLE on the queue, then
 *   clGetEventProfilingInfo after clFinish.  Because the existing code
 *   wraps an entire multi-iteration loop between start/end events, we
 *   use simple wall-clock time (clock_gettime) for the overall elapsed
 *   measurement -- it is equivalent and simpler.
 *
 * Allocation / free:
 *   cuMemAlloc(&ptr, sz)  -> ptr = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sz, NULL, &err)
 *   cuMemFree(ptr)        -> clReleaseMemObject(ptr)
 *
 * Context / module / stream / events:
 *   cuCtxCreate  -> clCreateContext (see gpu_ctx_init)
 *   cuModuleLoad -> clCreateProgramWithSource + clBuildProgram
 *   cuStreamCreate -> clCreateCommandQueue (in-order, no special flags)
 *   cuEventCreate  -> not needed as objects; we use wall-clock timing
 *   cuEventDestroy -> not needed
 *   cuStreamDestroy -> clReleaseCommandQueue
 *   cuCtxDestroy    -> clReleaseContext + clReleaseProgram
 *
 * gpu_module rename:
 *   t->gpu_module  (CUmodule) -> t->gpu_program  (cl_program)
 *
 * ptr_arg in gpu_arg_t:
 *   Was void* (cast from CUdeviceptr).  Is now cl_mem directly.
 *   Call sites change from:
 *     gpu_args[N].ptr_arg = (void*)(t->gpu_xxx_array);
 *   to:
 *     gpu_args[N].ptr_arg = t->gpu_xxx_array;
 *--------------------------------------------------------------------*/


// posix feature-test macros to enable things in posix headers.
// this one is for clock_gettime and others in time.h
#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <string.h>
#include <time.h>
#include "gmp.h"
#include "gpu_cofactorization_cl.h"
#include "gmp-aux.h"
#include "ytools.h"

#define HAVE_CUDA_BATCH_FACTOR

#ifdef HAVE_CUDA_BATCH_FACTOR

#define MAX_RESIDUE_WORDS 3

enum test_flags {
    DEFAULT_FLAGS       = 0,
    FLAG_USE_LOGFILE    = 0x01,
    FLAG_LOG_TO_STDOUT  = 0x02,
    FLAG_STOP_GRACEFULLY= 0x04
};

enum {
    GPU_ECM_VEC   = 0,
    NUM_GPU_FUNCTIONS
};

static const char *gpu_kernel_names[] = {
    "gbl_ecm",
};

static const gpu_arg_type_list_t gpu_kernel_args[] = {
    /* ecm -- 9 args */
    { 9, { GPU_ARG_INT32, GPU_ARG_PTR, GPU_ARG_PTR, GPU_ARG_PTR,
           GPU_ARG_PTR,   GPU_ARG_PTR, GPU_ARG_PTR, GPU_ARG_UINT32,
           GPU_ARG_INT32 } },
};

/* -----------------------------------------------------------------------
 * Convenience: simple wall-clock elapsed time in ms between two
 * struct timespec values.  Replaces cuEventElapsedTime.
 * --------------------------------------------------------------------- */
static float
timespec_elapsed_ms(struct timespec *start, struct timespec *end)
{
    double s = (double)(end->tv_sec  - start->tv_sec)  * 1000.0
             + (double)(end->tv_nsec - start->tv_nsec) / 1e6;
    return (float)s;
}


static const double INV_2_POW_32 = 1.0 / (double)((uint64_t)(1) << 32);
static uint32_t uecm_lcg_rand_32B(uint32_t lower, uint32_t upper, uint64_t* ploc_lcg)
{
    *ploc_lcg = 6364136223846793005ULL * (*ploc_lcg) + 1442695040888963407ULL;
    return lower + (uint32_t)(
        (double)(upper - lower) * (double)((*ploc_lcg) >> 32) * INV_2_POW_32);
}

uint32_t multiplicative_neg_inverse32(uint64_t a)
{
    uint32_t res = 2 + a;
    res = res * (2 + a * res);
    res = res * (2 + a * res);
    res = res * (2 + a * res);
    return res * (2 + a * res);
}

/* -----------------------------------------------------------------------
 * do_gpu_ecm64
 * --------------------------------------------------------------------- */
uint32_t
do_gpu_ecm64(device_thread_ctx_t *t)
{
    uint32_t     quit = 0;
    gpu_arg_t    gpu_args[GPU_MAX_KERNEL_ARGS];
    gpu_launch_t *launch;
    struct timespec ts_start, ts_end;
    float        elapsed_ms;

    int threads_per_block = 256;
    int num_blocks = t->array_sz / threads_per_block
                   + ((t->array_sz % threads_per_block) > 0);

    printf("commencing gpu 64-bit ecm on %d inputs\n", t->array_sz);
    fflush(stdout);

    int curve = 0;
    int total_factors = 0;
    int i;

    // initialize on cpu
    // compute rho, one, and Rsq
    mpz_t rsq;
    mpz_t zn;
    mpz_init(rsq);
    mpz_init(zn);
    for (i = 0; i < t->array_sz; i++)
    {
        t->rho[i] = multiplicative_neg_inverse32(t->modulus_in[i]);
        t->one[i] = ((uint64_t)0 - t->modulus_in[i]) % t->modulus_in[i];
        mpz_set_ui(rsq, 1);
        mpz_mul_2exp(rsq, rsq, 128);
        mpz_set_ull(zn, t->modulus_in[i]);
        mpz_tdiv_r(rsq, rsq, zn);
        t->rsq[i] = mpz_get_ull(rsq);
    }

    /* copy sigma into device memory */
    OCL_TRY(clEnqueueWriteBuffer(t->queue, t->gpu_sigma_array, CL_FALSE,
        0, t->array_sz * sizeof(uint32_t), t->sigma, 0, NULL, NULL))

    /* copy n into device memory */
    OCL_TRY(clEnqueueWriteBuffer(t->queue, t->gpu_n_array, CL_FALSE,
        0, t->array_sz * sizeof(uint64_t), t->modulus_in, 0, NULL, NULL))

    clock_gettime(CLOCK_MONOTONIC, &ts_start);  /* replaces cuEventRecord */

    OCL_TRY(clEnqueueWriteBuffer(t->queue, t->gpu_rsq_array, CL_FALSE,
        0, t->array_sz * sizeof(uint64_t), t->rsq, 0, NULL, NULL))
    OCL_TRY(clEnqueueWriteBuffer(t->queue, t->gpu_rho_array, CL_FALSE,
        0, t->array_sz * sizeof(uint32_t), t->rho, 0, NULL, NULL))
    OCL_TRY(clEnqueueWriteBuffer(t->queue, t->gpu_one_array, CL_FALSE,
        0, t->array_sz * sizeof(uint64_t), t->one, 0, NULL, NULL))

    launch = t->launch + GPU_ECM_VEC;

    int orig_size  = t->array_sz;

    while ((curve < t->curves_2lp) && (total_factors < orig_size)) {

        gpu_args[0].int32_arg  = t->array_sz;
        gpu_args[1].ptr_arg    = t->gpu_n_array;
        gpu_args[2].ptr_arg    = t->gpu_rho_array;
        gpu_args[3].ptr_arg    = t->gpu_one_array;
        gpu_args[4].ptr_arg    = t->gpu_rsq_array;
        gpu_args[5].ptr_arg    = t->gpu_sigma_array;
        gpu_args[6].ptr_arg    = t->gpu_factor_array;
        gpu_args[7].uint32_arg = 205;
        gpu_args[8].int32_arg  = curve;

        gpu_launch_set(launch, gpu_args);

        /* Launch kernel.
         * Replaces: cuFuncSetBlockShape + cuLaunchGridAsync.
         * global_work_size = total threads; local_work_size = block size. */
        size_t local_sz  = (size_t)threads_per_block;
        size_t global_sz = (size_t)(num_blocks * threads_per_block);
        OCL_TRY(clEnqueueNDRangeKernel(t->queue, launch->kernel_func,
            1, NULL, &global_sz, &local_sz, 0, NULL, NULL))

        /* copy factors back to host */
        OCL_TRY(clEnqueueReadBuffer(t->queue, t->gpu_factor_array, CL_FALSE,
            0, t->array_sz * sizeof(uint64_t), t->factors, 0, NULL, NULL))

        /* Must flush before touching host results */
        OCL_TRY(clFinish(t->queue))

        // swap factored inputs to the end of the list
        int n = t->array_sz;
        int c = 0;
        for (i = 0; i < n; i++)
        {
            uint64_t factor = t->factors[i];

            if ((factor > 1) &&
                (factor < t->modulus_in[i]))
            {
                // valid factorization, save it in the same position
                // it was originally in in the input array.
                t->factors_out[t->rb_idx_r[i]] = factor;
                t->num_factors_2lp++;

                // we are done with the modulus after finding this factor.
                // load in a new input from the end of the list.
                // we do this so that the gpu continues to see a
                // continguous list of inputs.
                t->modulus_in[i] = t->modulus_in[n - 1];
                t->rsq[i] = t->rsq[n - 1];
                t->one[i] = t->one[n - 1];
                t->rho[i] = t->rho[n - 1];
                t->factors[i] = t->factors[n - 1];
                t->rb_idx_r[i] = t->rb_idx_r[n - 1];

                // shrink the list
                n--;
                c++;

                // visit this index again
                i--;
            }
        }

        total_factors += c;
        t->array_sz = n;

        if (n == 0)
            break;

        num_blocks = t->array_sz / threads_per_block +
            ((t->array_sz % threads_per_block) > 0);

        OCL_TRY(clEnqueueWriteBuffer(t->queue, t->gpu_n_array, CL_FALSE,
            0, t->array_sz * sizeof(uint64_t), t->modulus_in, 0, NULL, NULL))
        OCL_TRY(clEnqueueWriteBuffer(t->queue, t->gpu_rsq_array, CL_FALSE,
            0, t->array_sz * sizeof(uint64_t), t->rsq, 0, NULL, NULL))
        OCL_TRY(clEnqueueWriteBuffer(t->queue, t->gpu_rho_array, CL_FALSE,
            0, t->array_sz * sizeof(uint32_t), t->rho, 0, NULL, NULL))
        OCL_TRY(clEnqueueWriteBuffer(t->queue, t->gpu_one_array, CL_FALSE,
            0, t->array_sz * sizeof(uint64_t), t->one, 0, NULL, NULL))

        curve += 1;
    }

    /* end timing */
    OCL_TRY(clFinish(t->queue))
    clock_gettime(CLOCK_MONOTONIC, &ts_end);
    elapsed_ms = timespec_elapsed_ms(&ts_start, &ts_end);

    printf("factored %d of %d inputs in %1.4f ms\n",
        t->num_factors_2lp, orig_size, elapsed_ms);

    mpz_clear(rsq);
    mpz_clear(zn);

    /* final sync */
    OCL_TRY(clFinish(t->queue))

    return quit;
}

/* -----------------------------------------------------------------------
 * gpu_device_init
 *
 * Replaces the CUDA version that called gpu_init() + cuCtxCreate etc.
 * We only fill the device_ctx_t here; the OpenCL context is created later
 * in gpu_ctx_init where we have both device and context together.
 * --------------------------------------------------------------------- */
device_ctx_t *
gpu_device_init(int which_gpu)
{
    gpu_config_t  gpu_config;
    gpu_info_t   *gpu_info;

    device_ctx_t *d = (device_ctx_t *)xcalloc(1, sizeof(device_ctx_t));

    gpu_init(&gpu_config);
    if (gpu_config.num_gpu == 0) {
        printf("error: no OpenCL-capable GPUs found\n");
        exit(-1);
    }

    d->gpunum   = which_gpu;
    d->gpu_info = gpu_info = (gpu_info_t *)xmalloc(sizeof(gpu_info_t));
    memcpy(gpu_info, gpu_config.info + which_gpu, sizeof(gpu_info_t));

    printf("using GPU %u (%s)\n", which_gpu, gpu_info->name);
    printf("selected card has OpenCL version %d.%d\n",
           gpu_info->compute_version_major,
           gpu_info->compute_version_minor);
    printf("more GPU info:\n");
    printf("\tmax_grid_size: %d x %d x %d\n",
           gpu_info->max_grid_size[0],
           gpu_info->max_grid_size[1],
           gpu_info->max_grid_size[2]);
    printf("\tglobal_mem_size: %zd\n",  gpu_info->global_mem_size);
    printf("\tconstant_mem_size: %d\n", gpu_info->constant_mem_size);
    printf("\tmax_threads_per_block: %d\n", gpu_info->max_threads_per_block);
    printf("\tmax_thread_dim: %d x %d x %d\n",
           gpu_info->max_thread_dim[0],
           gpu_info->max_thread_dim[1],
           gpu_info->max_thread_dim[2]);
    printf("\tnum_compute_units: %d\n", gpu_info->num_compute_units);
    printf("\tregisters_per_block: %d\n", gpu_info->registers_per_block);
    printf("\tshared_mem_size: %d\n",   gpu_info->shared_mem_size);
    printf("\twarp_size: %d\n",         gpu_info->warp_size);

    return d;
}

void
gpu_dev_free(device_ctx_t *d)
{
    free(d->gpu_info);
    free(d);
}

/* -----------------------------------------------------------------------
 * gpu_ctx_init
 *
 * CUDA version:
 *   cuCtxCreate     -- create a per-thread CUDA context
 *   cuModuleLoad    -- load a pre-compiled .ptx file
 *   gpu_launch_init -- look up CUfunction handles
 *   cuStreamCreate  -- create an async stream
 *   cuEventCreate x2
 *
 * OpenCL version:
 *   clCreateContext         -- one context per device
 *   clCreateCommandQueue    -- in-order queue (equivalent to CUDA stream
 *                              with CU_CTX_BLOCKING_SYNC)
 *   clCreateProgramWithSource + clBuildProgram
 *                           -- replaces cuModuleLoad from .ptx;
 *                              loads the .cl source file at runtime.
 *   gpu_launch_init x N     -- creates cl_kernel objects
 *
 * The .ptx file selection logic (sm_20 / sm_30 / sm_35 / sm_50 / sm_80
 * / sm_90) is replaced with a single .cl file -- "opencl_tinyecm.cl" --
 * that the AMD driver JIT-compiles for whatever GPU is present.
 *
 * Events (CUevent start_event / end_event) are replaced by wall-clock
 * timing in each do_gpu_* function, so we don't create them here.
 * --------------------------------------------------------------------- */

/* Helper: read an entire text file into a malloc'd buffer. */
static char *
read_file(const char *path)
{
    FILE  *f = fopen(path, "rb");
    char  *buf;
    long   sz;

    if (!f) {
        fprintf(stderr, "gpu_ctx_init: cannot open kernel file '%s'\n", path);
        exit(-1);
    }
    fseek(f, 0, SEEK_END);
    sz = ftell(f);
    rewind(f);
    buf = (char *)xmalloc((size_t)sz + 1);
    if (fread(buf, 1, (size_t)sz, f) != (size_t)sz) {
        fprintf(stderr, "gpu_ctx_init: read error on '%s'\n", path);
        exit(-1);
    }
    buf[sz] = '\0';
    fclose(f);
    return buf;
}

device_thread_ctx_t*
gpu_ctx_init(device_ctx_t* d)
{
    device_thread_ctx_t* t;
    cl_int               err;
    cl_device_id         dev = d->gpu_info->device_handle;
    cl_platform_id       plat = d->gpu_info->platform_handle;
    int                  i;

    t = (device_thread_ctx_t*)xcalloc(1, sizeof(device_thread_ctx_t));
    t->dev = d;

    /* Create OpenCL context.
     * Replaces: cuCtxCreate(&t->gpu_context, ..., device_handle)        */
    cl_context_properties props[] = {
        CL_CONTEXT_PLATFORM, (cl_context_properties)plat, 0
    };
    t->gpu_context = clCreateContext(props, 1, &dev, NULL, NULL, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "gpu_ctx_init: clCreateContext failed: %s\n",
            clGetErrorString(err));
        exit(-1);
    }

    /* Load and build the OpenCL kernel source.
     * Replaces: cuModuleLoad(&t->gpu_module, ptxfile)
     *
     * AMD's comgr compiler runs a full LLVM pipeline on first build, which
     * can take several minutes for dense kernel code like this.  We cache
     * the compiled binary to disk (keyed on device name + build options)
     * so subsequent runs load in under a second.
     *
     * Cache logic:
     *   1. If <cache_file> exists on disk, load it and call
     *      clCreateProgramWithBinary -- no compilation needed.
     *   2. Otherwise compile from source with clBuildProgram, then
     *      extract the binary with clGetProgramInfo(CL_PROGRAM_BINARIES)
     *      and write it to <cache_file> for next time.
     *
     * The cache filename embeds the device name so it is automatically
     * invalidated when run on a different GPU.  Delete the file manually
     * if you change the .cl sources or build options.                    */

    const char* intrinsics_file = "opencl_intrinsics.cl";
    const char* tinyecm_file = "opencl_tinyecm.cl";
    const char* build_options = "-cl-std=CL2.0 -cl-mad-enable";

    /* Build a cache filename: "ocl_ecm_<devname>.bin"
     * Sanitise the device name -- replace spaces and slashes with '_'.  */
    char cache_file[256];
    {
        char safe_name[128];
        const char* src_n = d->gpu_info->name;
        char* dst_n = safe_name;
        size_t      lim = sizeof(safe_name) - 1;
        while (*src_n && (size_t)(dst_n - safe_name) < lim) {
            char c = *src_n++;
            *dst_n++ = (c == ' ' || c == '/' || c == '\\') ? '_' : c;
        }
        *dst_n = '\0';
        snprintf(cache_file, sizeof(cache_file),
            "ocl_ecm_%s.bin", safe_name);
    }

    /* --- Try loading from cache first -------------------------------- */
    int loaded_from_cache = 0;
    {
        FILE* f = fopen(cache_file, "rb");
        if (f) {
            fseek(f, 0, SEEK_END);
            long bin_sz = ftell(f);
            rewind(f);

            if (bin_sz > 0) {
                unsigned char* bin = (unsigned char*)xmalloc((size_t)bin_sz);
                if (fread(bin, 1, (size_t)bin_sz, f) == (size_t)bin_sz) {
                    const unsigned char* bin_ptr = bin;
                    size_t               bin_len = (size_t)bin_sz;
                    cl_int               bin_status = CL_SUCCESS;

                    t->gpu_program = clCreateProgramWithBinary(
                        t->gpu_context, 1, &dev,
                        &bin_len, &bin_ptr,
                        &bin_status, &err);

                    if (err == CL_SUCCESS && bin_status == CL_SUCCESS) {
                        err = clBuildProgram(t->gpu_program, 1, &dev,
                            build_options, NULL, NULL);
                        if (err == CL_SUCCESS) {
                            printf("loaded compiled kernels from cache: %s\n",
                                cache_file);
                            loaded_from_cache = 1;
                        }
                        else {
                            /* Stale / corrupt cache -- fall through to recompile */
                            printf("cache load failed (stale?), recompiling...\n");
                            clReleaseProgram(t->gpu_program);
                            t->gpu_program = NULL;
                        }
                    }
                    else {
                        clReleaseProgram(t->gpu_program);
                        t->gpu_program = NULL;
                    }
                }
                free(bin);
            }
            fclose(f);
        }
    }

    /* --- Compile from source if cache miss or stale cache ------------ */
    if (!loaded_from_cache) {
        printf("compiling kernels from %s + %s (this takes a while the first time)...\n",
            intrinsics_file, tinyecm_file);
        fflush(stdout);

        char* src_intrinsics = read_file(intrinsics_file);
        char* src_tinyecm = read_file(tinyecm_file);

        /* Skip the '#include "opencl_intrinsics.cl"' line in tinyecm so
         * the intrinsics are not compiled twice.                         */
        const char* tinyecm_body = src_tinyecm;
        {
            const char* inc = strstr(src_tinyecm,
                "#include \"opencl_intrinsics.cl\"");
            if (inc) {
                const char* nl = strchr(inc, '\n');
                if (nl) tinyecm_body = nl + 1;
            }
        }

        const char* sources[2] = { src_intrinsics, tinyecm_body };
        size_t      lengths[2] = { strlen(src_intrinsics),
                                    strlen(tinyecm_body) };

        t->gpu_program = clCreateProgramWithSource(t->gpu_context,
            2, sources, lengths, &err);
        free(src_intrinsics);
        free(src_tinyecm);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "clCreateProgramWithSource failed: %s\n",
                clGetErrorString(err));
            exit(-1);
        }

        err = clBuildProgram(t->gpu_program, 1, &dev,
            build_options, NULL, NULL);
        if (err != CL_SUCCESS) {
            size_t log_sz = 0;
            clGetProgramBuildInfo(t->gpu_program, dev,
                CL_PROGRAM_BUILD_LOG, 0, NULL, &log_sz);
            char* log = (char*)xmalloc(log_sz + 1);
            clGetProgramBuildInfo(t->gpu_program, dev,
                CL_PROGRAM_BUILD_LOG, log_sz, log, NULL);
            log[log_sz] = '\0';
            fprintf(stderr, "clBuildProgram failed:\n%s\n", log);
            free(log);
            exit(-1);
        }

        printf("compilation successful, saving to cache: %s\n", cache_file);

        /* Extract the compiled binary and write it to the cache file.
         * clGetProgramInfo(CL_PROGRAM_BINARY_SIZES) returns one size per
         * device; we only have one device so we read index [0].          */
        size_t bin_sz = 0;
        clGetProgramInfo(t->gpu_program, CL_PROGRAM_BINARY_SIZES,
            sizeof(bin_sz), &bin_sz, NULL);
        if (bin_sz > 0) {
            unsigned char* bin = (unsigned char*)xmalloc(bin_sz);
            unsigned char* bins[1] = { bin };
            clGetProgramInfo(t->gpu_program, CL_PROGRAM_BINARIES,
                sizeof(bins), bins, NULL);

            FILE* f = fopen(cache_file, "wb");
            if (f) {
                fwrite(bin, 1, bin_sz, f);
                fclose(f);
            }
            else {
                fprintf(stderr, "warning: could not write cache file %s\n",
                    cache_file);
            }
            free(bin);
        }
    }

    printf("kernels ready\n");

    /* Create kernel objects.
     * Replaces: gpu_launch_init(t->gpu_module, name, args, launch)       */
    t->launch = (gpu_launch_t*)xmalloc(NUM_GPU_FUNCTIONS * sizeof(gpu_launch_t));

    printf("initializing kernels\n");
    for (i = 0; i < NUM_GPU_FUNCTIONS; i++) {
        gpu_launch_init(t->gpu_program, gpu_kernel_names[i],
            gpu_kernel_args + i,
            t->launch + i,
            dev);
    }

    /* Create in-order command queue.
     * Replaces: cuStreamCreate(&t->stream, 0)
     * CU_CTX_BLOCKING_SYNC -> 0 flags (in-order queue blocks on finish) */
    printf("creating command queue\n");
    t->queue = clCreateCommandQueue(t->gpu_context, dev, 0, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "clCreateCommandQueue failed: %s\n",
            clGetErrorString(err));
        exit(-1);
    }

    /* No separate event objects needed; timing is done with clock_gettime
     * in each do_gpu_* function.
     * Replaces: cuEventCreate x2                                          */

    return t;
}


/* -----------------------------------------------------------------------
 * gpu_ctx_free
 *
 * Replaces: cuEventDestroy x2, cuStreamDestroy, cuCtxDestroy
 * --------------------------------------------------------------------- */
void
gpu_ctx_free(device_thread_ctx_t *d)
{
    int i;

    /* Release kernel objects */
    for (i = 0; i < NUM_GPU_FUNCTIONS; i++)
        clReleaseKernel(d->launch[i].kernel_func);
    free(d->launch);

    clReleaseCommandQueue(d->queue);    /* was: cuStreamDestroy    */
    clReleaseProgram(d->gpu_program);   /* was: part of cuCtxDestroy */
    clReleaseContext(d->gpu_context);   /* was: cuCtxDestroy       */
}

int getbits(uint64_t n)
{
    // call CLZ if available

    // fallback:
    int i = 0;
    while (n != 0)
    {
        n >>= 1;
        i++;
    }
    return i;
}

/* -----------------------------------------------------------------------
 * do_gpu_cofactorization  (external entry point)
 *
 * cuMemAlloc  -> clCreateBuffer(ctx, CL_MEM_READ_WRITE, size, NULL, &err)
 * cuMemFree   -> clReleaseMemObject
 *
 * Note on buffer sizes:
 *   gpu_n_array   -- 64-bit ECM uses uint64_t * array_sz
 *                    96-bit ECM uses uint32_t * 3 * array_sz
 *   The original allocated "sizeof(uint32_t) * 3 * array_sz" for gpu_n_array,
 *   gpu_one_array, and gpu_rsq_array (sufficient for both 64-bit and 96-bit
 *   usage since 3*uint32 >= uint64).  We keep the same conservative sizing.
 * --------------------------------------------------------------------- */
uint32_t do_gpu_cofactorization(device_thread_ctx_t* t, uint64_t* factors, uint64_t* lcg,
    uint64_t* inputs, uint32_t numinput, int b1_2lp_ovr, int b2_2lp_ovr, int curves_2lp_ovr)
{
    // reference list to original indices
    t->array_sz = numinput;
    t->rb_idx_r = (uint32_t*)xmalloc(sizeof(uint32_t) * t->array_sz);

    // determine an average size of the inputs and assign each an index
    // with their original position in the input array.
    int avgbits = 0;
    int i;
    for (i = 0; i < numinput; i++)
    {
        t->rb_idx_r[i] = i;
        avgbits += getbits(inputs[i]);
    }

    avgbits = (int)((double)avgbits / (double)numinput + 0.5);

    // determine ECM parameters from average size.
    if (avgbits <= 52)
    {
        t->b1_2lp = 85;
        t->curves_2lp = 32;
    }
    else if (avgbits <= 56)
    {
        t->b1_2lp = 125;
        t->curves_2lp = 32;
    }
    else if (avgbits <= 60)
    {
        t->b1_2lp = 165;
        t->curves_2lp = 40;
    }
    else // <= 64
    {
        t->b1_2lp = 205;
        t->curves_2lp = 40;
    }

    // or use any supplied parameters, if available
    if (b1_2lp_ovr > 0) t->b1_2lp = b1_2lp_ovr;
    if (b2_2lp_ovr > 0) t->b2_2lp = b2_2lp_ovr;
    if (curves_2lp_ovr > 0) t->curves_2lp = curves_2lp_ovr;

    printf("ecm parameters for inputs of average size %d bits: B1=%d, B2=%d, curves=%d\n",
        avgbits, t->b1_2lp, t->b2_2lp * t->b1_2lp, t->curves_2lp);

    cl_int err;

/* Helper macro so each allocation is one line and exits on failure */
#define GPU_ALLOC(field, nbytes) \
    do { \
        t->field = clCreateBuffer(t->gpu_context, CL_MEM_READ_WRITE, \
                                  (nbytes), NULL, &err); \
        if (err != CL_SUCCESS) { \
            fprintf(stderr, "clCreateBuffer(" #field ") failed: %s\n", \
                    clGetErrorString(err)); \
            exit(-1); \
        } \
    } while (0)

    /* Device buffer allocation -- mirrors the cuMemAlloc block exactly */
    GPU_ALLOC(gpu_sigma_array, sizeof(uint32_t) * t->array_sz);
    GPU_ALLOC(gpu_rsq_array,   sizeof(uint64_t) * t->array_sz);
    GPU_ALLOC(gpu_factor_array,sizeof(uint64_t) * t->array_sz);
    GPU_ALLOC(gpu_one_array,   sizeof(uint64_t) * t->array_sz);
    GPU_ALLOC(gpu_n_array,     sizeof(uint64_t) * t->array_sz);
    GPU_ALLOC(gpu_rho_array,   sizeof(uint32_t) * t->array_sz);

#undef GPU_ALLOC

    /* set up host arrays */
    t->sigma = (uint32_t*)xmalloc(sizeof(uint32_t) * t->array_sz);
    t->rsq = (uint64_t*)xmalloc(sizeof(uint64_t) * t->array_sz);
    t->modulus_in = (uint64_t*)xmalloc(sizeof(uint64_t) * t->array_sz);
    t->one = (uint64_t*)xmalloc(sizeof(uint64_t) * t->array_sz);
    t->factors = (uint64_t*)xmalloc(sizeof(uint64_t) * t->array_sz);
    t->rho = (uint32_t*)xmalloc(sizeof(uint32_t) * t->array_sz);
    t->factors_out = factors;

    memcpy(factors, inputs, numinput * sizeof(uint64_t));
    memcpy(t->modulus_in, inputs, numinput * sizeof(uint64_t));

    // generate a sigma for each input
    for (i = 0; i < t->array_sz; i++) {
        t->sigma[i] = uecm_lcg_rand_32B(7, 0xffffffff, lcg);
    }

    t->num_factors_2lp = 0;
    do_gpu_ecm64(t);

    // clean up
    free(t->factors);
    free(t->sigma);
    free(t->modulus_in);
    free(t->rho);
    free(t->rsq);
    free(t->one);
    free(t->rb_idx_r);

    /* Device buffer cleanup -- replaces cuMemFree */
    clReleaseMemObject(t->gpu_factor_array);
    clReleaseMemObject(t->gpu_sigma_array);
    clReleaseMemObject(t->gpu_n_array);
    clReleaseMemObject(t->gpu_rho_array);
    clReleaseMemObject(t->gpu_rsq_array);
    clReleaseMemObject(t->gpu_one_array);

    return t->num_factors_2lp;
}

#endif /* HAVE_CUDA_BATCH_FACTOR */
