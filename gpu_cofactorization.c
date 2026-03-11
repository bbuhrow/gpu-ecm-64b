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
#include <stdio.h>
#include <string.h>
#include "gmp.h"
#include "gpu_cofactorization.h"
#include "ytools.h"

#ifdef HAVE_CUDA

#ifndef TOOLKIT_VERSION
#define toolkit_version 13
#else
#define toolkit_version TOOLKIT_VERSION
#endif

enum test_flags {
	DEFAULT_FLAGS = 0,				/* just a placeholder */
	FLAG_USE_LOGFILE = 0x01,	    /* append log info to a logfile */
	FLAG_LOG_TO_STDOUT = 0x02,		/* print log info to the screen */
	FLAG_STOP_GRACEFULLY = 0x04		/* tell library to stop */
};

// kernel function reference
enum {
	GPU_ECM_VEC = 0,
	NUM_GPU_FUNCTIONS /* must be last */
};

// kernel function name (corresponding to an implemented function 
// in a .cu file)
static const char* gpu_kernel_names[] =
{
	"gbl_ecm",
};

// argument type lists for the kernels
static const gpu_arg_type_list_t gpu_kernel_args[] =
{
	/* ecm */
	{ 9,
		{
		  GPU_ARG_INT32,
		  GPU_ARG_PTR,
		  GPU_ARG_PTR,
		  GPU_ARG_PTR,
		  GPU_ARG_PTR,
		  GPU_ARG_PTR,
		  GPU_ARG_PTR,
		  GPU_ARG_UINT32,
		  GPU_ARG_INT32,
		}
	 },
};

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

// the function we use to go and actually do work
// using the kernels, arguments, and GPU contexts/streams defined above.
uint32_t do_gpu_ecm64(device_thread_ctx_t* t)
{
	uint32_t quit = 0;

	gpu_arg_t gpu_args[GPU_MAX_KERNEL_ARGS];

	gpu_launch_t* launch;

	float elapsed_ms;
		
	int threads_per_block = 256;
	int num_blocks = t->array_sz / threads_per_block +
		((t->array_sz % threads_per_block) > 0);

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
		t->rsq[i] = mpz_tdiv_ui(rsq, t->modulus_in[i]);
	}

	// copy sigma into device memory
	CUDA_TRY(cuMemcpyHtoDAsync(t->gpu_sigma_array,
		t->sigma,
		t->array_sz * sizeof(uint32_t),
		t->stream))

	// copy n into device memory
	CUDA_TRY(cuMemcpyHtoDAsync(t->gpu_n_array,
		t->modulus_in,
		t->array_sz * sizeof(uint64_t),
		t->stream))

	CUDA_TRY(cuEventRecord(t->start_event, t->stream))

	// copy init values into device memory
	CUDA_TRY(cuMemcpyHtoDAsync(t->gpu_rsq_array,
		t->rsq,
		t->array_sz * sizeof(uint64_t),
		t->stream))

	CUDA_TRY(cuMemcpyHtoDAsync(t->gpu_rho_array,
		t->rho,
		t->array_sz * sizeof(uint32_t),
		t->stream))

	CUDA_TRY(cuMemcpyHtoDAsync(t->gpu_one_array,
		t->one,
		t->array_sz * sizeof(uint64_t),
		t->stream))

	// run 64-bit ecm curves
	launch = t->launch + GPU_ECM_VEC;

	int orig_size = t->array_sz;

	while ((curve < t->curves_2lp) && (total_factors < orig_size)) {

		gpu_args[0].int32_arg = t->array_sz;
		gpu_args[1].ptr_arg = (void*)(t->gpu_n_array);		// n
		gpu_args[2].ptr_arg = (void*)(t->gpu_rho_array);	// rho
		gpu_args[3].ptr_arg = (void*)(t->gpu_one_array);	// unity
		gpu_args[4].ptr_arg = (void*)(t->gpu_rsq_array);	// Rsq
		gpu_args[5].ptr_arg = (void*)(t->gpu_sigma_array);	// sigma
		gpu_args[6].ptr_arg = (void*)(t->gpu_factor_array);	// f
		gpu_args[7].uint32_arg = 205;		// fixed B1, for now, need more stage 1 work.
		gpu_args[8].int32_arg = curve;		// variable curves

		gpu_launch_set(launch, gpu_args);

		// specify the x,y, and z dimensions of the thread blocks
		// that are created for the specified kernel function
		CUDA_TRY(cuFuncSetBlockShape(launch->kernel_func,
			threads_per_block, 1, 1))

		// launch the kernel with the size we just set and 
		// arguments configured by the gpu_launch_set command.
		CUDA_TRY(cuLaunchGridAsync(launch->kernel_func,
			num_blocks, 1, t->stream))

		// copy factors back to host
		CUDA_TRY(cuMemcpyDtoHAsync(t->factors, t->gpu_factor_array,
			t->array_sz * sizeof(uint64_t), t->stream))

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

		// copy new list of N to the gpu
		CUDA_TRY(cuMemcpyHtoDAsync(t->gpu_n_array,
			t->modulus_in,
			t->array_sz * sizeof(uint64_t),
			t->stream))

		CUDA_TRY(cuMemcpyHtoDAsync(t->gpu_rsq_array,
			t->rsq,
			t->array_sz * sizeof(uint64_t),
			t->stream))

		CUDA_TRY(cuMemcpyHtoDAsync(t->gpu_rho_array,
			t->rho,
			t->array_sz * sizeof(uint32_t),
			t->stream))

		CUDA_TRY(cuMemcpyHtoDAsync(t->gpu_one_array,
			t->one,
			t->array_sz * sizeof(uint64_t),
			t->stream))

		// new curves.  The gpu does 1 at a time.
		curve += 1;

	}

	CUDA_TRY(cuEventRecord(t->end_event, t->stream))
	CUDA_TRY(cuEventSynchronize(t->end_event))
	CUDA_TRY(cuEventElapsedTime(&elapsed_ms,
		t->start_event, t->end_event))

	printf("factored %d of %d inputs in %1.4f ms\n", 
		t->num_factors_2lp, orig_size, elapsed_ms);

	mpz_clear(rsq);
	mpz_clear(zn);

	/* we have to synchronize now */
	CUDA_TRY(cuStreamSynchronize(t->stream))

	return quit;
}

/*------------------------------------------------------------------------*/
// definitions for ECM types/functions that use cuda_xface
/*------------------------------------------------------------------------*/

device_ctx_t* gpu_device_init(int which_gpu)
{
	gpu_config_t gpu_config;
	gpu_info_t* gpu_info;
	size_t gpu_mem;

	device_ctx_t* d = (device_ctx_t*)xcalloc(1, sizeof(device_ctx_t));

	gpu_init(&gpu_config);
	if (gpu_config.num_gpu == 0) {
		printf("error: no CUDA-enabled GPUs found\n");
		exit(-1);
	}

	d->gpunum = which_gpu;
	d->gpu_info = gpu_info = (gpu_info_t*)xmalloc(sizeof(gpu_info_t));
	memcpy(gpu_info, gpu_config.info + which_gpu,
		sizeof(gpu_info_t));

	printf("using GPU %u (%s)\n", which_gpu, gpu_info->name);
	printf("selected card has CUDA arch %d.%d\n",
		gpu_info->compute_version_major,
		gpu_info->compute_version_minor);
	printf("more GPU info:\n");
	printf("\tmax_grid_size: %d x %d x %d\n", gpu_info->max_grid_size[0],
		gpu_info->max_grid_size[1], gpu_info->max_grid_size[2]);
	printf("\tglobal_mem_size: %zd\n", gpu_info->global_mem_size);
	printf("\tconstant_mem_size: %d\n", gpu_info->constant_mem_size);
	printf("\tmax_threads_per_block: %d\n", gpu_info->max_threads_per_block);
	printf("\tmax_thread_dim: %d x %d x %d\n", gpu_info->max_thread_dim[0],
		gpu_info->max_thread_dim[1], gpu_info->max_thread_dim[2]);
	printf("\tnum_compute_units: %d\n", gpu_info->num_compute_units);
	printf("\tregisters_per_block: %d\n", gpu_info->registers_per_block);
	printf("\tshared_mem_size: %d\n", gpu_info->shared_mem_size);
	printf("\twarp_size: %d\n", gpu_info->warp_size);

	return d;
}

void gpu_dev_free(device_ctx_t* d)
{
	free(d->gpu_info);
	free(d);
}

device_thread_ctx_t* gpu_ctx_init(device_ctx_t* d) {

	device_thread_ctx_t* t;

	t = (device_thread_ctx_t*)xcalloc(1, sizeof(device_thread_ctx_t));

	t->dev = d;

	/* every thread needs its own context; making all
	   threads share the same context causes problems
	   with the sort engine, because apparently it
	   changes the GPU cache size on the fly */
#if toolkit_version >= 13
		CUctxCreateParams* ctxCreateParams;

		CUDA_TRY(cuCtxCreate(&t->gpu_context,
			ctxCreateParams,
			CU_CTX_BLOCKING_SYNC,
			d->gpu_info->device_handle))
#else
		CUDA_TRY(cuCtxCreate(&t->gpu_context,
			CU_CTX_BLOCKING_SYNC,
			d->gpu_info->device_handle))
#endif

	/* load GPU kernels */
	char ptxfile[80];
	if (d->gpu_info->compute_version_major == 2) {
		strcpy(ptxfile, "cuda_ecm20.ptx");
	}
	else if (d->gpu_info->compute_version_major == 3) {
		if (d->gpu_info->compute_version_minor < 5)
			strcpy(ptxfile, "cuda_ecm30.ptx");
		else
			strcpy(ptxfile, "cuda_ecm35.ptx");
	}
	else if (d->gpu_info->compute_version_major >= 9) {
		strcpy(ptxfile, "cuda_ecm90.ptx");
	}
	else if (d->gpu_info->compute_version_major >= 8) {
		strcpy(ptxfile, "cuda_ecm80.ptx");
	}
	else if (d->gpu_info->compute_version_major >= 5) {
		strcpy(ptxfile, "cuda_ecm50.ptx");
	}
	else
	{
		printf("no ptx file found for gpu arch %d.%d\n",
			d->gpu_info->compute_version_major, d->gpu_info->compute_version_minor);
		exit(-1);
	}

	printf("loading kernel code from %s\n",	ptxfile);

	CUDA_TRY(cuModuleLoad(&t->gpu_module, ptxfile))

	printf("successfully loaded kernel code from %s\n",
		ptxfile);

	t->launch = (gpu_launch_t*)xmalloc(NUM_GPU_FUNCTIONS *
		sizeof(gpu_launch_t));

	printf("initializing kernels\n");
	int i;
	for (i = 0; i < NUM_GPU_FUNCTIONS; i++) {
		gpu_launch_t* launch = t->launch + i;

		gpu_launch_init(t->gpu_module, gpu_kernel_names[i],
			gpu_kernel_args + i, launch);
	}

	printf("creating stream\n");
	/* threads each send a stream of kernel calls */
	CUDA_TRY(cuStreamCreate(&t->stream, 0))

		printf("creating events\n");
	// for measuring elapsed time
	CUDA_TRY(cuEventCreate(&t->start_event, CU_EVENT_BLOCKING_SYNC))
	CUDA_TRY(cuEventCreate(&t->end_event, CU_EVENT_BLOCKING_SYNC))

	return t;
}

void gpu_ctx_free(device_thread_ctx_t* d)
{
	CUDA_TRY(cuEventDestroy(d->start_event))
	CUDA_TRY(cuEventDestroy(d->end_event))
	CUDA_TRY(cuStreamDestroy(d->stream))
	free(d->launch);
	CUDA_TRY(cuCtxDestroy(d->gpu_context))
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

/* external entry point */
uint32_t do_gpu_cofactorization(device_thread_ctx_t* t, uint64_t *factors, uint64_t *lcg, 
	uint64_t *inputs, uint32_t numinput, int b1_2lp_ovr, int b2_2lp_ovr, int curves_2lp_ovr)
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

	/* set up device arrays */
	CUDA_TRY(cuMemAlloc(&t->gpu_sigma_array, sizeof(uint32_t) * t->array_sz))
	CUDA_TRY(cuMemAlloc(&t->gpu_rsq_array, sizeof(uint64_t) * t->array_sz))
	CUDA_TRY(cuMemAlloc(&t->gpu_factor_array, sizeof(uint64_t) * t->array_sz))
	CUDA_TRY(cuMemAlloc(&t->gpu_one_array, sizeof(uint64_t) * t->array_sz))
	CUDA_TRY(cuMemAlloc(&t->gpu_n_array, sizeof(uint64_t) * t->array_sz))
	CUDA_TRY(cuMemAlloc(&t->gpu_rho_array, sizeof(uint32_t) * t->array_sz))

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

	CUDA_TRY(cuMemFree(t->gpu_factor_array))
	CUDA_TRY(cuMemFree(t->gpu_sigma_array))
	CUDA_TRY(cuMemFree(t->gpu_n_array))
	CUDA_TRY(cuMemFree(t->gpu_rho_array))
	CUDA_TRY(cuMemFree(t->gpu_rsq_array))
	CUDA_TRY(cuMemFree(t->gpu_one_array))

	return t->num_factors_2lp;
}



#endif