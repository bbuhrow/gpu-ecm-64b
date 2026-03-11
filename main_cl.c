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

// ============================================================================
// test driver for gpu and cpu NFS 3LP relation cofactorization methods
// ============================================================================
#include <stdio.h>
#include <stdlib.h>
#include "gpu_cofactorization_cl.h"
#include <inttypes.h>
#include <string.h>
#include <immintrin.h>
#include "gmp.h"
#include "ytools.h"
#include "cmdOptions.h"
#include <math.h>

// ============================================================================
// precision time
// ============================================================================


#if defined(WIN32) || defined(_WIN64) 
#define WIN32_LEAN_AND_MEAN

#if defined(__clang__)
#include <time.h>
#endif
#include <windows.h>
#include <process.h>
#include <winsock.h>

#else
#include <sys/time.h>	//for gettimeofday using gcc
#include <unistd.h>
#endif

#if defined(_MSC_VER) || defined(_MSC_EXTENSIONS)
#define DELTA_EPOCH_IN_MICROSECS  11644473600000000Ui64
#else
#define DELTA_EPOCH_IN_MICROSECS  11644473600000000ULL
#endif

#if defined(__MINGW32__)
#include <sys/time.h>
#endif

#ifdef _MSC_VER
struct timezone
{
    int  tz_minuteswest; /* minutes W of Greenwich */
    int  tz_dsttime;     /* type of dst correction */
};
#endif


double _difftime(struct timeval* start, struct timeval* end);


#if defined(_MSC_VER)
int gettimeofday(struct timeval* tv, struct timezone* tz);
#endif


#if defined(_MSC_VER)

#if 0 // defined(__clang__)
int gettimeofday(struct timeval* tv, struct timezone* tz)
{
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);

    //printf("timespec_get returned sec = %lu, nsec = %lu\n", ts.tv_sec, ts.tv_nsec);

    tv->tv_sec = ts.tv_sec;
    tv->tv_usec = ts.tv_nsec / 1000;

    return 0;
}
#else
int gettimeofday(struct timeval* tv, struct timezone* tz)
{
    FILETIME ft;
    unsigned __int64 tmpres = 0;
    static int tzflag;

    if (NULL != tv)
    {
        GetSystemTimeAsFileTime(&ft);

        tmpres |= ft.dwHighDateTime;
        tmpres <<= 32;
        tmpres |= ft.dwLowDateTime;

        /*converting file time to unix epoch*/
        tmpres /= 10;  /*convert into microseconds*/
        tmpres -= DELTA_EPOCH_IN_MICROSECS;
        tv->tv_sec = (long)(tmpres / 1000000UL);
        tv->tv_usec = (long)(tmpres % 1000000UL);
    }

    return 0;
}
#endif
#endif

double _difftime(struct timeval* start, struct timeval* end)
{
    double secs;
    double usecs;

    if (start->tv_sec == end->tv_sec) {
        secs = 0;
        usecs = end->tv_usec - start->tv_usec;
    }
    else {
        usecs = 1000000 - start->tv_usec;
        secs = end->tv_sec - (start->tv_sec + 1);
        usecs += end->tv_usec;
        if (usecs >= 1000000) {
            usecs -= 1000000;
            secs += 1;
        }
    }

    return secs + usecs / 1000000.;
}

uint32_t process_batch(char* infile, char* outfile, int vflag,
	int b1, int b2, int curves)
{
	char buf[1024], str1[1024];
	struct timeval start;
	struct timeval stop;
	double ttime;
	uint64_t lcg_state = 0xbaddecafbaddecafull;
	int i;
	uint32_t line = 0;
	uint64_t allocinput = 1024;
	uint64_t* input = (uint64_t*)xmalloc(allocinput * sizeof(uint64_t));;
	uint64_t numinput = 0;

	if (vflag > 0)
	{
		printf("reading input file %s...\n", infile);
	}

	FILE* fid = fopen(infile, "r");
	if (fid == NULL)
	{
		printf("could not open %s to read\n", infile);
		exit(0);
	}

	gettimeofday(&start, NULL);

	while (~feof(fid))
	{
		uint64_t n;

		line++;
		char* ptr = fgets(buf, 1024, fid);
		if (ptr == NULL)
			break;

		strcpy(str1, buf);

		sscanf(buf, "%"PRIu64"", &n);

		if (numinput >= allocinput)
		{
			allocinput *= 2;
			input = (uint64_t*)xrealloc(input, allocinput * sizeof(uint64_t));
		}

		input[numinput++] = n;
	}
	fclose(fid);

	gettimeofday(&stop, NULL);
	ttime = ytools_difftime(&start, &stop);

	{
#ifdef HAVE_CUDA_BATCH_FACTOR
		if (vflag >= 0)
		{
			printf("file parsing took %1.2f sec, found %u inputs. "
				"now running gpu ecm...\n",
				ttime, numinput);
		}

		gettimeofday(&start, NULL);

		int gpu_num = 0;
		device_ctx_t* gpu_dev_ctx = gpu_device_init(gpu_num);

		// we must create the thread context here... the cuda context
		// init method must fold in the current thread info. 
		printf("creating gpu-ecm context\n");
		device_thread_ctx_t* gpu_cofactor_ctx =
			gpu_ctx_init(gpu_dev_ctx);

		gpu_cofactor_ctx->verbose = vflag;
		gpu_cofactor_ctx->stop_nofactor = 100;

		uint64_t* factors = (uint64_t*)xmalloc(numinput * sizeof(uint64_t));

		uint32_t num_success = do_gpu_cofactorization(gpu_cofactor_ctx, factors, &lcg_state,
			input, numinput, b1, b2, curves);

		// perhaps we can make the context persistent after we create it 
		// once in the thread?
		gpu_ctx_free(gpu_cofactor_ctx);
		gpu_dev_free(gpu_dev_ctx);

		gettimeofday(&stop, NULL);

		ttime = ytools_difftime(&start, &stop);

		// write the processed inputs.  
		printf("verifying factors and writing output file... ");

		char outname[80];
		sprintf(outname, "%s.out", infile);
		FILE* fout = fopen(outname, "w");
		int problems = 0;
		if (fout != NULL)
		{
			for (i = 0; i < numinput; i++)
			{
				if (input[i] % factors[i] == 0)
				{
					fprintf(fout, "%"PRIu64",%"PRIu64"\n", input[i], factors[i]);
				}
				else
				{
					problems++;
				}
			}
			fclose(fout);
			printf("done\n");
			if (problems)
				printf("problems verifying %d factors\n", problems);
		}
		else
		{
			printf("failed to create outputfile %s\n", outname);
		}

#endif
	}

	return 0;
}

int main(int argc, char **argv) {
	char fname[80];
	options_t* options = initOpt();

	processOpts(argc, argv, options);

	strcpy(fname, options->file);

	process_batch(options->file, "", 1,
		options->b1_2lp, options->b2_2lp, options->curves_2lp);

	return 0;
}
