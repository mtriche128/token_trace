/**************************************************************************//**
* @file   ocl_ttrace.cpp
* @brief  This source file implements the token-trace algorithm.
* @author Matthew Triche
*****************************************************************************/

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include <opencv2/opencv.hpp>
#include <string>
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stddef.h>

#include "ocl_ttrace.h"
#include "ocl_base.h"

using namespace std;
using namespace cv;

/* ------------------------------------------------------------------------- *
 * Define Constants                                                          *
 * ------------------------------------------------------------------------- */

#define LOCAL_SIZE    (64)

/* ------------------------------------------------------------------------- *
 * Define Types                                                          *
 * ------------------------------------------------------------------------- */

/**
 * @brief This struct defines a single entry in a token buffer.
 */

typedef struct __attribute__((__packed__)) TOKEN_ENTRY
{
	uint8_t  state; // contains flags related to contour type
	uint8_t  hist;  // pass/hold history for generating chain-codes
	uint32_t orow;  // contour's origin row coordinate
	uint32_t ocol;  // contour's origin column coordinate
	uint32_t id;    // contour identifier
	uint32_t cx;    // current index in the contour table
} token_t;

/* ------------------------------------------------------------------------- *
 * Define Methods                                                            *
 * ------------------------------------------------------------------------- */

/**
 * @brief default consturctor
 */

TimeProfile::TimeProfile()
{
	ul_time = 0.0;
	k_time  = 0.0;
	dl_time = 0.0;
}

/**
 * @brief event consturctor
 * 
 * This constructor derives a time profile from OCL events.
 *
 * @param ul_event upload event
 * @param k_event  kernel execution event
 * @param dl_event download event 
 */

TimeProfile::TimeProfile(cl_event *ul_event, 
                         cl_event *k_event,
                         cl_event *dl_event)
{
	cl_ulong start, stop;
	
	// if an upload event is specified
	if(ul_event)
	{
		clGetEventProfilingInfo(*ul_event, 
		                        CL_PROFILING_COMMAND_START, 
		                        sizeof(start), 
		                        &start, NULL);
		
		clGetEventProfilingInfo(*ul_event, 
		                        CL_PROFILING_COMMAND_END, 
		                        sizeof(stop), 
		                        &stop, NULL);
		ul_time = (double)(stop - start) / (double)1e9; 
	}
	
	// if a kernel execution event is specified
	if(k_event)
	{
		clGetEventProfilingInfo(*k_event, 
		                        CL_PROFILING_COMMAND_START, 
		                        sizeof(start), 
		                        &start, NULL);
		
		clGetEventProfilingInfo(*k_event, 
		                        CL_PROFILING_COMMAND_END, 
		                        sizeof(stop), 
		                        &stop, NULL);
		k_time = (double)(stop - start) / (double)1e9; 
	}
	
	// if a download event is specified
	if(dl_event)
	{
		clGetEventProfilingInfo(*dl_event, 
		                        CL_PROFILING_COMMAND_START, 
		                        sizeof(start), 
		                        &start, NULL);
		
		clGetEventProfilingInfo(*dl_event, 
		                        CL_PROFILING_COMMAND_END, 
		                        sizeof(stop), 
		                        &stop, NULL);
		dl_time = (double)(stop - start) / (double)1e9; 
	}
	
}

/**
 * @brief assignment constructor
 * 
 * @param tp Pointer to target time profile.
 */

TimeProfile::TimeProfile(TimeProfile *tp)
{
	ul_time = tp->ul_time;
	k_time = tp->k_time;
	dl_time = tp->dl_time;
}

/**
 * @brief Add time profiles.
 * 
 * @brief Target time profile.
 * 
 * @return Sumed time profile.
 */

TimeProfile TimeProfile::operator+(TimeProfile &tp)
{
	TimeProfile sum;
	
	sum.ul_time = ul_time + tp.ul_time;
	sum.k_time  = k_time + tp.k_time;
	sum.dl_time = dl_time + tp.dl_time;
	
	return sum;
}

/**
 * @brief consturctor
 * 
 * @param path       Path to the OCL source file.
 * @param img_width  Image width.
 * @param img_height Image height.
 */

OCL_TTrace::OCL_TTrace(string path, 
                       uint32_t img_width,
                       uint32_t img_height,
                       uint32_t ctbl_width,
                       uint32_t ctbl_height) : OCL_Base(path)
{
	cl_int err;
	
	cl_m_binimg = clCreateBuffer(context,
	                             CL_MEM_READ_WRITE,
	                             img_height*img_width, 
	                             NULL, &err);
	assert(err == CL_SUCCESS); // failed to create buffer object

	cl_m_dbgimg = clCreateBuffer(context,
	                             CL_MEM_READ_WRITE,
	                             3*img_height*img_width, 
	                             NULL, &err);
	assert(err == CL_SUCCESS); // failed to create buffer object
	
	cl_m_tokens = clCreateBuffer(context,
	                             CL_MEM_READ_WRITE,
	                             img_height*sizeof(token_t), 
	                             NULL, &err);
	assert(err == CL_SUCCESS); // failed to create buffer object
	
	cl_m_cnt = clCreateBuffer(context,
	                          CL_MEM_READ_WRITE,
	                          sizeof(uint32_t), 
	                          NULL, &err);
	assert(err == CL_SUCCESS); // failed to create buffer object
	
	cl_m_ctbl = clCreateBuffer(context,
	                           CL_MEM_READ_WRITE,
	                           sizeof(uint32_t)*ctbl_width*ctbl_height, 
	                           NULL, &err);
	assert(err == CL_SUCCESS); // failed to create buffer object

	cl_k_ttrace = clCreateKernel(program, "TOKEN_TRACE", &err);
	assert(err == CL_SUCCESS); // failed to create kernel
};

/**
 * @brief destructor
 */

OCL_TTrace::~OCL_TTrace()
{
	clReleaseMemObject(cl_m_binimg);
	clReleaseMemObject(cl_m_dbgimg);
}

void OCL_TTrace::Trace(const Mat &img_in, Mat &img_out, Mat &ctbl, TimeProfile &tp)
{
	cl_int err;
	cl_event ul_event, k_event, dl_event;
	
	uint32_t img_rows  = img_in.rows;
	uint32_t img_cols  = img_in.cols;
	uint32_t ctbl_rows = ctbl.rows;
	uint32_t ctbl_cols = ctbl.cols;
	uint32_t cnt_init  = 0; // the initial counter value
	
	size_t gsize = img_rows; // group size
	gsize += LOCAL_SIZE - (gsize%LOCAL_SIZE);
	
	size_t lsize = LOCAL_SIZE; // local size
	
	// upload the image
	OCL_UploadBuffer(cl_m_binimg, img_in.data, img_in.rows*img_in.cols, &ul_event);
	
	// initialize the counter for the contour table
	OCL_UploadBuffer(cl_m_cnt, &cnt_init, sizeof(uint32_t), &ul_event);
	
	err  = clSetKernelArg(cl_k_ttrace, 0, sizeof(cl_mem),   &cl_m_binimg);
	err |= clSetKernelArg(cl_k_ttrace, 1, sizeof(cl_mem),   &cl_m_dbgimg);
	err |= clSetKernelArg(cl_k_ttrace, 2, sizeof(cl_mem),   &cl_m_tokens);
	err |= clSetKernelArg(cl_k_ttrace, 3, sizeof(uint32_t), &img_rows);
	err |= clSetKernelArg(cl_k_ttrace, 4, sizeof(uint32_t), &img_cols);
	err |= clSetKernelArg(cl_k_ttrace, 5, sizeof(cl_mem),   &cl_m_cnt);
	err |= clSetKernelArg(cl_k_ttrace, 6, sizeof(cl_mem),   &cl_m_ctbl);
	err |= clSetKernelArg(cl_k_ttrace, 7, sizeof(uint32_t), &ctbl_rows);
	err |= clSetKernelArg(cl_k_ttrace, 8, sizeof(uint32_t), &ctbl_cols);
	assert(err == CL_SUCCESS); // failed to set arguments

	err = clEnqueueNDRangeKernel(queue, 
                                   cl_k_ttrace, 
	                             1, 
	                             NULL, 
	                             (const size_t*)&gsize, 
	                             (const size_t*)&lsize,
	                             0, 
	                             NULL,
	                             &k_event); 
	assert(err == CL_SUCCESS); // failed to execute kernel

	clFinish(queue); // let the kernel finish execution
	
	// download the debug image
	OCL_DownloadBuffer(cl_m_dbgimg, img_out.data, 3*img_in.rows*img_in.cols, NULL);
	
	// download the contour table
	OCL_DownloadBuffer(cl_m_ctbl, ctbl.data, ctbl.size, &dl_event);
	
	tp = TimeProfile(&ul_event, &k_event, &dl_event);
}
