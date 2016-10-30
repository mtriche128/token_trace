/**************************************************************************//**
* @file   ocl_base.cpp
* @brief  This source file implements the OpenCL interface base class.
* @author Matthew Triche
****************************************************************************/

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include <string>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <stddef.h>

#include "ocl_base.h"

using namespace std;

/* ------------------------------------------------------------------------- *
 * Define Constants                                                          *
 * ------------------------------------------------------------------------- */

// Uncomment to get debugging output.
#define OCLBASE_DEBUG

/* ------------------------------------------------------------------------- *
 * Declare Internal Functions                                                *
 * ------------------------------------------------------------------------- */

static char *read_kernel_source(const char *sz_fname);

/* ------------------------------------------------------------------------- *
 * Define Internal Functions                                                 *
 * ------------------------------------------------------------------------- */

/**
 * @brief Read a source file.
 *
 * The string returned by the function contains the entire contents of the
 * source file and is dynamically allocated. Garbage collection for this data
 * will need to be handled manually.
 *  
 * @param sz_fname The source file name.
 * 
 * @return A pointer to the null terminated string which contains the source.
 */

static char *read_kernel_source(const char *sz_fname)
{
	int fsize;
	char *sz_source;
	FILE *kfile = fopen(sz_fname, "rb");
	
	if(!kfile)
	{
		printf("ERROR: Failed to load the kernel source file!\r\n");
		exit(1);
	}
	
	// read the length of the source file
	fseek(kfile, 0, SEEK_END);
	fsize = ftell(kfile);
	rewind(kfile); 
	
	sz_source = (char*)malloc(fsize+1); // allocate memory for the source string
	*(sz_source+fsize) = 0; // null terminate source string
	
	
	fread(sz_source, 1, fsize, kfile); // read the file's contents
	fclose(kfile);
	
	return sz_source;
}

/* ------------------------------------------------------------------------- *
 * Define Methods                                                            *
 * ------------------------------------------------------------------------- */

/**
 * @brief This constructor shall read and compile a target OCL source file.
 * 
 * @param path Path to the target OCL source file.
 */

OCL_Base::OCL_Base(string path)
{
	cl_int err;
	
	sz_oclsrc = read_kernel_source(path.c_str());
	
	/* ------ Initialize OpenCL Resources ------ */
	
	// Bind to platform
	err = clGetPlatformIDs(1, &cpPlatform, NULL);
	
	// Get ID for the device
	err = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_CPU, 1, &device_id, NULL);
	
	#ifdef OCLBASE_DEBUG
	printf("creating context...");
	#endif
	
	// Create a context  
	context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
	
	#ifdef OCLBASE_DEBUG
	printf("done\r\n");
	printf("creating command queue...");
	#endif
	
	// Create a command queue 
	queue = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &err);
	
	#ifdef OCLBASE_DEBUG
	printf("done\r\n");
	printf("creating OpenCL program from kernel source...");
	#endif
	
	// Create the compute program from the source buffer
	program = clCreateProgramWithSource(context, 1, (const char**)(&sz_oclsrc), NULL, &err);
	
	if(err != CL_SUCCESS)
	{
		printf("failed: code = %i\r\n", err);
		exit(1);
	}
	
	else
	{
		#ifdef OCLBASE_DEBUG
		printf("done\r\n");
		#endif
	}
	
	#ifdef OCLBASE_DEBUG
	printf("building OpenCV program...");
	#endif
	
	// Build the program executable 
	err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	
	if(err != CL_SUCCESS)
	{
		printf("failed: code = %i\r\n", err);
		
		size_t len;
		char *logstr;
		clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, NULL, NULL, &len);
		
		logstr = new char[len];
		clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, len, logstr, NULL);
		
		printf("--------------------------------------------------\r\n"); 
		printf("[OpenCL Build Log]\r\n");
		printf("%s", logstr);
		printf("\r\n--------------------------------------------------\r\n");
		
		delete [] logstr;
		exit(1);
		
	}
	
	else
	{
		#ifdef OCLBASE_DEBUG
		printf("done\r\n");
		#endif
	}
}

/**
 * @brief destructor
 */

OCL_Base::~OCL_Base()
{
	printf("~OCL_Base(): start\r\n");
	
	free(sz_oclsrc);
	
	printf("~OCL_Base(): end\r\n");
}

/**
 * @brief Upload a data buffer using an OpenCL buffer object.
 * 
 * @param[in]  buff_obj The buffer object.
 * @param[in]  data     Pointer to the data buffer.
 * @param[in]  size     The data buffer's size in bytes.
 * @param[out] event    Event object.
 * 
 * @return True of the upload succeeded. False otherise.
 */

bool OCL_Base::OCL_UploadBuffer(cl_mem &buff_obj,
                                void *data, 
                                size_t size, 
                                cl_event *event)
{
	cl_int err;
	
	#ifdef OCLBASE_DEBUG
	printf("uploading data to external device...");
	#endif
	
	err = clEnqueueWriteBuffer(queue,
	                           buff_obj, 
	                           CL_TRUE, 
	                           0,
	                           size, 
	                           data,
	                           0, 
	                           NULL, 
	                           event);
	
	if(err != CL_SUCCESS)
	{
		#ifdef OCLBASE_DEBUG
		printf("enqueing buffers failed: code = %i\r\n", err);
		#endif
		
		return false;
	}
	
	else
	{
		#ifdef OCLBASE_DEBUG
		printf("done\r\n");
		#endif
	}
	
	return true;
}

/**
 * @brief Download a data buffer using an OpenCL buffer object.
 * 
 * @param[in]  buff_obj The buffer object.
 * @param[out] data     Pointer to the data buffer.
 * @param[in]  size     The data buffer's size in bytes.
 * @param[out] event    Event object.
 * 
 * @return True of the download succeeded. False otherise.
 */

bool OCL_Base::OCL_DownloadBuffer(cl_mem &buff_obj,
                                  void *data, 
                                  size_t size, 
                                  cl_event *event)
{
	cl_int err;
	
	#ifdef OCLBASE_DEBUG
	printf("downloading data from external device...");
	#endif
	
	err = clEnqueueReadBuffer(queue,
	                          buff_obj, 
	                          CL_TRUE, 
	                          0, 
	                          size,
	                          data,
	                          0, 
	                          NULL,
	                          event);
	
	if(err != CL_SUCCESS)
	{
		#ifdef OCLBASE_DEBUG
		printf("enqueing buffers failed: code = %i\r\n", err);
		#endif
		
		return false;
	}
	
	else
	{
		#ifdef OCLBASE_DEBUG
		printf("done\r\n");
		#endif
	}
	
	return true;
}