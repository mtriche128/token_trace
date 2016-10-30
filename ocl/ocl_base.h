/**************************************************************************//**
 * @file   ocl_base.h
 * @brief  Header file for the OpenCL interface base class.
 * @author Matthew Triche
 ****************************************************************************/

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
 
#include <string>

using namespace std;

#ifndef OCL_BASE_H_
#define OCL_BASE_H_

/* ------------------------------------------------------------------------- *
 * Define External Types                                                     *
 * ------------------------------------------------------------------------- */

/**
 * @brief Base class for an OpenCL interface.
 */

class OCL_Base
{
public:
	OCL_Base(string path);
	~OCL_Base();
	
protected:
	bool OCL_UploadBuffer(cl_mem &buff_obj, void *data, size_t size, cl_event *event);
	bool OCL_DownloadBuffer(cl_mem &buff_obj, void *data, size_t size, cl_event *event);
	
	cl_platform_id cpPlatform;        // OpenCL platform
	cl_device_id device_id;           // device ID
	cl_context context;               // context
	cl_command_queue queue;           // command queue
	cl_program program;               // program
	
	
private:
	char *sz_oclsrc;                  // contains the OCL source 
};

#endif