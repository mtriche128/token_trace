/**************************************************************************//**
 * @file   ocl_base.h
 * @brief  Header file for the OpenCL interface base class.
 * @author Matthew Triche
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights 
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell 
 * copies of the Software, and to permit persons to whom the Software is 
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
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