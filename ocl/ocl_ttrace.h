/**************************************************************************//**
 * @file   ocl_ttrace.h
 * @brief  Header file for the token-trace algorithm.
 * @author Matthew Triche
 *****************************************************************************/

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include <opencv2/opencv.hpp>
#include <string>

#include "ocl_base.h"

using namespace std;
using namespace cv;

#ifndef OCL_TTRACE_H_
#define OCL_TTRACE_H_

class TimeProfile
{
public:
	TimeProfile();
	TimeProfile(cl_event *ul_event, 
	            cl_event *k_event,
                  cl_event *dl_event);
	TimeProfile(TimeProfile *tp);
	TimeProfile operator+(TimeProfile &tp);
	
	double ul_time; // units in seconds
	double k_time;  // units in seconds
	double dl_time; // units in seconds
};

/**
 * @brief The token-trace OCL factory.
 */

class OCL_TTrace : public OCL_Base
{
public:
	OCL_TTrace(string path, uint32_t img_width, uint32_t img_height, 
	                        uint32_t ctbl_width, uint32_t ctbl_height);
	~OCL_TTrace();
	
	void Trace(const Mat &img_in, Mat &img_out, Mat &ctbl, TimeProfile &tp);
	
private:
	cl_mem    cl_m_binimg;  // buffer for binary image (U8)
	cl_mem    cl_m_dbgimg;  // buffer for debug image (U8C3)
	cl_mem    cl_m_tokens;  // buffer for passing token data (uint8)
	cl_mem    cl_m_cnt;     // buffer for the contour table counter (uint32)
	cl_mem    cl_m_ctbl;    // buffer for the contour table (uint32[][])
	cl_kernel cl_k_ttrace;  // handle for the token-trace kernel
};
	
#endif