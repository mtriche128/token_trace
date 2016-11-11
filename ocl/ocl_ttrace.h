/**************************************************************************//**
 * @file   ocl_ttrace.h
 * @brief  Header file for the token-trace algorithm.
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
	
	void Trace(const Mat &img_in, Mat &ctbl, TimeProfile &tp);
	
private:
	cl_mem    cl_m_binimg;  // buffer for binary image (U8)
	cl_mem    cl_m_tokens;  // buffer for passing token data (U8)
	cl_mem    cl_m_cnt;     // buffer for the contour table counter (uint32)
	cl_mem    cl_m_ctbl;    // buffer for the contour table (uint32[][])
	cl_kernel cl_k_ttrace;  // handle for the token-trace kernel
};
	
#endif