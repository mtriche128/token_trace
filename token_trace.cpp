/**************************************************************************//**
 * @file   token_trace.cpp
 * @brief  This source file contains the main function.
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
 
#include <iostream>
#include <vector>
#include <queue>
#include <string>
#include <opencv2/opencv.hpp>
#include <time.h>
#include <stdio.h>
#include <math.h>

#include "ocl/ocl_ttrace.h"

using namespace std;
using namespace cv;

void DrawContourTable(Mat &img, Mat &ctbl);

int main(int argc, char **argv)
{
	cout << "===== Token Trace =====" << endl;
	
	/* ------ Handle Arguments ------ */
	
	if(argc == 1)
	{
		cout << "Error: Missing image path command-line argument." << endl;
		exit(1);
	}
	
	else if(argc > 2)
	{
		cout << "Error: Too many command-line arguments given." << endl;
		exit(1);
	}
	
	if(!strcmp(argv[1], "--help"))
	{
		cout << "Usage: token_trace <IMAGE_PATH>" << endl;
		exit(0);
	}

	/* ------ Initialize Data and Objects ------ */
	
	TimeProfile tp;
	
	Mat bin_img;
	Mat dbg_img = imread(argv[1]);

	if(!dbg_img.data)
	{
		cout << "Error: Unable to read '" << argv[1] << "'." << endl;
		exit(1);
	}
	
	cvtColor(dbg_img,bin_img,CV_BGR2GRAY);
	bitwise_not(bin_img,bin_img);
	
	OCL_TTrace contour("kernel.cl", 100, 100, 50, 50);
	
	// allocate space for the local copy of the contour table
	Mat ctbl = Mat::zeros(31, 31, CV_32S);
	
	/* ------ Run Test ------ */
	
	contour.Trace(bin_img, ctbl, tp);
	
	/* ------ Output Results ------ */

	cout << "---------------------------------" << endl;

	DrawContourTable(dbg_img, ctbl);
	
	/* ------ Output Results ------ */
	
	cout << "upload time   = " << tp.ul_time * 1e6 << " us" << endl;
	cout << "kernel time   = " << tp.k_time * 1e6 << " us" << endl;
	cout << "download time = " << tp.dl_time * 1e6 << " us" << endl;

	Mat output;
	resize(dbg_img,output,Size(20*dbg_img.cols,20*dbg_img.rows),0,0,INTER_NEAREST);
	imshow("output", output);
	
	while(1)
	{
		if(waitKey(1) >= 0) break;
	}

	return 0;
}

void DrawContourTable(Mat &img, Mat &ctbl)
{
	for(int row = 0; row < ctbl.rows; row++)
	{
		cout << row << " : ";
		uint32_t size = ctbl.at<uint32_t>(row,0);
		for(int col = 1; col < size; col+=2)
		{
			uint32_t irow = ctbl.at<uint32_t>(row, col);
			uint32_t icol = ctbl.at<uint32_t>(row, col+1);
			cout << "(" << irow << "," << icol << ") ";
			img.at<Vec3b>(Point(icol,irow)) = Vec3b(0,0,255);
		}
		
		cout << endl;
	}
}
