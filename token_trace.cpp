#include <iostream>
#include <vector>
#include <queue>
#include <opencv2/opencv.hpp>
#include <time.h>
#include <math.h>

#include "ocl/ocl_ttrace.h"

using namespace std;
using namespace cv;


int main(void)
{
	cout << "===== Token Trace =====" << endl;

	/* ------ Initialize Data and Objects ------ */
	
	TimeProfile tp;
	
	Mat bin_img;
	Mat dbg_img = imread("chia_demo.bmp");
	//Mat dbg_img = imread("square.bmp");
	//Mat dbg_img = imread("test2.bmp");
	
	if(!dbg_img.data)
	{
		cout << "Error: Unable to read image." << endl;
		exit(1);
	}
	
	cvtColor(dbg_img,bin_img,CV_BGR2GRAY);
	bitwise_not(bin_img,bin_img);
	
	OCL_TTrace contour("kernel.cl", 100, 100);
	
	/* ------ Run Test ------ */
	
	contour.Trace(bin_img, dbg_img, tp);
	
	/* ------ Output Results ------ */
	
	cout << "upload time   = " << tp.ul_time * 1e6 << " us" << endl;
	cout << "kernel time   = " << tp.k_time * 1e6 << " us" << endl;
	cout << "download time = " << tp.dl_time * 1e6 << " us" << endl;
	

	Mat output;
	resize(dbg_img,output,Size(20*dbg_img.cols,20*dbg_img.rows),0,0,INTER_NEAREST);
	imshow("output", output);
	//imshow("output", dbg_img);
	
	while(1)
	{
		if(waitKey(1) >= 0) break;
	}

	return 0;
}
