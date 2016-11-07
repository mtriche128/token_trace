#include <iostream>
#include <vector>
#include <queue>
#include <opencv2/opencv.hpp>
#include <time.h>
#include <stdio.h>
#include <math.h>

#include "ocl/ocl_ttrace.h"

using namespace std;
using namespace cv;

void DrawContourTable(Mat &img, Mat &ctbl);

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
	
	OCL_TTrace contour("kernel.cl", 100, 100, 50, 50);
	
	// allocate space for the local copy of the contour table
	Mat ctbl = Mat::zeros(31, 31, CV_32S);
	
	/* ------ Run Test ------ */
	
	contour.Trace(bin_img, ctbl, tp);
	
	/* ------ Output Results ------ */
	/*
	for(int row = 0; row < 25; row++)
	{
		printf("%2i : ", row);
		
		for(int col = 0; col < 25; col++)
		{
			printf("%2i ", ctbl.at<uint32_t>(row,col));
		}
		
		printf("\r\n");
	}
	*/
	cout << "---------------------------------" << endl;
	
	
	DrawContourTable(dbg_img, ctbl);
	
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
