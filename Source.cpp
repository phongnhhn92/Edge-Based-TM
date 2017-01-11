#include <iostream>
#include <time.h>
#include <opencv2\core.hpp>
#include <opencv2\opencv.hpp>
#include <opencv2\highgui\highgui.hpp>
#include "GeoMatch.h"

using namespace cv;
using namespace std;

string type2str(int type) {
	string r;

	uchar depth = type & CV_MAT_DEPTH_MASK;
	uchar chans = 1 + (type >> CV_CN_SHIFT);

	switch (depth) {
	case CV_8U:  r = "8U"; break;
	case CV_8S:  r = "8S"; break;
	case CV_16U: r = "16U"; break;
	case CV_16S: r = "16S"; break;
	case CV_32S: r = "32S"; break;
	case CV_32F: r = "32F"; break;
	case CV_64F: r = "64F"; break;
	default:     r = "User"; break;
	}

	r += "C";
	r += (chans + '0');

	return r;
}

int main(int argc, char** argv)
{
	GeoMatch GM;				// object to implent geometric matching	
	int lowThreshold = 10;		//deafult value
	int highThreashold = 100;	//deafult value

	double minScore = 0.7;		//deafult value
	double greediness = 0.8;		//deafult value

	double total_time = 0;
	double score = 0;
	Point result;

	//Load Template image 
	Mat templateImage = imread("template_2.bmp", -1);

	//Check Mat type
	string ty = type2str(templateImage.type());
	printf("Matrix: %s %dx%d \n", ty.c_str(), templateImage.cols, templateImage.rows);

	//Load Search Image
	Mat searchImage = imread("roi_2.bmp", -1);


	lowThreshold = 10;
	highThreashold = 100;
	minScore = 0.7;
	greediness = 0.5;

	Size templateSize(templateImage.size().width, templateImage.size().height);
	Mat grayTemplateImg(templateSize, IPL_DEPTH_8U, 1);

	// Convert color image to gray image.
	if (templateImage.channels() == 3)
	{
		cvtColor(templateImage, grayTemplateImg, CV_RGB2GRAY);		
	}
	else
	{
		templateImage.copyTo(grayTemplateImg);
	}
	cout << "\n Edge Based Template Matching Program\n";
	cout << " ------------------------------------\n";

	if (!GM.CreateGeoMatchModel(grayTemplateImg, lowThreshold, highThreashold))
	{
		cout << "ERROR: could not create model...";
		return 0;
	}

	GM.DrawContours(templateImage, CV_RGB(255, 0, 0), 1);
	cout << " Shape model created.." << "with  Low Threshold = " << lowThreshold << " High Threshold = " << highThreashold << endl;
	Size searchSize(searchImage.size().width, searchImage.size().height);
	Mat graySearchImg(searchSize, IPL_DEPTH_8U, 1);

	// Convert color image to gray image. 
	if (searchImage.channels() == 3)
		cvtColor(searchImage, graySearchImg, CV_RGB2GRAY);
	else
	{		
		searchImage.copyTo(graySearchImg);		
	}
	cout << " Finding Shape Model.." << " Minumum Score = " << minScore << " Greediness = " << greediness << "\n\n";
	cout << " ------------------------------------\n";
	clock_t start_time1 = clock();
	score = GM.FindGeoMatchModel(graySearchImg, minScore, greediness, &result);
	clock_t finish_time1 = clock();
	total_time = (double)(finish_time1 - start_time1) / CLOCKS_PER_SEC;

	if (score>minScore) // if score is atleast 0.4
	{
		cout << " Found at [" << result.x << ", " << result.y << "]\n Score = " << score << "\n Searching Time = " << total_time * 1000 << "ms";
		GM.DrawContours(searchImage, result, CV_RGB(0, 255, 0), 1);
	}
	else
		cout << " Object Not found";

	cout << "\n ------------------------------------\n\n";
	cout << "\n Press any key to exit!";

	//Display result
	namedWindow("Template", CV_WINDOW_AUTOSIZE); 
	imshow("Template", templateImage);
	namedWindow("Search Image", CV_WINDOW_AUTOSIZE);
	imshow("Search Image", searchImage);
	Mat temp(templateImage);
	Mat search(searchImage);
	int centerx = result.x;
	int centery = result.y;
	circle(search, Point(centerx, centery), 40, Scalar(255, 0, 0), 1, 8, 0);
	imwrite("a.jpg", temp);
	imwrite("b.jpg", search);
	// wait for both windows to be closed before releasing images
	cvWaitKey(0);
	return 1;
}


