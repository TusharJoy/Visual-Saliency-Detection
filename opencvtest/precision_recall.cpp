#include <iostream>
#include<set>
#include <vector>
#include <math.h>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include <string>
#include <algorithm>

#include "Superpixels.h"

using namespace std;
using namespace cv;
vector<String> fn;
vector<String>fnGT;
int main()
{
	freopen("Precision and Recall for AC.txt", "w", stdout);

	//Mat input, GT;
	//input = imread("MSRAdataset1/1.jpg");
	//GT = imread("MSRAdataset1/137.png");

	String inputpath= "OthersalienyCut/*_AC.png";
	String GTpath= "MSRAdataset/*.png";

	glob(inputpath, fn, true); // recursive, if you want
	glob(GTpath, fnGT, true);
	for (size_t i = 0; i < fn.size(); i++)
	{
		Mat input = imread(fn[i]);
		Mat GT= imread(fnGT[i]);



		double precision = 0, recall = 0, divider = 0, divider2 = 0;
		for (int i = 0; i <= 255; i++)
		{
			Mat binaryMat(input.size(), input.type());
			threshold(input, binaryMat, i, 255, THRESH_BINARY);

			int tp = 0, tn = 0, fp = 0, fn = 0;

			for (int j = 0; j < (int)input.rows; ++j)
			{
				for (int k = 0; k < (int)input.cols; k++)
				{
					int pixelValue = (int)binaryMat.at<uchar>(j, k);
					int pixelValuegt = (int)GT.at<uchar>(j, k);

					if (pixelValue == pixelValuegt)
					{
						if (pixelValue == 255)
							tp++;
						else
							tn++;
					}

					else
					{
						if (pixelValue == 255)
							fp++;
						else
							fn++;
					}

				}
			}
			//cout <<"Thresold value = "<<i<<"   "<< tp << " " << tn << " " <<fp << " " << fn << endl;
			if (!(tp == 0 && fp == 0))
			{
				precision += (tp*1.0) / ((tp + fp)*1.0);
				divider++;

			}
			if (!(tp == 0 && fn == 0))
			{
				recall += ((tp)*1.0 / (tp + fn)*1.0);
			}



			//cout << precision << " " << recall << endl;
		}
		precision = precision / divider;
		recall = recall / divider;

		cout << precision << " " << recall << endl;
	}
	
	waitKey(0);

	return 0;
}

