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

int width, height;
int image_num = 0;
vector<String> fn;
vector<String>Output;

float Euclidistance(float x1, float y1, float x2, float y2)
{
	float x = (x1 - x2); //calculating number to square in next step
	float y = (y1 - y2);
	float dist;

	dist = powf(x, 2.0) + powf(y, 2.0);
	dist = sqrtf(dist);

	return dist;
}

void draw_Map(vector<float>sal , Mat img , Superpixels sp,string s)
{
	Mat labels = sp.getLabels();
	Mat output(img);
	cvtColor(output, output, CV_BGR2GRAY);

	for (int y = 0; y < (int)output.rows; ++y) {
		for (int x = 0; x < (int)output.cols; ++x)

		{
			int lbl = labels.at<int>(y, x);
			uchar intensity = sal[lbl] * 255.0;
			output.at<uchar>(y, x) = intensity;
		}
	}

	string ss = "Output/"+ fn[image_num];
	ss += s+".jpg";
	imwrite(ss,output);
}



void normalized(vector<float>* sal_Map)
{
	float min = (float)*min_element((*sal_Map).begin(), (*sal_Map).end());
	float max = (float)*max_element((*sal_Map).begin(), (*sal_Map).end());


	for (int i = 0; i < (*sal_Map).size(); i++)
	{
		float sal = (*sal_Map)[i];
		sal = (sal - min) / (max - min);
		(*sal_Map)[i] = sal;
	}
}



float d_foci(Point X , vector<Point>attention_region)
{
	float dis, d_foci=9999;

	for (int i = 0; i < attention_region.size() ; i++)
	{
		float x1, x2, y1, y2;

		x1 = X.x/(width*1.0) ;
		y1 = X.y/(height*1.0) ;
		x2 = attention_region[i].x/(width*1.0) ;
		y2 = attention_region[i].y/(height*1.0) ;

		dis = Euclidistance(x1, y1, x2, y2);
		
		if (dis < d_foci)
		{
			d_foci = dis;
		}
		
	}
	return d_foci ;
}

vector<float> Color_importance_map(Mat img,Superpixels sp)
{
	vector<float>Color_sal;
	Mat colorImage(img);
	Mat src;
	cvtColor(colorImage, src, CV_BGR2Lab);


	//imshow("COLOR image", colorImage);
	//imshow("LAB image", src) ;

	// Create a vector for the channels and split the original image into B G R colour channels.
	// Keep in mind that OpenCV uses BGR and not RGB images


	vector<Mat> spl;
	split(src, spl);

	//imshow("LIGHT channel ", spl[0]);//L
	//imshow("A channel ", spl[1]);//A
	//imshow("B channel ", spl[2]);//B
	Scalar mean_labimage = mean(src);

	/*
	mean_labimage[0] ------->>>> light channel mean color
	mean_labimage[1]------->>>>> a channel mean color
	mean_labimage[2]------->>>>> b channel mean color
	*/


	/// Establish the number of bins

	int histSize = 256;

	/// Set the ranges ( for B,G,R) )
	float range[] = { 0, 256 };
	const float* histRange = { range };

	bool uniform = true; bool accumulate = false;

	Mat l_hist, a_hist, b_hist;

	/// Compute the histograms:

	calcHist(&spl[0], 1, 0, Mat(), l_hist, 1, &histSize, &histRange, uniform, accumulate);
	calcHist(&spl[1], 1, 0, Mat(), a_hist, 1, &histSize, &histRange, uniform, accumulate);
	calcHist(&spl[2], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate);


	int mode_light, max_l = l_hist.at<float>(0, 0);
	int mode_a, max_a = a_hist.at<float>(0, 0);
	int mode_b, max_b = b_hist.at<float>(0, 0);

	for (int i = 0; i<histSize; i++)
	{
		int val_l = l_hist.at<float>(i, 0);
		int val_a = a_hist.at<float>(i, 0);
		int val_b = b_hist.at<float>(i, 0);

		if (max_l < val_l)
		{
			max_l = val_l;
			mode_light = i;
		}

		if (max_a < val_a)
		{
			max_a = val_a;
			mode_a = i;
		}
		if (max_b < val_b)
		{
			max_b = val_b;
			mode_b = i;
		}
	}

	// Hurray we calculated all the parameter that we need to calculate color Importance Map
	// COLOR IMPORTANCE MAP 
	// Compute adjusted global average color vector

	float L_avg = MIN((mean_labimage[0] + mode_light) / 2.0, mean_labimage[0]);
	float A_avg = (mean_labimage[1] + mode_a) / 2.0;
	float B_avg = (mean_labimage[2] + mode_b) / 2.0;

	//cout << "L channel  = " << "Mean = " << mean_labimage[0] << "  Mode = " << mode_light << "   AVG = " << L_avg << endl;
	//cout << "A channel  = " << "Mean = " << mean_labimage[1] << "  Mode = " << mode_a << "  AVG = " << A_avg << endl;
	//cout << "B channel  = " << "Mean = " << mean_labimage[2] << "  Mode = " << mode_b << "  AVG = " << B_avg << endl;


	Mat tar(src.rows, src.cols, CV_8UC3, Scalar(L_avg, A_avg, B_avg));
	absdiff(src, tar, src);

	//imshow(" abs difference", src);

	vector<Mat> spl_diff;
	split(src, spl_diff);

//	imshow("Diff L channel", spl_diff[0]);  //L
	//imshow("Diff A channel ", spl_diff[1]);  //A
	//imshow("Diff B channel ", spl_diff[2]);  //B

											 /*
											 Combine the diff. Lab Image
											 channels into a single channel
											 Color Importance map
											 */


	float wl = 0.1, wa = 0.45, wb = 0.45;

	Mat g_xy(spl[0]);

	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			int val1 = (spl_diff[0].at<uchar>(i, j)); val1 *= wl;
			int val2 = (spl_diff[1].at<uchar>(i, j)); val2 *= wa;
			int val3 = (spl_diff[2].at<uchar>(i, j)); val3 *= wb;
			int val = val1 + val2 + val3;
			uchar intensity = val;
			g_xy.at<uchar>(i, j) = intensity;
		}
	}
	//imshow("G_XY", g_xy);

	Mat n_xy(spl[0]);

	for (int i = 0; i < n_xy.rows; i++)
	{
		for (int j = 0; j < n_xy.cols; j++)
		{
			int val1 = (spl_diff[0].at<uchar>(i, j)); val1 *= val1;
			int val2 = (spl_diff[1].at<uchar>(i, j)); val2 *= val2;
			int val3 = (spl_diff[2].at<uchar>(i, j)); val3 *= val3;
			int val = sqrt((val1 + val2 + val3));
			uchar intensity = val;
			n_xy.at<uchar>(i, j) = intensity;
		}
	}

	//imshow("N_XY", n_xy);
	Mat Imp_prime(spl_diff[0]);
	for (int i = 0; i < Imp_prime.rows; i++)
	{
		for (int j = 0; j < Imp_prime.cols; j++)
		{
			int val1 = (g_xy.at<uchar>(i, j));
			int val2 = (n_xy.at<uchar>(i, j));
			int val = val1*0.677*val2;
			uchar u = val /1050;
			Imp_prime.at<uchar>(i, j) = u;
		}
	}

	normalize(Imp_prime, Imp_prime, 0, 255, NORM_MINMAX, CV_8UC1);
	medianBlur(Imp_prime, Imp_prime, 7);
	//imshow("mediun PRIME ", Imp_prime);

	//return Color_sal;
	// Now calculate Color Saliency For Color Model of an image
	
	vector <float>color_saliency;
	Mat labels = sp.getLabels();
	vector<Point> centers = sp.getCenters();
	vector<float>num_pixels(centers.size(), 0);
	vector<int>pixels_counter(centers.size(), 0);

	for (int y = 0; y < (int)Imp_prime.rows; ++y) {
		for (int x = 0; x < (int)Imp_prime.cols; ++x) {
			int lbl = labels.at<int>(y, x);
			num_pixels[lbl]+=Imp_prime.at<uchar>(y, x);
			pixels_counter[lbl]++;
		}
	}
	for (int i = 0; i < centers.size(); i++)
	{
		num_pixels[i] = num_pixels[i] / (255*pixels_counter[i]);
	}
	normalized(&num_pixels);
	draw_Map(num_pixels, img, sp, "COLOR SALIENCY");
	return num_pixels;
}


int main()
{

	String imagesPath = "MSRAdataset1K/*.jpg"; // it has filters, too !
	
	glob(imagesPath, fn, true); // recursive, if you want
	for (size_t i = 0; i<fn.size(); i++)
	{
		Mat img = imread(fn[i]);

	
		//Mat img = imread("75.jpg");
		if (!img.data) {
			cout << "Bad image ..." << endl;
			return 0;
		}

		cout << fn.size();


		width = img.cols;
		height = img.rows;

		//imshow("Original image", img);
		Superpixels sp(img);

		Mat labels = sp.getLabels();
		Mat boundaries = sp.viewSuperpixels();
		Mat recolored = sp.colorSuperpixels();

		//imshow("Average superpixel colors", recolored);

		// centers of the K clusters
		vector<Point> centers = sp.getCenters();
		vector<Vec3b> avg_colors(centers.size());

		Mat trial;

		cvtColor(recolored, trial, COLOR_BGR2Lab);

		for (int i = 0; i < centers.size(); i++)
		{
			avg_colors[i] = trial.at<Vec3b>(centers[i]);
		}

		// Global Contrast Map


		vector<float>global_sal;

		for (int i = 0; i < centers.size(); i++)
		{
			float sal1 = 0.0;
			for (int j = 0; j < centers.size(); j++)
			{
				if (i != j)
				{
					float x1, x2, y1, y2;
					x1 = centers[i].x / (width*1.0);
					x2 = centers[j].x / (width*1.0);
					y1 = centers[i].y / (height*1.0);
					y2 = centers[j].y / (height*1.0);

					float ds = Euclidistance(x1, y1, x2, y2);

					Vec3f intensity1, intensity2;

					intensity1[0] = avg_colors[i][0] / 255.0; intensity1[1] = avg_colors[i][1] / 255.0; intensity1[2] = avg_colors[i][2] / 255.0;
					intensity2[0] = avg_colors[j][0] / 255.0; intensity2[1] = avg_colors[j][1] / 255.0; intensity2[2] = avg_colors[j][2] / 255.0;
					float color_dis = sqrtf(((intensity2[2] - intensity1[2])*(intensity2[2] - intensity1[2])) + ((intensity2[1] - intensity1[1])*(intensity2[1] - intensity1[1])) + ((intensity2[0] - intensity1[0])*(intensity2[0] - intensity1[0])));

					//cout << "color Distance = " << color_dis << endl;

					float similarity = expf((-color_dis / 0.25));

					float dissimilarity = 1 - similarity;

					float lamda = 1 - expf(-1.0 / 0.25);
					float  dis_ds = (dissimilarity) / (ds*ds + lamda);

					sal1 += dis_ds;

				}
			}

			sal1 = sal1 / centers.size();
			global_sal.push_back(sal1);

		}
		normalized(&global_sal);
		draw_Map(global_sal, img, sp, "global saliency Map");

		//Boundary Aware Contrast Map
		//Calculating boundary region and storing their center and color in a vector 

		vector<float>boundary_sal;

		vector<Point> boundary_centers;
		set<int>point_lbl;


		for (int i = 0; i < img.cols; i++) {
			int lbl = labels.at<int>(0, i);
			point_lbl.insert(lbl);

			int lbl1 = labels.at<int>(img.rows - 2, i);
			point_lbl.insert(lbl1);
		}

		for (int i = 0; i < img.rows; i++) {
			int lbl = labels.at<int>(i, 0);
			point_lbl.insert(lbl);

			int lbl1 = labels.at<int>(i, img.cols - 2);
			point_lbl.insert(lbl1);
		}

		int M_superPixels = point_lbl.size();
		set<int>::iterator it;

		for (it = point_lbl.begin(); it != point_lbl.end(); it++) {
			Point data = centers[*it];
			boundary_centers.push_back(data);
		}

		vector<Vec3b> boundary_avg_colors(boundary_centers.size());

		for (int i = 0; i < boundary_centers.size(); i++)
		{
			int lbl = labels.at<int>(boundary_centers[i]);
			boundary_avg_colors[i] = avg_colors[lbl];

		}


		for (int i = 0; i < centers.size(); i++)
		{
			float sal2 = 0.0;
			vector<float>Dissimilarity_vector;
			for (int j = 0; j < boundary_centers.size(); j++)
			{

				if (centers[i].x != boundary_centers[j].x && centers[i].y != boundary_centers[j].y) {

					float x1, x2, y1, y2;
					x1 = centers[i].x / width;
					x2 = boundary_centers[j].x / width;
					y1 = centers[i].y / height;
					y2 = boundary_centers[j].y / height;
					float ds = Euclidistance(x1, y1, x2, y2);
					Vec3f intensity1, intensity2;

					intensity1[0] = avg_colors[i][0] / 255.0; intensity1[1] = avg_colors[i][1] / 255.0; intensity1[2] = avg_colors[i][2] / 255.0;
					intensity2[0] = boundary_avg_colors[j][0] / 255.0; intensity2[1] = boundary_avg_colors[j][1] / 255.0; intensity2[2] = boundary_avg_colors[j][2] / 255.0;

					float color_dis = sqrtf(((intensity2[2] - intensity1[2])*(intensity2[2] - intensity1[2])) + ((intensity2[1] - intensity1[1])*(intensity2[1] - intensity1[1])) + ((intensity2[0] - intensity1[0])*(intensity2[0] - intensity1[0])));
					float similarity = expf((-color_dis / 0.25));
					float dissimilarity = 1 - similarity;
					float lamda = 1 - expf(-1.0 / 0.25);
					float  dis_ds = (dissimilarity) / (ds*ds + lamda);
					Dissimilarity_vector.push_back(dis_ds);
				}
			}

			sort(Dissimilarity_vector.begin(), Dissimilarity_vector.end());

			for (int k = 0; k < (int)Dissimilarity_vector.size()*.3; k++)
			{
				//cout<<Dissimilarity_vector[k]<<endl  ;
				sal2 += Dissimilarity_vector[k];
			}
			sal2 /= (int)Dissimilarity_vector.size()*.3;
			boundary_sal.push_back(sal2);
		}

		normalized(&boundary_sal);
		draw_Map(boundary_sal, img, sp, "Boundary saliency Map");


		// Color image and lab image 
		/*
		Various Experiment on color mOdel
		We will Experiment on Specially CIE LAB color Model Which give importance to Light and Color Complement
		Lets see what we can achieve .All the good wishes for Us
		*/
		//Color_saliency;
		vector<float>color_Saliency(Color_importance_map(img, sp));

		// combining ColorImportance image and Salient Map Image
		// Taking   average saliency  method From ColorImportance Image


		//Now Combining both the Global Saliency Map and the Boundary Saliency Map 
		// Saliency Map  S  = S_boundary * (1 + S_global)

		vector <float> sal_Map;

		for (int i = 0; i < boundary_sal.size(); i++)
		{
			float sal = boundary_sal[i] * (1 + global_sal[i]) + color_Saliency[i] * .5;
			sal_Map.push_back(sal);
		}
		normalized(&sal_Map);

		draw_Map(sal_Map, img, sp, "Final saliency Map Without Smoothing ");
		// Now smoothing TEchnique

		float std_deviation, sqrd_distance = 0.0;
		float sum = 0.0, mean;
		for (int i = 0; i < sal_Map.size(); i++)
		{
			sum += sal_Map[i];
		}
		mean = sum / sal_Map.size();

		for (int i = 0; i < sal_Map.size(); i++)
		{
			sqrd_distance += powf((abs(sal_Map[i] - mean)), 2.0);
		}
		std_deviation = sqrtf(sqrd_distance / sal_Map.size());



		vector <Point> attention_region;
		for (int i = 0; i < sal_Map.size(); i++)
		{
			if (sal_Map[i] >= 0.8)
			{
				Point X = centers[i];
				attention_region.push_back(X);
			}
		}


		for (int i = 0; i < sal_Map.size(); i++)
		{
			float sal = sal_Map[i];
			float d_focus = d_foci(centers[i], attention_region);

			if ((d_focus <= 0.4) && (sal > mean + std_deviation))
			{

				float val = sal * (1.0 / (1 - d_focus));
				sal_Map[i] = MIN(1.0, val);
			}

			if (d_focus <= 0.5)
			{

				sal_Map[i] = sal* (1 - d_focus);
			}
			else {

				sal_Map[i] = sal*(1 - d_focus)*(1 - d_focus);
			}
		}

		normalized(&sal_Map);
		//draw_Map(sal_Map, img, sp, "Final saliency Map");


		for (int i = 0; i < sal_Map.size(); i++)
		{
			float sum = 0.0;
			float z_sum = 0.0;
			for (int k = 0; k < sal_Map.size(); k++)
			{
				float kalkulation = expf(-abs(sal_Map[i] - sal_Map[k]) / 0.50);
				float cal = sal_Map[k] * kalkulation;

				z_sum += kalkulation;
				sum += cal;
			}
			sal_Map[i] = sum / z_sum;
		}

		normalized(&sal_Map);
		draw_Map(sal_Map, img, sp, "Final saliency Map With Smoothing ");
		image_num++;
		//waitKey(0);
	}
	
	return 0;
}
