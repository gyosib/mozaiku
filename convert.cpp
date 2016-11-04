#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <sys/stat.h>
#include <string>
#include <fstream>
#include <iostream>
#include <vector>
#include <sstream>
#include <typeinfo>
#define ORIGINAL_W 800
#define ORIGINAL_H 500
#define STRUCTURE_W 16 
#define STRUCTURE_H 10
using namespace std;
using namespace cv;

void loadimage(vector<Mat>*, string);
Mat convert(string,vector<Mat>*);
Mat cont(Mat);
Mat group(Mat);

int main(){
	vector<Mat> in;
	cout<<"load"<<endl;
	loadimage(&in, "in/");
	Mat out = convert("original.png", &in);
	namedWindow("mozaiku", WINDOW_AUTOSIZE);
	imshow("mozaiku",out);
	waitKey(0);
	return 0;
}

void loadimage(vector<Mat> *in, string path){
	int i = 0;
	Mat img;
	for(int i=0;;i++){
		img = imread(path+to_string(i)+".png", IMREAD_COLOR);
		//img = cont(img);
		if(img.empty()) break;
		resize(img,img,Size(STRUCTURE_W,STRUCTURE_H),INTER_CUBIC);
		in->push_back(img);
	}
}

Mat convert(string origin, vector<Mat> *in){
	cout<<"convert"<<endl;
	//load original image & resize
	Mat original = imread(origin, IMREAD_COLOR);
	//original = group(original);
	original = cont(original);
	//resize(original, original, Size(STRUCTURE_W,STRUCTURE_H), INTER_CUBIC);
	resize(original, original, Size(ORIGINAL_W,ORIGINAL_H), INTER_CUBIC);

	//return original;
	
	//read pix from original
	vector<Vec3b> pixs;
	for(int y0=0;y0<original.size().height;y0++){
		Vec3b* src = original.ptr<Vec3b>(y0);
		for(int x0=0;x0<original.size().width;x0++){
			pixs.push_back(src[x0]);
		}
	}
	
	cout<<"load original"<<endl;
	//read pix from in
	//tmp: pixel data in one image
	vector<vector<Vec3b>> inpixs;
	for(int i=0;i<in->size();i++){
		vector<Vec3b> tmp;
		Mat img = (*in)[i];
		if(img.empty()) break;
		for(int y=0;y<(*in)[i].size().height;y++){
			Vec3b* src = (*in)[i].ptr<Vec3b>(y);
			for(int x=0;x<(*in)[i].size().width;x++){
				tmp.push_back(src[x]);
			}
		}
		inpixs.push_back(tmp);
	}
		
	//find nealy image
	/*
	begin: left up of finding block
	border: left side of finding line
	p:finding pixel
	*/
	
	cout<<"find"<<endl;
	vector<int> nealy_count(in->size(),0);
	for(int begin=0;begin<=pixs.size()-STRUCTURE_W-(STRUCTURE_H-1)*ORIGINAL_W;begin+=STRUCTURE_W){
		if(begin%ORIGINAL_W == 0 && begin != 0) 
			begin += ORIGINAL_W*(STRUCTURE_H-1);

		if(begin == pixs.size()-ORIGINAL_W-ORIGINAL_W*(STRUCTURE_H))
			begin -= ORIGINAL_W*(STRUCTURE_H-1);

		int mindis = 99999999;
		int minnum = 0; //most nealy number(unit: /pixel)
		//cout<<begin<<endl;
		for(int i=0;i<in->size();i++){
			int s = 0; //inpix
			int dis = 0;
			for(int border=begin;border<=begin+ORIGINAL_W*(STRUCTURE_H-1);border+=ORIGINAL_W){
				//cout<<i<<","<<border<<","<<begin<<endl;
				for(int p=0;p<STRUCTURE_W;p++){
					/* WAR: span is one pixel! */
					//distance between originl and in
					//cout<<border<<","<<p<<","<<s<<endl;
					dis += sqrt(
						((int)pixs[p+border][0]-(int)inpixs[i][s][0])*
						((int)pixs[p+border][0]-(int)inpixs[i][s][0])+
						((int)pixs[p+border][1]-(int)inpixs[i][s][1])*
						((int)pixs[p+border][1]-(int)inpixs[i][s][1])+
						((int)pixs[p+border][2]-(int)inpixs[i][s][2])*
						((int)pixs[p+border][2]-(int)inpixs[i][s][2]));
					s++;

				}
			}
			/* replace */
			if(dis < mindis){
				minnum = i;
				mindis = dis;
			}
		}
		int s = 0;
		for(int ry=0;ry<STRUCTURE_H;ry++){
			Vec3b* src2 = original.ptr<Vec3b>(ry+begin/ORIGINAL_W);
			for(int rx=0;rx<STRUCTURE_W;rx++){
				src2[rx+begin%ORIGINAL_W] = Vec3b(inpixs[minnum][s][0],inpixs[minnum][s][1],inpixs[minnum][s][2]);
				//src2[rx+begin%ORIGINAL_W] = Vec3b(pixs[begin][0],pixs[begin][1],pixs[begin][2]);
				s++;
			}
		}
	}
	/*cout<<"finish"<<endl;
	  namedWindow("mozaiku", WINDOW_AUTOSIZE);
	  cout<<"finish"<<endl;
	  imshow("mozaiku",original);
	  cout<<"finish"<<endl;
	  waitKey(0);
	  cout<<"finish"<<endl;*/
	return original;
}

Mat cont(Mat img){
	Mat result;

	float a = 25.0;
	uchar lut[256];
	for(int i=0;i<256;i++)
		lut[i] = 255.0/(1+exp(-a*(i-128)/255));

	/*Mat p = original.reshape(0,1).clone();
	  for(int i=0;i<p.cols;i++){
	  p.at<uchar>(0,i) = lut[p.at<uchar>(0,i)];
	  }*/
	LUT(img,Mat(Size(256,1),CV_8U,lut),result);

	return result;
}

Mat group(Mat img){
	const int cluster_count = 32; //number of cluster

	/* reshape to 1 column matrix */
	Mat points;
	img.convertTo(points,CV_32FC3);
	points = points.reshape(3,img.rows*img.cols);

	/* k-means clustering */
	TermCriteria criteria(TermCriteria::COUNT,100,1); //100 loop
	Mat centers;
	Mat_<int> clusters(points.size(),CV_32SC1);
	kmeans(
		points,cluster_count,clusters,criteria,1,KMEANS_RANDOM_CENTERS,centers);
	Mat dst_img(img.size(),img.type());
	MatIterator_<Vec3b> 
		itd = dst_img.begin<Vec3b>(),
		itd_end = dst_img.end<Vec3b>();
	for(int i=0;itd != itd_end;++itd,++i){
		Vec3f &color = centers.at<Vec3f>(clusters(i),0);
		(*itd)[0] = saturate_cast<uchar>(color[0]);
		(*itd)[1] = saturate_cast<uchar>(color[1]);
		(*itd)[2] = saturate_cast<uchar>(color[2]);
	}

	return dst_img;
}
