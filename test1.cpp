#include <cv.h>
#include <cxcore.h>
#include <highgui.h>
#include <iostream>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <fstream>
#include <time.h>
#include <ctype.h>
#include <list>
#include "ZernikeMoment.h"
#include "svm.h"
#include "mclmcr.h"
#include "matrix.h"
#include "mclcppclass.h"
#include "libmfcc.h"
#include "libextra.h"
#include <mmsystem.h> 
#include<windows.h>
#include <atlbase.h>
#pragma comment(lib,"winmm.lib") 


using namespace std;
using namespace cv;



char image_name[50];

CvSize ImageSize = cvSize(720,480);
CvSize ImageSize3 = cvSize(151,151);

int point_y=0;

void capture_frame();

double Red;
double Green;
double Blue;
double FacePercentPixel;
double FacePercent;

CvScalar Scalar1;
int label;

IplImage *FaceImage,*EigenfaceImage,*src2,*dst,*dst1,*dst2,*BinaryImage,*resultimage,*resultwave;
IplImage *difference, *frame1_1, *frame2_1,*difference1,*frame1_2, *frame2_2, *cumulative_difference,*cumulative_difference1,*trans,*trans1;
clock_t start,phase1,phase2, finish;

double FindFace(double );
int Labeling();
void LabelSet(int ,int ,int );
int  m1;
void FindBiggestComponent();

CvCapture *capture;
IplImage *frame;


bool initial=true;
bool initialmfcc_1=true;
bool initialmfcc_3=true;
bool initialmfcc_5=true;
bool initialmfcc_7=true;

bool class1=false;
bool class2=false;
bool class3=false;
bool class4=false;
bool class5=false;
bool class6=false;
bool class7=false;

bool framefirst=true;
bool music=true;
bool create_flag = true;
bool create_flag1 = true;
bool flag_difference = false;

#define FRAMES 6
double w=0.001;
double w_pre=0.001;
int w_update=0;
int z_moment=0;
int mfcc_a=0;
int mfcc_count=0;
int acc_thresh = 20;//累積圖 下降的pixel
int eigen_w=0;
int eigen_h=0;
int framenum=0;
int number=0;

int acc_num=10;
int abc_x,abc_y;
int wave=0;

int pre_max_x=0;
int pre_max_y=0;
int pre_min_x=0;
int pre_min_y=0;

double pz[6][50];
int pz_label[50];
double pz1[6][50];
int pz_label1[50];
double psound[24][600];
int psound_label[600];
double waveimg_y;
int max_label=0;
int max_label1=0;
int max_image=0;
int max_speech=0;

int class_image[3];
int class_speech[7];

int fs;


char AviFileName[50];
char wav[50];
char source[]="D:\\all";//

char num[50];
char file[10]="a";

//char filename[]="D:\\testing data\\laugh8.txt";
//char filename2[]="D:\\testing data\\all_sound.txt";
fstream fp;
fstream fp2;

CvFont Font1=cvFont(1.5,1);

double abc;

double FindFace1(double white)
{
 int	TotalPixel=0;
 double	FacePercentPixel=0;
 double s;
 double r,g,b;
 double f1,f2,f3,f4;
 double L,A,B;
			for(int j=0;j<frame->height;j++)
			{
				for(int i=0;i<frame->width;i++)
				{
		
					Scalar1=cvGet2D(frame,j,i);
					Blue = (double)Scalar1.val[0];
					Green = (double) Scalar1.val[1];
					Red = (double) Scalar1.val[2];
						
					s = Red + Green + Blue;
					
					r = Red/s;
					g = Green/s;
					b = Blue/s;

					//f1 = ((-1.37)*r*r)+((1.0743)*r)+0.2;
					//f2 = ((-0.776)*r*r)+((0.5601)*r)+0.18;
					f3 = ((r-0.33)*(r-0.33))+((g-0.33)*(g-0.33));
					//f4 = ((r-0.5)*(r-0.5))+((g-0.5)*(g-0.5));


					L=((13933*Red)+(46871*Green)+(4732*Blue))/(2^16);
					A=((377*((14503*Red)-(22218*Green)+(7714*Blue)))/(2^24))+128;
					B=((160*((12773*Red)+(39695*Green)-(52468*Blue)))/(2^24))+128;

					TotalPixel++;
					
					if( L>=255&&A>=255&&B>=255&&f3>white)
					{

						cvSet2D(BinaryImage,j,i,CV_RGB(255,255,255));
						FacePercentPixel=FacePercentPixel+1;

					}
					else
					{
						cvSet2D(BinaryImage,j,i,CV_RGB(0,0,0));
					}

				}
			}
		              
	FacePercentPixel=(double)(FacePercentPixel/TotalPixel);
    return FacePercentPixel;
}

void ConnectedComponent()
{
	CvMemStorage* storage = cvCreateMemStorage(0);
	CvSeq* contour = 0;
	int contour_num = cvFindContours(BinaryImage, storage, &contour, sizeof(CvContour),  CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

	CvSeq *_contour = contour; 
	double maxarea = 0;
	double minarea = 100;
	
	for( ; contour != 0; contour = contour->h_next )  
	{  
		double tmparea = fabs(cvContourArea(contour));
		if(tmparea < minarea)   
		{  
			cvSeqRemove(contour, 0); 
			continue;
		}  
  
		if(tmparea > maxarea)  
		{  
			maxarea = tmparea;
		}  
		
	}  
	contour = _contour;
	for(; contour != 0; contour = contour->h_next)
	{  
		double tmparea = fabs(cvContourArea(contour));
		if (tmparea == maxarea)  
		{  
			cvDrawContours(BinaryImage, contour, CV_RGB(255, 255, 255), CV_RGB(255, 255, 255),0, CV_FILLED, 8, cvPoint(0, 0));
		}  
	}

	int	max_x=0;
	int	max_y=0;
	int	min_x=360;
	int	min_y=240;

	for(int y=0;y<BinaryImage->height;y++)	
	{
		for(int x=0;x<BinaryImage->width;x++)
		{
			Scalar1=cvGet2D(BinaryImage,y,x);

			if( Scalar1.val[0]==255 && x>max_x)
			{
				max_x=x;                         
			}    
			if(Scalar1.val[0]==255 && x<min_x)
			{
				min_x=x;                         
			}
			if(Scalar1.val[0]==255&& y>max_y)
			{
				max_y=y;                         
			}    
			if(Scalar1.val[0]==255 && y<min_y)
			{
				min_y=y;                         
			}
		}
	}

	bool flag_cut_max_x =true;
	bool flag_cut_min_x =true;
	bool flag_cut_max_y =true;
	bool flag_cut_min_y =true;

	int local_pixel;
	int local_skin_pixel;
	double local_percent;
    int cut=10;

	

	while(flag_cut_max_x)
	{
		local_pixel=0;
		local_skin_pixel=0;
		local_percent=0;

		for(int cut_max_x=(max_x-cut);cut_max_x<=max_x;cut_max_x++)
		{
			for(int cut_max_y=min_y;cut_max_y<=max_y;cut_max_y++)
			{
				Scalar1=cvGet2D(BinaryImage,cut_max_y,cut_max_x);
				local_pixel++;
				
				if(Scalar1.val[0]==255)
				{
					local_skin_pixel++;
				}
			}
		
		}
		local_percent=((double)local_skin_pixel/(double)local_pixel);
		
		if(local_percent<0.5)
		{
			max_x=max_x-cut;
		}
		else 
		{
			flag_cut_max_x= false;
		}
	}
	while(flag_cut_min_x)
	{
		local_pixel=0;
		local_skin_pixel=0;
		local_percent=0;
		for(int cut_min_x=min_x;cut_min_x<=min_x+cut;cut_min_x++)
		{
			for(int cut_min_y=min_y;cut_min_y<=max_y;cut_min_y++)
			{
				Scalar1=cvGet2D(BinaryImage,cut_min_y,cut_min_x);
				local_pixel++;
				
				if(Scalar1.val[0]==255)
				{
					local_skin_pixel++;
				}
			}
		
		}
		local_percent=((double)local_skin_pixel/(double)local_pixel);
		if(local_percent<0.5)
		{
			min_x=min_x+cut;
		}
		else 
		{	
			flag_cut_min_x= false;
		}

	}

	while(flag_cut_max_y)
	{
		local_pixel=0;
		local_skin_pixel=0;
		local_percent=0;
		for(int cut_max_y=(max_y-cut);cut_max_y<=max_y;cut_max_y++)
		{
			for(int cut_max_x=min_x;cut_max_x<=max_x;cut_max_x++)
			{
				Scalar1=cvGet2D(BinaryImage,cut_max_y,cut_max_x);
				local_pixel++;
				if(Scalar1.val[0]==255)
				{
					local_skin_pixel++;
				}
			}
		
		}
		local_percent=((double)local_skin_pixel/(double)local_pixel);
		if(local_percent<0.5)
		{
			max_y=max_y-cut;
		}
		else 
		{	
			flag_cut_max_y= false;
		}
	}

	while(flag_cut_min_y)
	{
		local_pixel=0;
		local_skin_pixel=0;
		local_percent=0;

		for(int cut_min_y=min_y;cut_min_y<=min_y+cut;cut_min_y++)
		{
			for(int cut_min_x=min_x;cut_min_x<=max_x;cut_min_x++)
			{
				Scalar1=cvGet2D(BinaryImage,cut_min_y,cut_min_x);
				local_pixel++;
				
				if(Scalar1.val[0]==255)
				{
					local_skin_pixel++;
				}
			}
		}
		local_percent=((double)local_skin_pixel/(double)local_pixel);
		if(local_percent<0.5)
		{
			min_y=min_y+cut;
		}
		else 
		{
			flag_cut_min_y= false;
		}
	}

	if(create_flag1)
	{
		eigen_w=max_x-min_x;
		eigen_h=max_y-min_y;
		EigenfaceImage=cvCreateImage(cvSize(eigen_w,eigen_h),IPL_DEPTH_8U, 3);
		src2=cvCreateImage(cvGetSize(EigenfaceImage),8,1);
		create_flag1 = false;

		pre_max_x=max_x;
		pre_max_y=max_y;
		pre_min_x=min_x;
		pre_min_y=min_y;
	
	}

	/*if(((max_x-min_x)-(pre_max_x-pre_min_x))>50)
	{
		max_x=pre_max_x;
		max_y=pre_max_y;
		min_x=pre_min_x;
		min_y=pre_min_y;
	}

	else if(((max_x-min_x)-(pre_max_x-pre_min_x))<-50)
	{
		max_x=pre_max_x;
		max_y=pre_max_y;
		min_x=pre_min_x;
		min_y=pre_min_y;
	}
	else if(((max_y-min_y)-(pre_max_y-pre_min_y))>50)
	{
		max_x=pre_max_x;
		max_y=pre_max_y;
		min_x=pre_min_x;
		min_y=pre_min_y;
	}
	else if(((max_y-min_y)-(pre_max_y-pre_min_y))<-50)
	{
		max_x=pre_max_x;
		max_y=pre_max_y;
		min_x=pre_min_x;
		min_y=pre_min_y;
	}*/

	int x1,x2=0;
	int y1,y2=0;
	int center_x=(min_x+max_x)/2;
	int center_y=(min_y+max_y)/2;


	if(center_x-(int)(eigen_w/2)<0)
		x1=center_x;
	else  
		x1=(int)(eigen_w/2);
	
	if(center_y-(int)(eigen_h/2)<0)
		y1=center_y;
	else 
		y1=(int)(eigen_h/2);
	
	if(center_x+(int)(eigen_w/2)>=ImageSize.width)
		x2=ImageSize.width-center_x;
	else
		x2=(int)(eigen_w/2);

	if(center_y+(int)(eigen_h/2)>=ImageSize.height)
		y2=ImageSize.height-center_y;
	else
		y2=(int)(eigen_h/2);

	

	for(int i=center_x-x1;i<center_x+x2;i++)
	{
		for(int j=center_y-y1;j<center_y+y2;j++)
		{
				CvScalar s=cvGet2D(BinaryImage,j,i);
				Scalar1=cvGet2D(frame,j,i);
				if(s.val[0]==255)
					cvSet2D(EigenfaceImage,j-(center_y-y1),i-(center_x-x1),Scalar1); 
				else
					cvSet2D(EigenfaceImage,j-(center_y-y1),i-(center_x-x1),CV_RGB(255,255,255));
		}
	}

	//cvRectangle(frame,cvPoint(min_x,min_y),cvPoint(max_x,max_y),CV_RGB(255,0,0),2,CV_AA,0);
	pre_max_x=max_x;
	pre_max_y=max_y;
	pre_min_x=min_x;
	pre_min_y=min_y;

}

void LTP(IplImage* src, IplImage* dst,IplImage* dst1)
{
	for(int j=1;j<src->width;j++)
	{
		for(int i=1;i<src->height;i++)
		{
			uchar neighborhood[8]={0};
			neighborhood[0]	= CV_IMAGE_ELEM( src, uchar, i-1, j-1);
			neighborhood[1]	= CV_IMAGE_ELEM( src, uchar, i, j-1);
			neighborhood[2]	= CV_IMAGE_ELEM( src, uchar, i+1, j-1);
			neighborhood[3]	= CV_IMAGE_ELEM( src, uchar, i, j+1);
			neighborhood[4]	= CV_IMAGE_ELEM( src, uchar, i+1, j+1);
			neighborhood[5]	= CV_IMAGE_ELEM( src, uchar, i+1, j);
			neighborhood[6]	= CV_IMAGE_ELEM( src, uchar, i-1, j+1);
			neighborhood[7]	= CV_IMAGE_ELEM( src, uchar, i-1, j);

			uchar center = CV_IMAGE_ELEM( src, uchar, i, j);
			uchar temp=0;
			uchar temp1=0;
			for(int k=0;k<8;k++)
			{
				if((neighborhood[k]-center)>10)
				{
					temp+=1*(1<<k);
				}
				else if((neighborhood[k]-center)<-10)
				{
					temp1+=1*(1<<k);
				}

			}

			CV_IMAGE_ELEM( dst , uchar, i, j)=temp;
			CV_IMAGE_ELEM( dst1, uchar, i, j)=temp1;
		}
	}
}


void subImage(const IplImage* first, const IplImage* second, IplImage* difference)
{
	cvAbsDiff(first,second,difference);
}

void zmoment(IplImage* src,int ltp,int z_framenum)
{
	int MaxNum_n=3;  // The MaxNum_n'th Zernike
	int TotalNum; // the total number of zernike 

	TotalNum = (1+(MaxNum_n/2+1))*(MaxNum_n/2+1); // assume MaxNum_n is even

	ZERNIKERESULT *pZernike =  new ZERNIKERESULT[TotalNum];

	int index = 0; 
	int i,j;
	
	ZernikeMoment *p_ZernikeMoment = new ZernikeMoment();
	   
	double ** pPixelValueCircle_RS = NULL;  

	pPixelValueCircle_RS = p_ZernikeMoment->CreateSpace_CircleImage(src,pPixelValueCircle_RS);
	p_ZernikeMoment->GrayImg_SquaretoCircleTrans(src,pPixelValueCircle_RS);

	 //Caculate Zernike
	for(i=0;i<=MaxNum_n;i++)
	{
		
		if(ltp==1)
		{
			if(i%2 == 0) // if i is even
			{
				for(j=i;j>=0;j=j-2)
				{
					pZernike[index] = p_ZernikeMoment->GetZernike_Mukundan(i,j,src,pPixelValueCircle_RS);

			    
					pz[index][z_framenum]=sqrt(pow(pZernike[index].zernikevalue.Cnm,2)+pow(pZernike[index].zernikevalue.Snm,2));

					/*cout<<"when n ="<<pZernike[index].n<<", m = "<<pZernike[index].m<<"; real = "<<
						  pZernike[index].zernikevalue.Cnm<<"; imaginary = "<<
						  pZernike[index].zernikevalue.Snm<<"; z moment = "<<pz<<endl;*/

					//cout<<pz[index][z_framenum]<<" ";
					//fp<<pz[index][z_framenum]<<" ";
					index++;
				}
			}
			else //if i is odd 
			{
				for(j=i;j>=1;j=j-2)
				{
					pZernike[index] = p_ZernikeMoment->GetZernike_Mukundan(i,j,src, pPixelValueCircle_RS);

					pz[index][z_framenum]=sqrt(pow(pZernike[index].zernikevalue.Cnm,2)+pow(pZernike[index].zernikevalue.Snm,2));
					// Show Zernike result
						/*	cout<<"when n ="<<pZernike[index].n<<", m = "<<pZernike[index].m<<"; real = "<<
						pZernike[index].zernikevalue.Cnm<<"; imaginary = "<<
						  pZernike[index].zernikevalue.Snm<<"; z moment = "<<pz<<endl;*/

						//cout<<pz[index][z_framenum]<<" ";
						//fp<<pz[index][z_framenum]<<" ";
						index++;
				}
			}	
		}

		else if(ltp==2)
		{
			if(i%2 == 0) // if i is even
			{
				for(j=i;j>=0;j=j-2)
				{
					pZernike[index] = p_ZernikeMoment->GetZernike_Mukundan(i,j,src,pPixelValueCircle_RS);

			    
					pz1[index][z_framenum]=sqrt(pow(pZernike[index].zernikevalue.Cnm,2)+pow(pZernike[index].zernikevalue.Snm,2));

					/*cout<<"when n ="<<pZernike[index].n<<", m = "<<pZernike[index].m<<"; real = "<<
						  pZernike[index].zernikevalue.Cnm<<"; imaginary = "<<
						  pZernike[index].zernikevalue.Snm<<"; z moment = "<<pz<<endl;*/

					//cout<<"when n ="<<pZernike[index].n<<", m = "<<pZernike[index].m<<"; z moment = "<<pz[index][z_framenum]<<endl;
					//fp<<pz1[index][z_framenum]<<" ";
					index++;
				}
			}
			else //if i is odd 
			{
				for(j=i;j>=1;j=j-2)
				{
					pZernike[index] = p_ZernikeMoment->GetZernike_Mukundan(i,j,src, pPixelValueCircle_RS);

					pz1[index][z_framenum]=sqrt(pow(pZernike[index].zernikevalue.Cnm,2)+pow(pZernike[index].zernikevalue.Snm,2));
					// Show Zernike result
						/*	cout<<"when n ="<<pZernike[index].n<<", m = "<<pZernike[index].m<<"; real = "<<
						pZernike[index].zernikevalue.Cnm<<"; imaginary = "<<
						  pZernike[index].zernikevalue.Snm<<"; z moment = "<<pz<<endl;*/

						//cout<<"when n ="<<pZernike[index].n<<", m = "<<pZernike[index].m<<"; z moment = "<<pz[index][z_framenum]<<endl;
						//fp<<pz1[index][z_framenum]<<" ";
						index++;
				}
			}	
		
		
		}
	}

}

///////////////////////////////////

int NUM = 12;
int NUM1 = 24;

struct point {
	double *feature;
	signed char value;
	//int value;
};

list<point> point_list;
list<point> point_list1;

void clear_all()
{
	point_list.clear();
	point_list1.clear();
}

void svmimage(int z_framenum)
{
	clear_all();
	for(int a =0;a<z_framenum;a++)
	{
		double *line = new double[NUM];
		for(int i =0;i<NUM;i++)
		{
			
			line[i]=pz[i][a];
		}
		point p;
		p.value=pz_label[a];
		p.feature = line;
		point_list.push_back(p);
	}
	point_list.pop_back();
}

void testData_image()
{
	svm_model *model = svm_load_model("D:\\train_image.txt");
	svm_node *x_space = new svm_node[NUM+1];
	double d;
	int i;
	int k =0;
	
	for(int i=0;i<3;i++)
		class_image[i]=0;
	for (list<point>::iterator q = point_list.begin(); q != point_list.end(); q++)
	{
		for(i =0 ;i<NUM;i++)
		{
			x_space[i].index = i+1;
			x_space[i].value = q->feature[i];
		}
		x_space[NUM].index = -1;

		d = svm_predict(model, x_space);

		if(d == 1)
			class_image[0]++;
		else if(d == 2)
		{
			class_image[1]++;
		}
		else if(d == 3)
		{
			class_image[2]++;
		}
		
	}

	//cout << class_image[0]<<" "<<class_image[1]<<" "<<class_image[2]<<endl;
	delete [] x_space;
}

void svmspeech()
{
	clear_all();
	for(int a =0;a<600;a++)
	{
		double *line = new double[NUM1];
		for(int i =0;i<NUM1;i++)
		{
			
			line[i]=psound[i][a];
		}
		point p;
		p.value=psound_label[a];
		p.feature = line;
		point_list.push_back(p);
	}
	point_list.pop_back();
}

void testData_speech()
{
	svm_model *model = svm_load_model("D:\\newtrain_speechQQ.txt");
	svm_node *x_space = new svm_node[NUM1+1];
	double d;
	int i;
	int k =0;
	
	for(int i=0;i<7;i++)
		class_speech[i]=0;
	for (list<point>::iterator q = point_list.begin(); q != point_list.end(); q++)
	{
		for(i =0 ;i<NUM1;i++)
		{
			x_space[i].index = i+1;
			x_space[i].value = q->feature[i];
		}
		x_space[NUM1].index = -1;

		d = svm_predict(model, x_space);
		if(d == 1)
			class_speech[0]++;
		else if(d == 2)
		{
			class_speech[1]++;
		}
		else if(d == 3)
		{
			class_speech[2]++;
		}
		else if(d == 4)
		{
			class_speech[3]++;
		}
		else if(d == 5)
		{
			class_speech[4]++;
		}
		else if(d == 6)
		{
			class_speech[5]++;
		}
		else if(d == 7)
		{
			class_speech[6]++;
		}
	}
	//cout << class_speech[0]<<" "<<class_speech[1]<<" "<<class_speech[2]<<" "<<class_speech[3]<<" "<<class_speech[4]<<" "<<class_speech[5]<<" "<<class_speech[6]<<endl;
	delete [] x_space;
}

int main()
{

	if( !libmfccInitialize())
	{
		cout <<"Could not initialize libmfcc!" << endl;
		return -1;
	}

	/*if( !libextraInitialize())
	{
		cout <<"Could not initialize libextra!" << endl;
		return -1;
	}*/

	mwArray mwA(16000,1, mxDOUBLE_CLASS); 
	mwArray mwB(24,120, mxDOUBLE_CLASS);
	mwArray mwC(1,1, mxDOUBLE_CLASS);
	mwArray mwFre(1,1, mxDOUBLE_CLASS);

	//extra(source);

	strcpy(AviFileName,source);
	strcat (AviFileName,".avi");

	strcpy(wav,source);
	strcat (wav,".wav");


    capture = cvCaptureFromAVI(AviFileName);

	for(int i=0;i<50;i++)
		pz_label[i]=1;
	for(int i=0;i<600;i++)
		psound_label[i]=1;

	resultimage=cvCreateImage(cvSize(720,680),IPL_DEPTH_8U,3);
	resultwave=cvCreateImage(cvSize(720,200),IPL_DEPTH_8U,3);

	cvSetZero(resultwave);
	cvLine(resultwave,cvPoint(144,0),cvPoint(144,200),CV_RGB(0, 115, 230),2,CV_AA,0);
	cvLine(resultwave,cvPoint(288,0),cvPoint(288,200),CV_RGB(0, 115, 230),2,CV_AA,0);
	cvLine(resultwave,cvPoint(432,0),cvPoint(432,200),CV_RGB(0, 115, 230),2,CV_AA,0);
	cvLine(resultwave,cvPoint(576,0),cvPoint(576,200),CV_RGB(0, 115, 230),2,CV_AA,0);

	start = clock();
	
	//fp.open(filename, ios::out|ios::app);
	//fp2.open(filename2, ios::out|ios::app);
	for(;;)   
        {	 
			if (z_moment==0&&framefirst)
			{
				phase1 = clock();
				framefirst=false;
			}
		
                frame = cvQueryFrame(capture);
				if(frame)
				{
					framenum++;
					cvSetImageROI(resultimage, cvRect(0, 0, frame->width, frame->height));
					cvCopy(frame,resultimage);
					cvResetImageROI(resultimage);

					cvSetImageROI(resultimage, cvRect(0, frame->height, resultwave->width, resultwave->height));
					cvCopy(resultwave,resultimage);
					cvResetImageROI(resultimage);



					if(max_label==1)
						{
							if (max_label1==1)
							{
								cvPutText(resultimage,"cry",cvPoint(10,25),&Font1,CV_RGB(255,0,0));
								cvPutText(resultimage,"pain",cvPoint(10,50),&Font1,CV_RGB(255,0,0));
								cvPutText(resultimage,"high distress",cvPoint(10,75),&Font1,CV_RGB(255,0,0));
							}
							else if (max_label1==2)
							{
								cvPutText(resultimage,"cry",cvPoint(10,25),&Font1,CV_RGB(255,0,0));
								cvPutText(resultimage,"angry",cvPoint(10,50),&Font1,CV_RGB(255,0,0));
								cvPutText(resultimage,"high distress",cvPoint(10,75),&Font1,CV_RGB(255,0,0));
							}
							else if (max_label1==3)
							{
								cvPutText(resultimage,"cry",cvPoint(10,25),&Font1,CV_RGB(255,0,0));
								cvPutText(resultimage,"hungry",cvPoint(10,50),&Font1,CV_RGB(255,0,0));
								cvPutText(resultimage,"medium distress",cvPoint(10,75),&Font1,CV_RGB(255,0,0));
							}
							else if (max_label1==4)
							{
								cvPutText(resultimage,"cry",cvPoint(10,25),&Font1,CV_RGB(255,0,0));
								cvPutText(resultimage,"sleepy",cvPoint(10,50),&Font1,CV_RGB(255,0,0));
								cvPutText(resultimage,"low distress",cvPoint(10,75),&Font1,CV_RGB(255,0,0));
							}
							else if (max_label1==5)
							{
								cvPutText(resultimage,"cry",cvPoint(10,25),&Font1,CV_RGB(255,0,0));
								cvPutText(resultimage,"cuddle",cvPoint(10,50),&Font1,CV_RGB(255,0,0));
								cvPutText(resultimage,"low distress",cvPoint(10,75),&Font1,CV_RGB(255,0,0));
							}
								
						}

						if(max_label==2)
						{
							cvPutText(resultimage,"laugh",cvPoint(10,25),&Font1,CV_RGB(255,0,0));
							cvPutText(resultimage,"laugh",cvPoint(10,50),&Font1,CV_RGB(255,0,0));
							cvPutText(resultimage,"delight",cvPoint(10,75),&Font1,CV_RGB(255,0,0));
						}

						if(max_label==3)
						{
							cvPutText(resultimage,"expressionless",cvPoint(10,25),&Font1,CV_RGB(255,0,0));
							cvPutText(resultimage,"soundless",cvPoint(10,50),&Font1,CV_RGB(255,0,0));
							cvPutText(resultimage,"peace",cvPoint(10,75),&Font1,CV_RGB(255,0,0));
						}

				
			
					cvNamedWindow("System");
					cvShowImage("System",resultimage);
										
					if(initial && framenum==1)
					{
						BinaryImage=cvCreateImage(cvSize(frame->width,frame->height), IPL_DEPTH_8U, 1);
						dst=cvCreateImage(ImageSize3,8,1);
						frame1_1 = cvCreateImage(ImageSize3,IPL_DEPTH_8U ,1);
						frame2_1 = cvCreateImage(ImageSize3,IPL_DEPTH_8U ,1);
						frame1_2 = cvCreateImage(ImageSize3,IPL_DEPTH_8U ,1);
						frame2_2 = cvCreateImage(ImageSize3,IPL_DEPTH_8U ,1);
						trans = cvCreateImage(cvSize(121,121),IPL_DEPTH_8U ,1);
						trans1 = cvCreateImage(cvSize(121,121),IPL_DEPTH_8U ,1);
						
						FacePercent=0;
						FacePercent=FindFace1(w);

						if(FacePercent>0.45)
						{
								FacePercent=0;
								w=0.05;
								FacePercent=FindFace1(w);

								for(;;)
								{
									if(FacePercent>0.45)
									{			
										w_pre=w;
										w=w*2;							
										FacePercent=0;
										FacePercent=FindFace1(w);
									}

									else if(FacePercent<0.35)
									{
										w_pre=w;
										w=w*3/4;
										FacePercent=0;
										FacePercent=FindFace1(w);
									}

									else
										break;

								}
						}

						else if(FacePercent<0.35)
						{
							while(FacePercent<0.35)
							{
									FacePercent=0;
									if(w-0.0005>0)
									{
										w=w-0.0005;
										FacePercent=FindFace1(w);
									}
									else
									{
										w=0;
										FacePercent=FindFace1(w);
										break;
									}
							}
						}


						ConnectedComponent();
						cvCvtColor(EigenfaceImage, src2, CV_BGR2GRAY);
						cvResize( src2, dst,  CV_INTER_AREA );
						LTP(dst,frame1_1,frame1_2);
						
						framenum = 1;	
						initial = false;
					}

					else if (framenum==FRAMES)
					{
						framenum = 0;
						number++;
						w_update++;//計算30秒更新1次W值
					
						//strcpy(image_name,"D:\\testing data\\");  //儲存frame
						//strcat (image_name,file);
						//sprintf(num,"\\%d.jpg",number);
						//strcat (image_name,num);
						//cvSaveImage(image_name,frame);
												
						FacePercent=0;
						FacePercent=FindFace1(w); 
						

						if(mfcc_count==1&&initialmfcc_1)
						{
							mwC.SetData(&mfcc_a,1);
							mfcc(3,mwA,mwB,mwFre,wav,mwC);
							
							for(int i=1;i<=120;i++)
							{
								for(int j=0;j<24;j++)
								{
									psound[j][((i-1)+120*mfcc_a)%600]=mwB.Get(1, i+120*j);
									//fp2<<psound[j][((i-1)+120*mfcc_a)%600]<<" ";
								}
							//	fp2<<endl;
							}
					
							if(music)
							{
								LPCWSTR ppp = CA2W(wav);
								PlaySound(ppp,NULL, SND_ASYNC|SND_NODEFAULT );
								music=false;
							}

							for(int i=1;i<4000;i=i+334)
							{				
								
								if(wave==240)
								{
									wave=0;
								}

								waveimg_y=mwA.Get(1, i);
								cvLine(resultwave,cvPoint((wave-1)*3,point_y),cvPoint(wave*3,100-waveimg_y*100),CV_RGB(255,255,112),1,CV_AA,0);
								point_y=100-waveimg_y*100;
								wave++;

								cvSetImageROI(resultimage, cvRect(0, frame->height, resultwave->width, resultwave->height));
								cvCopy(resultwave,resultimage);
								cvResetImageROI(resultimage);

								//cvNamedWindow("System");
								cvShowImage("System",resultimage);
								cvWaitKey(1);
							}
							initialmfcc_1=false;
						}


						if(mfcc_count==3&&initialmfcc_3)
						{
							for(int i=4001;i<8000;i=i+334)
							{				
								
								waveimg_y=mwA.Get(1, i);
								cvLine(resultwave,cvPoint((wave-1)*3,point_y),cvPoint(wave*3,100-waveimg_y*100),CV_RGB(255,255,112),1,CV_AA,0);
								point_y=100-waveimg_y*100;
								wave++;

								cvSetImageROI(resultimage, cvRect(0, frame->height, resultwave->width, resultwave->height));
								cvCopy(resultwave,resultimage);
								cvResetImageROI(resultimage);

							//	cvNamedWindow("System");
								cvShowImage("System",resultimage);
								cvWaitKey(1);
							}
							initialmfcc_3=false;
						}

						if(mfcc_count==5&&initialmfcc_5)
						{
							for(int i=8001;i<12000;i=i+334)
							{				
								
								waveimg_y=mwA.Get(1, i);
								cvLine(resultwave,cvPoint((wave-1)*3,point_y),cvPoint(wave*3,100-waveimg_y*100),CV_RGB(255,255,112),1,CV_AA,0);
								point_y=100-waveimg_y*100;
								wave++;

								cvSetImageROI(resultimage, cvRect(0, frame->height, resultwave->width, resultwave->height));
								cvCopy(resultwave,resultimage);
								cvResetImageROI(resultimage);

								//cvNamedWindow("System");
								cvShowImage("System",resultimage);
								cvWaitKey(1);
							}
							initialmfcc_5=false;
						}

						if(mfcc_count==7&&initialmfcc_7)
						{
							for(int i=12001;i<16000;i=i+334)
							{				
								
								waveimg_y=mwA.Get(1, i);
								cvLine(resultwave,cvPoint((wave-1)*3,point_y),cvPoint(wave*3,100-waveimg_y*100),CV_RGB(255,255,112),1,CV_AA,0);
								point_y=100-waveimg_y*100;
								wave++;

								cvSetImageROI(resultimage, cvRect(0, frame->height, resultwave->width, resultwave->height));
								cvCopy(resultwave,resultimage);
								cvResetImageROI(resultimage);

								//cvNamedWindow("System");
								cvShowImage("System",resultimage);
								cvWaitKey(1);
							}
							initialmfcc_7=false;
						}

						if(w_update==150)
						{
							if(FacePercent>0.45)
							{
								while(FacePercent>0.45)
								{
						
									FacePercent=0;
									w=w+0.001;
									FacePercent=FindFace1(w);
								}
							}

							else if(FacePercent<0.35)
							{
								while(FacePercent<0.35)
								{
										FacePercent=0;
										if(w-0.0005>0)
										{
										w=w-0.0005;
										FacePercent=FindFace1(w);
										}
										else
										{
										w=0;
										FacePercent=FindFace1(w);
										break;
										}

								}
							}

							w_update=0;
							cout<<"*************************************************"<<endl;
								
								
						}
								    
						/*strcpy(image_name,"D:\\testing data\\");
						strcat (image_name,file);
						sprintf(num,"\\skin_%d.jpg",number);
						strcat (image_name,num);
						cvSaveImage(image_name,BinaryImage);*/
						

						ConnectedComponent();
																							
						/*strcpy(image_name,"D:\\testing data\\");
						strcat (image_name,file);
						sprintf(num,"\\find_%d.jpg",number);
						strcat (image_name,num);
						cvSaveImage(image_name, frame);

						strcpy(image_name,"D:\\testing data\\");
						strcat (image_name,file);
						sprintf(num,"\\cut_%d.jpg",number);
						strcat (image_name,num);
						cvSaveImage(image_name, EigenfaceImage);*/

						cvCvtColor(EigenfaceImage, src2, CV_BGR2GRAY);
						cvResize( src2, dst,  CV_INTER_AREA );

						LTP(dst,frame2_1,frame2_2);

						
						/*strcpy(image_name,"D:\\testing data\\");
						strcat (image_name,file);
						sprintf(num,"\\ltp1_%d.jpg",number);
						strcat (image_name,num);
						cvSaveImage(image_name,frame2_1);

						strcpy(image_name,"D:\\testing data\\");
						strcat (image_name,file);
						sprintf(num,"\\ltp2_%d.jpg",number);
						strcat (image_name,num);
						cvSaveImage(image_name,frame2_2);*/

						if(create_flag)	//初始化只做一次
						{

							difference = cvCreateImage(cvGetSize(frame1_1),IPL_DEPTH_8U,1);
							cumulative_difference = cvCreateImage(cvGetSize(frame1_1),IPL_DEPTH_8U,1);

							difference1 = cvCreateImage(cvGetSize(frame1_1),IPL_DEPTH_8U,1);
							cumulative_difference1 = cvCreateImage(cvGetSize(frame1_1),IPL_DEPTH_8U,1);

							cvSetZero(cumulative_difference);
							cvSetZero(cumulative_difference1);
							create_flag = false;

						}
				
						cvSetZero(difference);
						cvSetZero(difference1);
						cvSetZero(trans);
						cvSetZero(trans1);
							
						subImage(frame1_1,frame2_1,difference);	
						subImage(frame1_2,frame2_2,difference1);	
							
						/*strcpy(image_name,"D:\\testing data\\");
						strcat (image_name,file);
						sprintf(num,"\\diff_%d.jpg",number);
						strcat (image_name,num);
						cvSaveImage(image_name,difference);

						strcpy(image_name,"D:\\testing data\\");
						strcat (image_name,file);
						sprintf(num,"\\diff1_%d.jpg",number);
						strcat (image_name,num);
						cvSaveImage(image_name,difference1);*/
							
						
						if(number==1)
						{
							cvAdd(cumulative_difference,difference,cumulative_difference); 
							cvAdd(cumulative_difference1,difference1,cumulative_difference1); 

							for(int j=0;j<cumulative_difference->width;j++)
							{
								for(int i=0;i<cumulative_difference->height;i++)
								{
								
									CvScalar s=cvGet2D(cumulative_difference,j,i);
									CvScalar s1=cvGet2D(cumulative_difference1,j,i);
																					
									if(j>=14&&j<135&&i>=14&&i<135)
									{
										cvSet2D(trans,j-14,i-14,CV_RGB(0,0,s.val[0]));
										cvSet2D(trans1,j-14,i-14,CV_RGB(0,0,s1.val[0]));
									}

								}
							}
								
							/*strcpy(image_name,"D:\\testing data\\");
							strcat (image_name,file);
							sprintf(num,"\\acc_%d.jpg",number);
							strcat (image_name,num);
							cvSaveImage(image_name,cumulative_difference);

							strcpy(image_name,"D:\\testing data\\");
							strcat (image_name,file);
							sprintf(num,"\\acc1_%d.jpg",number);
							strcat (image_name,num);
							cvSaveImage(image_name,cumulative_difference1);*/

							zmoment(trans,1,z_moment);
							zmoment(trans1,2,z_moment);
							//fp<<endl;

							z_moment++;
							mfcc_count++;

						}

						else
						{
							for(int j=0;j<cumulative_difference->width;j++)
							{
								for(int i=0;i<cumulative_difference->height;i++)
								{
								
									CvScalar s=cvGet2D(cumulative_difference,j,i);
									CvScalar s1=cvGet2D(difference,j,i);
											
									if((s.val[0]-acc_thresh)<10)
									{
										cvSet2D(cumulative_difference,j,i,CV_RGB(0,0,s1.val[0]));
										if(j>=14&&j<135&&i>=14&&i<135)
										{
											cvSet2D(trans,j-14,i-14,CV_RGB(0,0,s1.val[0]));
										}
									}
									else 
									{
										if((s.val[0]-acc_thresh)>=s1.val[0])
										{
											cvSet2D(cumulative_difference,j,i,CV_RGB(0,0,(s.val[0]-acc_thresh)));
											if(j>=14&&j<135&&i>=14&&i<135)
											{
												cvSet2D(trans,j-14,i-14,CV_RGB(0,0,(s.val[0]-acc_thresh)));
											}
										}
										else 
										{
											cvSet2D(cumulative_difference,j,i,CV_RGB(0,0,s1.val[0]));
											if(j>=14&&j<135&&i>=14&&i<135)
											{
												cvSet2D(trans,j-14,i-14,CV_RGB(0,0,s1.val[0]));
											}
										}
									}

									CvScalar s2=cvGet2D(cumulative_difference1,j,i);
									CvScalar s3=cvGet2D(difference1,j,i);
									if((s2.val[0]-acc_thresh)<10)
									{
										cvSet2D(cumulative_difference1,j,i,CV_RGB(0,0,s3.val[0]));
										if(j>=14&&j<135&&i>=14&&i<135)
										{
											cvSet2D(trans1,j-14,i-14,CV_RGB(0,0,s3.val[0]));
										}
									}
									else 
									{
										if((s2.val[0]-acc_thresh)>=s3.val[0])
										{
											cvSet2D(cumulative_difference1,j,i,CV_RGB(0,0,(s2.val[0]-acc_thresh)));
											if(j>=14&&j<135&&i>=14&&i<135)
											{
												cvSet2D(trans1,j-14,i-14,CV_RGB(0,0,(s2.val[0]-acc_thresh)));
											}
										}
										else 
										{
											cvSet2D(cumulative_difference1,j,i,CV_RGB(0,0,s3.val[0]));
											if(j>=14&&j<135&&i>=14&&i<135)
											{
												cvSet2D(trans1,j-14,i-14,CV_RGB(0,0,s3.val[0]));
											}
										}
									}
								}
							}

						   /* strcpy(image_name,"D:\\testing data\\");
							strcat (image_name,file);
							sprintf(num,"\\acc_%d.jpg",number);
							strcat (image_name,num);
							cvSaveImage(image_name,cumulative_difference);

							strcpy(image_name,"D:\\testing data\\");
							strcat (image_name,file);
							sprintf(num,"\\acc1_%d.jpg",number);
							strcat (image_name,num);
							cvSaveImage(image_name,cumulative_difference1);*/

													
							zmoment(trans,1,z_moment);
							zmoment(trans1,2,z_moment);
							//fp<<endl;
							
							z_moment++;
							mfcc_count++;
							
	
						}


						if(mfcc_count==10)
						{
							initialmfcc_1=true;
							initialmfcc_3=true;
							initialmfcc_5=true;
							initialmfcc_7=true;
							mfcc_count=0;
							mfcc_a++;

						}
						

						cvCopy(frame2_1,frame1_1,NULL);
						if(z_moment==50)
						{	

							svmimage(z_moment);
						    testData_image();
						    svmspeech();
						    testData_speech();
							
							max_image=0;
							max_speech=0;
							max_label=0;
							max_label1=0;

							for(int i=0;i<3;i++)
							{
								if(class_image[i]>max_image)
								{
									max_image=class_image[i];
									max_label=i+1;
								}
							}

							if(max_label==1)
							{
								for(int i=0;i<5;i++)
								{
									if(class_speech[i]>max_speech)
									{
										max_speech=class_speech[i];
										max_label1=i+1;
									}
								}	

								if (max_label1==1)
								{
									cvPutText(resultimage,"cry",cvPoint(10,25),&Font1,CV_RGB(255,0,0));
								    cvPutText(resultimage,"pain",cvPoint(10,50),&Font1,CV_RGB(255,0,0));
									cvPutText(resultimage,"high distress",cvPoint(10,75),&Font1,CV_RGB(255,0,0));
								}
								else if (max_label1==2)
								{
									cvPutText(resultimage,"cry",cvPoint(10,25),&Font1,CV_RGB(255,0,0));
								    cvPutText(resultimage,"angry",cvPoint(10,50),&Font1,CV_RGB(255,0,0));
									cvPutText(resultimage,"high distress",cvPoint(10,75),&Font1,CV_RGB(255,0,0));
								}
								else if (max_label1==3)
								{
									cvPutText(resultimage,"cry",cvPoint(10,25),&Font1,CV_RGB(255,0,0));
								    cvPutText(resultimage,"hungry",cvPoint(10,50),&Font1,CV_RGB(255,0,0));
									cvPutText(resultimage,"medium distress",cvPoint(10,75),&Font1,CV_RGB(255,0,0));
								}
								else if (max_label1==4)
								{
									cvPutText(resultimage,"cry",cvPoint(10,25),&Font1,CV_RGB(255,0,0));
								    cvPutText(resultimage,"sleepy",cvPoint(10,50),&Font1,CV_RGB(255,0,0));
									cvPutText(resultimage,"low distress",cvPoint(10,75),&Font1,CV_RGB(255,0,0));
								}
								else if (max_label1==5)
								{
									cvPutText(resultimage,"cry",cvPoint(10,25),&Font1,CV_RGB(255,0,0));
								    cvPutText(resultimage,"cuddle",cvPoint(10,50),&Font1,CV_RGB(255,0,0));
									cvPutText(resultimage,"low distress",cvPoint(10,75),&Font1,CV_RGB(255,0,0));
								}
								
							}

							if(max_label==2)
							{
								cvPutText(resultimage,"laugh",cvPoint(10,25),&Font1,CV_RGB(255,0,0));
								cvPutText(resultimage,"laugh",cvPoint(10,50),&Font1,CV_RGB(255,0,0));
								cvPutText(resultimage,"delight",cvPoint(10,75),&Font1,CV_RGB(255,0,0));
							}

							if(max_label==3)
							{
								cvPutText(resultimage,"expressionless",cvPoint(10,25),&Font1,CV_RGB(255,0,0));
								cvPutText(resultimage,"soundless",cvPoint(10,50),&Font1,CV_RGB(255,0,0));
								cvPutText(resultimage,"peace",cvPoint(10,75),&Font1,CV_RGB(255,0,0));
							}

						
						

							z_moment=0;
							//system("pause");
							cout<<"========================================"<<endl;
					        cvSetZero(resultwave);
							cvLine(resultwave,cvPoint(144,0),cvPoint(144,200),CV_RGB(0, 115, 230),2,CV_AA,0);
							cvLine(resultwave,cvPoint(288,0),cvPoint(288,200),CV_RGB(0, 115, 230),2,CV_AA,0);
							cvLine(resultwave,cvPoint(432,0),cvPoint(432,200),CV_RGB(0, 115, 230),2,CV_AA,0);
							cvLine(resultwave,cvPoint(576,0),cvPoint(576,200),CV_RGB(0, 115, 230),2,CV_AA,0);
							framefirst=true;

							//cvNamedWindow("System");
							cvShowImage("System",resultimage);

							/*strcpy(image_name,"D:\\testing data\\");
							strcat (image_name,file);
							sprintf(num,"\\1\\result_%d.jpg",number);
							strcat (image_name,num);
							cvSaveImage(image_name,resultimage);*/

							phase2 = clock();
							double p1 = (double)(phase2 - phase1);
							
							if((10000-p1)>0)
							cvWaitKey(10000-p1);

							cout<<10000-p1<<"ms"<<endl;

				
						}		
						cvCopy(frame2_2,frame1_2,NULL);
					}
				}
				else 
				{
					PlaySound(NULL,NULL,SND_FILENAME); 
					break;	
				}
		}
    printf("--------video to image over----------\n");

	finish = clock(); 
	double duration = (double)(finish - start);
	cout<<"共"<<duration/1000<<"s"<<endl;
	system("pause");
	//fp.close();

	//fp2.close();

    cvReleaseCapture(&capture);
	cvReleaseImage(&frame);
	cvReleaseImage(&FaceImage);
	cvReleaseImage(&EigenfaceImage);
	cvReleaseImage(&frame1_1);
	cvReleaseImage(&frame2_1);
	cvReleaseImage(&difference);
	cvReleaseImage(&cumulative_difference);
	cvReleaseImage(&frame1_2);
	cvReleaseImage(&frame2_2);
	cvReleaseImage(&difference1);
	cvReleaseImage(&cumulative_difference1);
	cvReleaseImage(&src2);
	cvReleaseImage(&dst);
	cvReleaseImage(&trans);
	cvReleaseImage(&trans1);
	libmfccTerminate();
	libextraTerminate();
	mclTerminateApplication();



	return 0;
}