/*
  Author: Changmao Cheng 
  Email: uestcdaniel@gmail.com
  
  The work is a C++ implementation based on:
  Guo, Chenlei, Qi Ma, and Liming Zhang. "Spatio-temporal saliency detection using phase spectrum of quaternion fourier transform." Computer vision and pattern recognition, 2008. cvpr 2008. ieee conference on. IEEE, 2008.
*/

#include <opencv2/opencv.hpp>

void PQFT(cv::Mat &pre_img, cv::Mat &next_img, cv::Mat &saliency_map)
{
  	CV_Assert( !img.empty() && !next_img.empty() );
	CV_Assert( 3 == pre_img.channels() && 3 == next_img.channels() )
 	CV_Assert(pre_img.rows == next_img.rows && pre_img.cols == next_img.cols);
	const int rescale_size = 64;
	cv::Mat rescaled; //expand input image to optimal size
  	cv::resize( next_img, rescaled, cv::Size(rescale_size, rescale_size), CV_INTER_LINEAR );
	// seperate the image in 3 places (B, G and R)
	std::vector< cv::Mat > bgr_planes;
	cv::split( rescaled, bgr_planes );
	cv::Mat b = bgr_planes[0];
	cv::Mat g = bgr_planes[1];
	cv::Mat r = bgr_planes[2];
	
	// get R, G, B, Y
	cv::Mat R = r - (g + b) / 2;
	cv::Mat G = g - (r + b) / 2;
	cv::Mat B = b - (r + g) / 2;
	cv::Mat Y = (r + g) / 2 -  cv::abs(r - g) / 2 - b;
	
	// two color channels
	cv::Mat RG = R - G;
	cv::Mat BY = B - Y;
	// intensity channel and motion channel
	cv::Mat I = (r + g + b) / 3;
  	cv::resize( pre_img, rescaled, cv::Size(rescale_size, rescale_size), CV_INTER_LINEAR );
	cv::split(rescaled, bgr_planes);
	cv::Mat last_I = (bgr_planes[0] + bgr_planes[1] + bgr_planes[2]) / 3;
	cv::Mat M = cv::abs(I - last_I);
	
	// f1 = M + RG * u1
	// attention: float precision is not enough, must use double type
	cv::Mat planes[] = { cv::Mat_< double >(M), cv::Mat_< double >(RG) };
	cv::Mat f1;
	cv::merge(planes, 2, f1);
	// f1 pft get F1
	cv::dft(f1, f1);
	
	cv::split(f1, planes);
	cv::Mat mag1;
	cv::magnitude(planes[0], planes[1], mag1);
	cv::multiply(mag1, mag1, mag1);
	
	// f2 = BY + I * u1
	// attention: float precision is not enough, must use double type
	BY.convertTo(planes[0], CV_64F);
	I.convertTo(planes[1], CV_64F);
	cv::Mat f2;
	// f2 pft get F2
	cv::merge(planes, 2, f2);
	cv::dft(f2, f2);
	
	cv::split(f2, planes);
	cv::Mat mag2;
	cv::magnitude(planes[0], planes[1], mag2);
	cv::multiply(mag2, mag2, mag2);
	
	// get magnitude
	cv::Mat mag = mag1 + mag2;
	cv::sqrt(mag, mag);
	// normalize, only save phase
    	planes[0] = planes[0] / mag;
    	planes[1] = planes[1] / mag;
	cv::merge(planes, 2, f2); // note that what planes contains is f2's data
	
	cv::split(f1, planes);
	// obeying the paper, rescale_size multiplied to justify the coefficient of ifft
	planes[0] = rescale_size * planes[0] / mag;
    	planes[1] = rescale_size * planes[1] / mag;
	cv::merge(planes, 2, f1);
	
	// get phase spectrum, inverse pft
	cv::dft(f1, f1, cv::DFT_INVERSE | cv::DFT_SCALE);
	cv::dft(f2, f2, cv::DFT_INVERSE | cv::DFT_SCALE);

	// get magnitude, which is phase information
	cv::split(f1, planes);
	cv::magnitude(planes[0], planes[1], mag1);
	cv::multiply(mag1, mag1, mag1);
	
	cv::split(f2, planes);
	cv::magnitude(planes[0], planes[1], mag2);
	cv::multiply(mag2, mag2, mag2);
	
  	// the square of q'(t) magnitude
	mag = mag1 + mag2;
	// smoothing
	cv::GaussianBlur( mag, mag, cv::Size(5, 5), 8., 8. );
	// scale to [0, 1], if save to .jpg, rescale it to [0, 255]
	cv::normalize(mag, saliency_map, 0, 1, CV_MINMAX);
	return;
}
