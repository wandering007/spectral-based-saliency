/*
  Author: Changmao Cheng
  Email: uestcdaniel@gmail.com
  
  The work is a C++ implementation of:
  Hou, Xiaodi, and Liqing Zhang. "Saliency detection: A spectral residual approach." 2007 IEEE Conference on Computer Vision and Pattern Recognition. IEEE, 2007.
*/

#include <opencv2/opencv.hpp>
#include <iostream>

bool SR_saliency(cv::Mat &img, cv::Mat &saliency_map)
{
  if( img.empty() )
	{
		std::cerr << "img data is empty, cannot perform saliency detection" << std::endl;
		return false;
	}
	cv::Mat rescaled;
	// convert to grayscale
	if( 3 == img.channels() )
	{
		cv::cvtColor(img, rescaled, CV_BGR2GRAY);
	}
	else if( 1 != img.channels() )
	{
		std::cerr << "img channel number is " << img.channels() << ", cannot perform saliency detection" << std::endl;
		return false;
	}
  
	const int rescale_size = 64;
  cv::resize( rescaled, rescaled, cv::Size(rescale_size, rescale_size) );
  
  cv::Mat planes[] = { cv::Mat_< double >(rescaled), cv::Mat::zeros( rescaled.size(), CV_64F ) };
  cv::Mat complexImg;
  cv::merge(planes, 2, complexImg); // Add to the expanded another plane with zeros
  cv::dft(complexImg, complexImg);  // this way the result may fit in the source matrix
  cv::split(complexImg, planes); // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))

  cv::Mat mag, logmag, smooth, spectralResidual;
  cv::magnitude(planes[0], planes[1], mag);	
	// compute the magnitude and switch to logarithmic scale
  // => log(sqrt(Re(DFT(I))^2 + Im(DFT(I))^2))
  cv::log(mag, logmag);
  cv::boxFilter(logmag, smooth, -1, cv::Size(3,3));
  cv::subtract(logmag, smooth, spectralResidual);
  cv::exp(spectralResidual, spectralResidual);
    
	// real part 
  planes[0] = planes[0].mul(spectralResidual) / mag;
	// imaginary part 
  planes[1] = planes[1].mul(spectralResidual) / mag;

  cv::merge(planes, 2, complexImg);
  cv::dft(complexImg, complexImg, cv::DFT_INVERSE | cv::DFT_SCALE);
  cv::split(complexImg, planes);
	// get magnitude
  cv::magnitude(planes[0], planes[1], mag);
	// get square of magnitude
  cv::multiply(mag, mag, mag);
	// Gaussian kernel size. ksize.width and ksize.height can differ but they both must be positive and odd
	
  cv::GaussianBlur(mag, mag, cv::Size(5,5), 8, 8);
	cv::normalize(mag, saliency_map, 0, 1, CV_MINMAX);
	return saliency_ratio;
}

/*
----Matlab code----
% Preparing the image 
inImg = im2double(rgb2gray(inImage));
[rows cols]=size(inImg);
inImg = imresize(inImg, [64, 64], 'bilinear');

% The actual Spectral Residual computation: just 5 Matlab lines!
myFFT = fft2(inImg); 
myLogAmplitude = log(abs(myFFT));
myPhase = angle(myFFT);
mySpectralResidual = myLogAmplitude - imfilter(myLogAmplitude, fspecial('average', 3), 'replicate'); 
saliencyMap = abs(ifft2(exp(mySpectralResidual + 1i*myPhase))).^2;

% After Effect
saliencyMap = imfilter(saliencyMap, fspecial('disk', 3));

% Resizing from 64*64 to the original size
saliencyMap = mat2gray(saliencyMap);
saliencyMap = imresize(saliencyMap, [rows cols], 'bilinear');

salMap=im2double(saliencyMap);
end
*/
