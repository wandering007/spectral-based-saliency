# Spectral-based Saliency
Visual saliency algorithms based on spectral analysis models, implemented by OpenCV C++.  
`void SR_saliency(cv::Mat &img, cv::Mat &saliency_map)`: `img` is the input grayscale image, `saliency_map` is the resulting 64x64 saliency map using the algorithm proposed by [1].  
`void PQFT(cv::Mat &pre_img, cv::Mat &next_img, cv::Mat &saliency_map)`: For video processing, `pre_img` is the previous video BGR frame, `next_img` is the current video BGR frame; For static image processing, `pre_img` and `next_img` should be the same. `saliency_map` is the resulting 64x64 saliency map using the algorithm proposed by [2].
##References  
[1] Hou, Xiaodi, and Liqing Zhang. "Saliency detection: A spectral residual approach." 2007 IEEE Conference on Computer Vision and Pattern Recognition. IEEE, 2007.  
[2] Guo, Chenlei, Qi Ma, and Liming Zhang. "Spatio-temporal saliency detection using phase spectrum of quaternion fourier transform." Computer vision and pattern recognition, 2008. cvpr 2008. ieee conference on. IEEE, 2008.

##License
[MIT](https://github.com/wandering007/spectral-based-saliency/blob/master/LICENSE)
