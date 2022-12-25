# cv-rbe549-camera-calibration

# AutoCalib: Zhang's Camera Calibration

## Introduction:

This project is an implementation of [Zhang's paper](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr98-71.pdf) which involves using a flat board with distinctive features as a reference. Zhang's method, also known as the Zhang's camera calibration algorithm, is a method for estimating the intrinsic parameters of a pinhole camera model from a set of observed image points. The intrinsic parameters of a camera are those that describe the internal properties of the camera, such as the focal length and the principal point, which are necessary for accurate 3D reconstruction from 2D images.

To estimate the intrinsic parameters using Zhang's method, you will need to first collect a set of 2D image points that correspond to 3D points in the world. These image points can be obtained by manually clicking on the image points of interest, or by using a special calibration pattern (such as a checkerboard pattern) and automatically detecting the corner points of the pattern in the images.

Once you have a set of 2D-3D point correspondences, you can use Zhang's method to estimate the intrinsic parameters by minimizing the reprojection error between the observed 2D image points and the projected 3D points using an optimization algorithm. The reprojection error is the difference between the observed image points and the image points that are projected from the 3D points using the estimated intrinsic parameters.

The optimization algorithm iteratively adjusts the intrinsic parameters to minimize the reprojection error until it reaches a satisfactory level of accuracy. The intrinsic parameters that result in the lowest reprojection error are then taken as the final estimates of the intrinsic parameters of the camera.

Overall, Zhang's method is a reliable and widely used technique for estimating the intrinsic parameters of a pinhole camera model, and it is a key step in many computer vision and 3D reconstruction applications.

## Camera Calibration:

This is the first important step to use camera accurately and effectively for any application. Camera calibration aims at estimating intrinsic (focal length, optical center and distortion) and extrinsic (rotation and translation with respect to world frame) parameters of camera. The intrisics are given by K(3x3) matrix and extrinsics are given by R(3x3) and vector t(3x1)

- Inputs : 2D image co-ordinates and world co-ordinates.
- Output : K, R, t

For calculating intrinsics we calculate the homography matrices between the images captured from different view points and the world points.

## Calculate Homography:
We calculate Homography between the image points and the world points. This gives us 13 Homography (H) matrices. 

## Estimate Intrinsic Parameters:
For calculating intrinsic parameters, we use the B matrix which is obtained from Homography matrices calculated above. Refer paper for detailed mathematical equations. From this B matrix, we get alpha, gamma, beta, u0 and v0 to get the initial estimate (A) matrix.

## Estimate Extrinsic Parameters:
Now we calculate extrinsic parameters for all the Homographies calculated and append them to get the initial estimate of the extrinsics.

## Optimization loss and Reprojection error:
Once we get the initial estimates for the parameters, we try to minimize the re-projection error. The calculated error is passed on to the least squares function as the loss function. After calculating the optimized parameters, we then calculate the re-projected points and average reprojected error.


## Running the code

```sh
python3 Wrapper.py
```
