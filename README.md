#include <iostream>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/cuda.hpp>
#include<cuda.h>
#include<cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_functions.h>
#include <cuda_runtime_api.h>
#include<device_atomic_functions.h>

using namespace std;
using namespace cv;
using namespace dnn;
using namespace cuda;

__global__ void gaussianFilterKernel(unsigned char* input, unsigned char* output, int i_cols, int i_rows, int colorWidthStep, double* d_ker) {
	const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

	float pixs_b = 0;
	float pixs_g = 0;
	float pixs_r = 0;
	int a;
	int b;

	if ((xIndex < i_cols) && (yIndex < i_rows)) {
		for (int k = -2;k <= 2;k++) {
			for (int l = -2;l <= 2;l++) {
				a = xIndex + k;
				b = yIndex + l;
				if (a >= 0 && a < i_cols && b >= 0 && b < i_rows) {
					int colourID = b * colorWidthStep + (3 * a);
					float pix_b = input[colourID];
					float pix_g = input[colourID + 1];
					float pix_r = input[colourID + 2];
					pix_b = pix_b * d_ker[(k + 2) * 5 + (l + 2)];
					pix_g = pix_g * d_ker[(k + 2) * 5 + (l + 2)];
					pix_r = pix_r * d_ker[(k + 2) * 5 + (l + 2)];
					pixs_b = pixs_b + pix_b;
					pixs_g = pixs_g + pix_g;
					pixs_r = pixs_r + pix_r;
				}

			}
		}
		int colourID = yIndex * colorWidthStep + (3 * xIndex);
		output[colourID] = (unsigned char)pixs_b;
		output[colourID + 1] = (unsigned char)pixs_g;
		output[colourID + 2] = (unsigned char)pixs_r;
	}
}


void gaussianFilterGPU(Mat& input, Mat& output, double *g_ker) {
	const int colourBytes = input.step * input.rows;
	unsigned char* d_input, * d_output;
	double* d_ker;

	if (cudaMalloc(&d_input, colourBytes) != cudaSuccess) {
		cout << "Allocation Error 1" << endl;
		return;
	}
	if (cudaMalloc(&d_output, colourBytes) != cudaSuccess) {
		cout << "Allocation Error 2" << endl;
		cudaFree(d_input);
		return;
	}

	if (cudaMalloc(&d_ker, sizeof(double)*25) != cudaSuccess) {
		cout << "Allocation Error 3" << endl;
		cudaFree(d_input);
		cudaFree(d_output);
		return;
	}

	if (cudaMemcpy(d_input, input.ptr(), colourBytes, cudaMemcpyHostToDevice) != cudaSuccess) {
		cout << "Cuda Copy Failure 1" << endl;
		cudaFree(d_input);
		cudaFree(d_output);
		cudaFree(d_ker);
		return;
	}

	if (cudaMemcpy(d_ker, g_ker, (sizeof(double) * 25), cudaMemcpyHostToDevice) != cudaSuccess) {
		cout << "Cuda Copy Failure 2" << endl;
		cudaFree(d_input);
		cudaFree(d_output);
		cudaFree(d_ker);
		return;
	}


	const dim3 block(32, 32);
	const dim3 grid((input.cols + block.x - 1) / block.x, (input.rows + block.y - 1) / block.y);

	gaussianFilterKernel<<<grid, block>>>(d_input, d_output, input.cols, input.rows, input.step,d_ker);

	if (cudaDeviceSynchronize() != cudaSuccess) {
		cout << "Error Executing Kernel" << endl;
		cudaFree(d_input);
		cudaFree(d_output);
		cudaFree(d_ker);
		return;
	}

	if (cudaMemcpy(output.ptr(), d_output, colourBytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
		cudaFree(d_input);
		cudaFree(d_output);
		cudaFree(d_ker);
		cout << "Cuda Copy Failure 3" << endl;
	}

	cudaFree(d_input);
	cudaFree(d_output);
	cudaFree(d_ker);
	return;
}


int main(int, char**) {
	double g_ker[25];
	double sig = 1;
	double pi = 3.14159;

	for (int i = -2;i <= 2; i++) {
		cout << endl;
		for (int j = -2; j <= 2; j++) {
			g_ker[(i + 2)* 5 + (j + 2)] = (exp(-(i * i + j * j) / (2 * sig * sig))) / (2 * pi * sig * sig);
			cout << (exp(-(i * i + j * j) / (2 * sig * sig))) / (2 * pi * sig * sig) << " ";
		}
	}

	VideoCapture cap(0);
	if (cap.isOpened() == false)
	{
		cout << "Cannot open Webcam" << endl;
		return -1;
	}

	while (true)
	{
		Mat image;
		cap.read(image);
		imshow("Input", image);

		Mat output(image.rows, image.cols, CV_8UC3);

		gaussianFilterGPU(image, output, g_ker);

		imshow("Output", output);

		if (waitKey(1) == 'q')
		{
			break;
		}
	}


	return 0;
}
