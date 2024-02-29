// CUDA runtime
#include <cuda_runtime.h>

// Utilities and system includes
#include <helper_functions.h>
#include <helper_cuda.h>
#include "source/cuda/image_enhancement.cu.h"
#include <iostream>

using namespace std;

////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
  // start logs
  cout << "Started" << endl;

  // Use command-line specified CUDA device, otherwise use device with highest
  // Gflops/s
  findCudaDevice(argc, (const char **)argv);

  CImg<unsigned char> src("images/test_img.png");
  int width = src.width();
  int height = src.height();
  cout << "Test passed" << endl;
  exit(EXIT_SUCCESS);
}