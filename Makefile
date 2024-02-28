.PHONY: all

all: compile run

compile:
	nvcc --cubin -arch=sm_80 -use_fast_math -O3 source/cuda/image_enhancement.cu -o source/image_enhancement.cubin

run:
	python python3 source/image_enhancement_cuda.py