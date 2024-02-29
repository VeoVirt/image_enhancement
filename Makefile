CXX = nvcc -O3 -arch=sm_80 -use_fast_math -D_FORCE_INLINES -Wno-deprecated-gpu-targets -std=c++11

image_enhancement.o: source/cuda/image_enhancement.cu
	$(CXX) -c $@ $<

image_enhance.o: source/cuda/image_enhance.cu
	$(CXX) -c $@ $<

test: image_enhancement.o image_enhance.o
	$(CXX) -o $@ $+


.PHONY: all

all: compile run

compile:
	nvcc --cubin -arch=sm_80 -use_fast_math -O3 source/cuda/image_enhancement.cu -o source/image_enhancement.cubin

run:
	python3 source/image_enhancement_cuda.py

image:
	scp ubuntu@gpu:/home/ubuntu/image_enhancement/output.png output.png
	scp ubuntu@gpu:/home/ubuntu/image_enhancement/mask.png mask.png
	scp ubuntu@gpu:/home/ubuntu/image_enhancement/output-e.png output-e.png
	scp ubuntu@gpu:/home/ubuntu/image_enhancement/mask-e.png mask-e.png
