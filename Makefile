.PHONY: all

all: compile run

compile:
	nvcc --cubin -arch=sm_80 -use_fast_math -O3 source/cuda/image_enhancement.cu -o source/image_enhancement.cubin

run:
	python3 source/image_enhancement_cuda.py

image:
	scp ubuntu@gpu:/home/ubuntu/image_enhancement/output.png output.png && eog output.png
	scp ubuntu@gpu:/home/ubuntu/image_enhancement/mask.png mask.png && eog mask.png
	scp ubuntu@gpu:/home/ubuntu/image_enhancement/output-e.png output-e.png && eog output-e.png
	scp ubuntu@gpu:/home/ubuntu/image_enhancement/mask-e.png mask-e.png && eog mask-e.png