.PHONY: clean run all

ARCH ?= sm_86
NVCC = nvcc
NVCC_FLAGS = -O3 -arch=$(ARCH)
SRC_DIR = src

all: pendulum

pendulum:
	#$(NVCC) $(NVCC_FLAGS) -o pendulum $(SRC_DIR)/pendulum.cu
	#$(NVCC) $(NVCC_FLAGS) -o sphericalPendulum $(SRC_DIR)/sphericalPendulum.cu
	#$(NVCC) $(NVCC_FLAGS) -o hybridPendulumSk8 $(SRC_DIR)/hybridPendulumSk8.cu
	#$(NVCC) $(NVCC_FLAGS) -o sk8Dyn $(SRC_DIR)/sk8Dyn.cu
	$(NVCC) $(NVCC_FLAGS) -o autonomous $(SRC_DIR)/autonomous.cu

run: pendulum
	#./sphericalPendulum
	#./hybridPendulumSk8
	#./sk8Dyn
	./autonomous

clean:
	rm -f pendulum
	rm -f sphericalPendulum
	rm -f hybridPendulumSk8
	rm -f sk8Dyn
	rm -f autonomous
	rm -f initial_states.bin
	rm -f final_states.bin
