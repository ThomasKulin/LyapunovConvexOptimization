#include <stdio.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <fstream>

#define NUM_SIMULATIONS 1000000
#define NUM_TIMESTEPS 1000
#define DT 0.01

// Pendulum parameters
#define LENGTH 1.0
#define MASS 1.0
#define GRAVITY 9.81

// PID parameters
#define KP 10.0
#define KI 0.0
#define KD 1.5
#define CONTROL_LIMIT 2.4

__global__ void init(unsigned int seed, curandState_t* states) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= NUM_SIMULATIONS) return;

    curand_init(seed, idx, 0, &states[idx]);
}

__global__ void simulate(curandState_t* states, double *state, double *initial_state, double *final_state) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= NUM_SIMULATIONS) return;

    // Initial state
    double theta = curand_uniform_double(&states[idx]) * 2 * M_PI - M_PI; // -pi to pi
    double omega = curand_uniform_double(&states[idx]) * 4 - 2; // -1 to 1

    initial_state[idx*2] = theta;
    initial_state[idx*2+1] = omega;

    double integral = 0;
    double previous_error = 0;

    for (int t = 0; t < NUM_TIMESTEPS; t++) {
        // PID controller
        double error = 0 - theta; // error is difference from upright position
        integral += error * DT;
        double derivative = (error - previous_error) / DT;
        double control = KP * error + KI * integral + KD * derivative;
        previous_error = error;

        // Clip control to reasonable values
        if (control > CONTROL_LIMIT) control = CONTROL_LIMIT;
        if (control < -CONTROL_LIMIT) control = -CONTROL_LIMIT;

        // Pendulum dynamics
        double alpha = GRAVITY/LENGTH * sin(theta) + control/MASS/LENGTH/LENGTH;

        // Update state using Euler integration
        theta += DT * omega;
        omega += DT * alpha;

        // Wrap theta to -pi to pi
        if (theta > M_PI) theta -= 2 * M_PI;
        if (theta < -M_PI) theta += 2 * M_PI;

        // Write new state back to global memory
        state[idx*2] = theta;
        state[idx*2+1] = omega;
    }
    
    final_state[idx*2] = theta;
    final_state[idx*2+1] = omega;
}

int main() {
    double *d_state;
    curandState_t *d_states;
    double *d_initial_state;
    double *d_final_state;
    
    cudaError_t error1, error2, error3, error4;
    error1 = cudaMalloc((void**)&d_state, NUM_SIMULATIONS*2*sizeof(double));
    error2 = cudaMalloc((void**)&d_states, NUM_SIMULATIONS*sizeof(curandState_t));
    error3 = cudaMalloc((void**)&d_initial_state, NUM_SIMULATIONS*2*sizeof(double));
    error4 = cudaMalloc((void**)&d_final_state, NUM_SIMULATIONS*2*sizeof(double));
    
    if (error1 != cudaSuccess) {
        fprintf(stderr, "cudaMalloc1 failed: %s\n", cudaGetErrorString(error1));
    }
    else if (error2 != cudaSuccess) {
        fprintf(stderr, "cudaMalloc2 failed: %s\n", cudaGetErrorString(error2));
    }
    else if (error3 != cudaSuccess) {
        fprintf(stderr, "cudaMalloc3 failed: %s\n", cudaGetErrorString(error3));
    }
    else if (error4 != cudaSuccess) {
        fprintf(stderr, "cudaMalloc4 failed: %s\n", cudaGetErrorString(error4));
    }


    init<<<(NUM_SIMULATIONS + 255) / 256, 256>>>(time(NULL), d_states);

    simulate <<< (NUM_SIMULATIONS + 255) / 256, 256 >>> (d_states, d_state, d_initial_state, d_final_state);

    double *h_initial_state = new double[NUM_SIMULATIONS*2];
    double *h_final_state = new double[NUM_SIMULATIONS*2];

    cudaMemcpy(h_initial_state, d_initial_state, NUM_SIMULATIONS*2*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_final_state, d_final_state, NUM_SIMULATIONS*2*sizeof(double), cudaMemcpyDeviceToHost);

    // Write initial states to binary file
    std::ofstream initial_file("initial_states.bin", std::ios::binary);
    initial_file.write(reinterpret_cast<char*>(h_initial_state), NUM_SIMULATIONS*2*sizeof(double));
    initial_file.close();

    // Write final states to binary file
    std::ofstream final_file("final_states.bin", std::ios::binary);
    final_file.write(reinterpret_cast<char*>(h_final_state), NUM_SIMULATIONS*2*sizeof(double));
    final_file.close();

    delete[] h_initial_state;
    delete[] h_final_state;
    cudaFree(d_state);
    cudaFree(d_states);
    cudaFree(d_initial_state);
    cudaFree(d_final_state);
    return 0;
}


