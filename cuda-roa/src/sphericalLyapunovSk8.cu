#include <stdio.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <fstream>
#include <iostream>

#define NUM_SIMULATIONS 1000000
#define NUM_TIMESTEPS 1500
#define DT 0.001

// Model parameters
#define l 1
#define L 0.9
#define m1 4
#define m2 90.0
#define g 9.81
#define Ib 0.475  // moment of inertia of board+feet
#define k_truck 250  // skateboard truck spring constant

// Human control parameters
#define Kpy 1066
#define Kdy 349
#define Wprop 0.5
#define Ky 94
#define Cy 5.8
#define Kpprop 1200
#define Kpv 1000
#define Kdprop 210
#define Kdv 500


__global__ void init(unsigned int seed, curandState_t* states) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= NUM_SIMULATIONS) return;

    curand_init(seed, idx, 0, &states[idx]);
}

__global__ void simulate(curandState_t* states, double *state, double *initial_state, double *final_state, int slice_x, int slice_y) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= NUM_SIMULATIONS) return;
    
    double Vx = (curand_uniform_double(&states[idx]) * 2 - 1) * 15;

    // Initial state
    double x = 0;
    double xdot = (curand_uniform_double(&states[idx]) * 2 - 1) * 15;
    double phi = 1e-12;
    double phi_dot = 0;
    double theta = (curand_uniform_double(&states[idx]) * 1 * M_PI - M_PI/2)*1; 
    double theta_dot = (curand_uniform_double(&states[idx]) * 2 - 1)*4; 
    double psi = (curand_uniform_double(&states[idx]) * 1 * M_PI - M_PI/2)*1; 
    double psi_dot = (curand_uniform_double(&states[idx]) * 2 - 1)*4; 
    
    // Set initial states to only vary in 2D slice.
    for(int i = 0; i < 8; i++){
    	if(slice_x == i || slice_y == i){
    	    i = i;
    	}
    	else{
    	    switch(i){
    	    	case 2:
    	    	    theta = 0;
    	    	    break;
    	    	case 3:
    	    	    psi = 0;
    	    	    break;
    	    	case 4:
    	    	    xdot = 0;
    	    	    Vx = 0;
    	    	case 6:
    	    	    theta_dot = 0;
    	    	    break;
    	    	case 7:
    	    	    psi_dot = 0;
    	    	    break;
    	    }
    	}
    }

    // Store initial state in global memory
    initial_state[idx*8 + 0] = x; 
    initial_state[idx*8 + 1] = phi;
    initial_state[idx*8 + 2] = theta; 
    initial_state[idx*8 + 3] = psi;
    initial_state[idx*8 + 4] = xdot; 
    initial_state[idx*8 + 5] = phi_dot;     
    initial_state[idx*8 + 6] = theta_dot; 
    initial_state[idx*8 + 7] = psi_dot;
    

    for (int t = 0; t < NUM_TIMESTEPS; t++) {
    	
    	double x_dot = xdot - Vx;
    	
    	// Ankle torque inputs
        double tau_theta = (-(Kpprop + Kpv) * theta - (Kdprop + Kdv) * theta_dot);
        double tau_psi = (-Kpy * (psi - Wprop * phi) - Kdy * (psi_dot - Wprop * phi_dot) - Ky * (psi - phi) - Cy * (psi_dot - phi_dot));
        double tau_phi = (-tau_psi);
    	
        // Control Inputs
        double y = 0;
        double y_dot = 0;
        
        int k1 = 10;
        int k2 = 10;
        int kp = 5;
        
        double u_x = -(pow(L, 2)*x_dot + pow(L, 3)*x_dot - L*kp*x_dot + pow(L, 2)*k1*x + pow(L, 3)*k1*x + L*pow(kp, 2)*pow(theta_dot, 2)*sin(theta) - pow(L, 2)*kp*pow(theta_dot, 2)*sin(theta) - pow(L, 3)*kp*pow(theta_dot, 2)*sin(theta) + pow(L, 2)*kp*theta_dot*cos(theta) + pow(L, 3)*kp*theta_dot*cos(theta) + L*kp*x_dot*pow(cos(theta), 2) - L*k1*pow(kp, 2)*sin(theta) + pow(L, 2)*k1*kp*sin(theta) + pow(L, 3)*k1*kp*sin(theta) - L*k1*kp*x + L*k1*pow(kp, 2)*pow(cos(theta), 2)*sin(theta) - L*k2*pow(kp, 2)*pow(cos(theta), 2)*sin(theta) - g*pow(kp, 2)*cos(psi)*cos(theta)*sin(theta) + L*k1*kp*x*pow(cos(theta), 2) + L*pow(kp, 2)*pow(psi_dot, 2)*pow(cos(theta), 2)*sin(theta) - pow(L, 2)*kp*pow(psi_dot, 2)*pow(cos(theta), 2)*sin(theta) - pow(L, 3)*kp*pow(psi_dot, 2)*pow(cos(theta), 2)*sin(theta) - L*pow(kp, 2)*theta_dot*pow(cos(psi), 2)*cos(theta) - L*kp*x_dot*pow(cos(psi), 2)*pow(cos(theta), 2) + L*g*kp*cos(psi)*cos(theta)*sin(theta) - L*k1*pow(kp, 2)*pow(cos(psi), 2)*pow(cos(theta), 2)*sin(theta) + L*k2*pow(kp, 2)*pow(cos(psi), 2)*pow(cos(theta), 2)*sin(theta) - L*kp*y_dot*cos(theta)*sin(psi)*sin(theta) - L*k1*kp*x*pow(cos(psi), 2)*pow(cos(theta), 2) + pow(L, 2)*g*kp*cos(psi)*cos(theta)*sin(theta) - L*k2*kp*y*cos(theta)*sin(psi)*sin(theta) - L*pow(kp, 2)*psi_dot*cos(psi)*pow(cos(theta), 2)*sin(psi)*sin(theta))/((pow(L, 2) + L - kp)*(pow(L, 2) + L - kp*pow(cos(psi), 2)*pow(cos(theta), 2)));
	double u_y = (L*kp*y_dot*pow(cos(theta), 2) - pow(L, 3)*y_dot - pow(L, 2)*k2*y - pow(L, 3)*k2*y - pow(L, 2)*y_dot - L*pow(kp, 2)*pow(theta_dot, 2)*cos(theta)*sin(psi) + pow(L, 2)*kp*pow(theta_dot, 2)*cos(theta)*sin(psi) + pow(L, 3)*kp*pow(theta_dot, 2)*cos(theta)*sin(psi) + L*k2*kp*y*pow(cos(theta), 2) - L*pow(kp, 2)*pow(psi_dot, 2)*pow(cos(theta), 3)*sin(psi) + pow(L, 2)*kp*pow(psi_dot, 2)*pow(cos(theta), 3)*sin(psi) + pow(L, 3)*kp*pow(psi_dot, 2)*pow(cos(theta), 3)*sin(psi) + g*pow(kp, 2)*cos(psi)*pow(cos(theta), 2)*sin(psi) - pow(L, 2)*kp*psi_dot*cos(psi)*cos(theta) - pow(L, 3)*kp*psi_dot*cos(psi)*cos(theta) + L*k1*pow(kp, 2)*cos(theta)*sin(psi) - pow(L, 2)*k2*kp*cos(theta)*sin(psi) - pow(L, 3)*k2*kp*cos(theta)*sin(psi) + pow(L, 2)*kp*theta_dot*sin(psi)*sin(theta) + pow(L, 3)*kp*theta_dot*sin(psi)*sin(theta) + L*pow(kp, 2)*psi_dot*cos(psi)*pow(cos(theta), 3) - L*k1*pow(kp, 2)*pow(cos(theta), 3)*sin(psi) + L*k2*pow(kp, 2)*pow(cos(theta), 3)*sin(psi) + L*kp*x_dot*cos(theta)*sin(psi)*sin(theta) - L*g*kp*cos(psi)*pow(cos(theta), 2)*sin(psi) - pow(L, 2)*g*kp*cos(psi)*pow(cos(theta), 2)*sin(psi) + L*k1*kp*x*cos(theta)*sin(psi)*sin(theta))/((pow(L, 2) + L - kp)*(pow(L, 2) + L - kp*pow(cos(psi), 2)*pow(cos(theta), 2)));	

	double fx = (m1+m2*pow(sin(theta), 2))*u_x;
	double fy = m2*u_y;
	x_dot = xdot;
        
        // Calculate second derivatives based on equations of motion
        //double x_ddot = (M_RIDER * LENGTH * sin(theta) * (theta_dot * theta_dot + psi_dot * psi_dot * cos(theta)*cos(theta)) - GRAVITY * M_RIDER * sin(theta) * cos(theta) * cos(psi) + u_x + u_y*sin(psi)*sin(theta)*cos(theta)/LENGTH) / (M_BOARD+M_RIDER*sin(theta)*sin(theta));
        //double theta_ddot = (GRAVITY*M_RIDER*cos(psi)*sin(theta)/LENGTH - M_BOARD*psi_dot*psi_dot*sin(theta)*cos(theta) + M_BOARD*GRAVITY*cos(psi)*sin(theta)/LENGTH -(psi_dot*psi_dot+theta_dot*theta_dot)*M_RIDER*sin(theta)*cos(theta) - u_x*cos(theta)/LENGTH - (M_BOARD+M_RIDER)*u_y*sin(psi)*sin(theta)/(M_RIDER*LENGTH*LENGTH)) /(M_BOARD+M_RIDER*sin(theta)*sin(theta));
        //double psi_ddot =(2*psi_dot*psi_dot*theta_dot*theta_dot*sin(theta) + GRAVITY*sin(psi)/LENGTH)/cos(theta) + u_y*cos(psi)/(M_RIDER*LENGTH*LENGTH*cos(theta)*cos(theta));
        
        double r1 = l/(2*(tan(phi)));
        
        double x_ddot = fx/(m1+m2*pow(sin(theta), 2)) - fy*sin(psi)*sin(theta)*cos(theta)/(L*(m1+m2*pow(sin(theta), 2))) + (4*L*pow(r1, 3)*tau_theta*pow(cos(theta), 2)*sin(psi) - 4*pow(r1, 4)*tau_theta*cos(theta) - pow(L, 2)*pow(r1, 2)*tau_theta*pow(cos(theta), 3)*pow(sin(psi), 2) + 4*pow(L, 2)*m2*pow(r1, 4)*pow(theta_dot, 2)*pow(cos(psi), 2)*pow(sin(theta), 3) + 4*pow(L, 2)*m2*pow(r1, 4)*pow(theta_dot, 2)*pow(sin(psi), 2)*pow(sin(theta), 3) + pow(L, 6)*m2*pow(theta_dot, 2)*pow(cos(theta), 6)*pow(sin(psi), 4)*sin(theta) + pow(L, 4)*m2*pow(x_dot, 2)*pow(cos(theta), 4)*pow(sin(psi), 4)*sin(theta) - 8*pow(L, 3)*m2*pow(r1, 3)*pow(theta_dot, 2)*pow(cos(theta), 3)*sin(psi)*sin(theta) - 6*pow(L, 5)*m2*r1*pow(theta_dot, 2)*pow(cos(theta), 5)*pow(sin(psi), 3)*sin(theta) - 6*pow(L, 3)*m2*r1*pow(x_dot, 2)*pow(cos(theta), 3)*pow(sin(psi), 3)*sin(theta) - 8*L*m2*pow(r1, 3)*pow(x_dot, 2)*cos(theta)*sin(psi)*sin(theta) + 4*pow(L, 2)*m2*pow(psi_dot, 2)*pow(r1, 4)*pow(cos(psi), 2)*pow(cos(theta), 2)*sin(theta) + 4*pow(L, 2)*m2*pow(r1, 4)*pow(theta_dot, 2)*pow(cos(psi), 2)*pow(cos(theta), 2)*sin(theta) + 4*pow(L, 2)*m2*pow(psi_dot, 2)*pow(r1, 4)*pow(cos(theta), 2)*pow(sin(psi), 2)*sin(theta) - 4*pow(L, 3)*m2*pow(psi_dot, 2)*pow(r1, 3)*pow(cos(theta), 3)*pow(sin(psi), 3)*sin(theta) + pow(L, 4)*m2*pow(psi_dot, 2)*pow(r1, 2)*pow(cos(theta), 4)*pow(sin(psi), 4)*sin(theta) + 4*pow(L, 2)*m2*pow(r1, 4)*pow(theta_dot, 2)*pow(cos(theta), 2)*pow(sin(psi), 2)*sin(theta) - 8*pow(L, 3)*m2*pow(r1, 3)*pow(theta_dot, 2)*cos(theta)*pow(sin(psi), 3)*pow(sin(theta), 3) - 4*pow(L, 3)*m2*pow(r1, 3)*pow(theta_dot, 2)*pow(cos(theta), 3)*pow(sin(psi), 3)*sin(theta) + 12*pow(L, 4)*m2*pow(r1, 2)*pow(theta_dot, 2)*pow(cos(theta), 4)*pow(sin(psi), 2)*sin(theta) + pow(L, 4)*m2*pow(r1, 2)*pow(theta_dot, 2)*pow(cos(theta), 4)*pow(sin(psi), 4)*sin(theta) + 12*pow(L, 2)*m2*pow(r1, 2)*pow(x_dot, 2)*pow(cos(theta), 2)*pow(sin(psi), 2)*sin(theta) - 4*pow(L, 2)*m2*pow(r1, 3)*theta_dot*x_dot*pow(sin(psi), 3)*pow(sin(theta), 3) + 2*pow(L, 5)*m2*theta_dot*x_dot*pow(cos(theta), 5)*pow(sin(psi), 4)*sin(theta) + 3*pow(L, 4)*m2*pow(r1, 2)*pow(theta_dot, 2)*pow(cos(theta), 2)*pow(sin(psi), 4)*pow(sin(theta), 3) - 4*L*g*m2*pow(r1, 4)*cos(psi)*cos(theta)*sin(theta) + 3*pow(L, 4)*m2*pow(r1, 2)*pow(theta_dot, 2)*pow(cos(psi), 2)*pow(cos(theta), 2)*pow(sin(psi), 2)*pow(sin(theta), 3) - 16*pow(L, 2)*m2*pow(r1, 3)*theta_dot*x_dot*pow(cos(theta), 2)*sin(psi)*sin(theta) - 12*pow(L, 4)*m2*r1*theta_dot*x_dot*pow(cos(theta), 4)*pow(sin(psi), 3)*sin(theta) - 4*pow(L, 3)*m2*pow(psi_dot, 2)*pow(r1, 3)*pow(cos(psi), 2)*pow(cos(theta), 3)*sin(psi)*sin(theta) - 8*pow(L, 3)*m2*pow(r1, 3)*pow(theta_dot, 2)*pow(cos(psi), 2)*cos(theta)*sin(psi)*pow(sin(theta), 3) - 4*pow(L, 3)*m2*pow(r1, 3)*pow(theta_dot, 2)*pow(cos(psi), 2)*pow(cos(theta), 3)*sin(psi)*sin(theta) + 4*pow(L, 2)*m2*psi_dot*pow(r1, 3)*x_dot*pow(cos(psi), 3)*cos(theta)*pow(sin(theta), 2) + 4*pow(L, 2)*g*m2*pow(r1, 3)*cos(psi)*pow(cos(theta), 2)*sin(psi)*sin(theta) - 4*pow(L, 2)*m2*pow(r1, 3)*theta_dot*x_dot*pow(cos(psi), 2)*sin(psi)*pow(sin(theta), 3) + 24*pow(L, 3)*m2*pow(r1, 2)*theta_dot*x_dot*pow(cos(theta), 3)*pow(sin(psi), 2)*sin(theta) + 2*pow(L, 3)*m2*pow(r1, 2)*theta_dot*x_dot*cos(theta)*pow(sin(psi), 4)*pow(sin(theta), 3) + pow(L, 4)*m2*pow(psi_dot, 2)*pow(r1, 2)*pow(cos(psi), 2)*pow(cos(theta), 4)*pow(sin(psi), 2)*sin(theta) + pow(L, 4)*m2*pow(r1, 2)*pow(theta_dot, 2)*pow(cos(psi), 2)*pow(cos(theta), 4)*pow(sin(psi), 2)*sin(theta) + 4*pow(L, 3)*m2*psi_dot*pow(r1, 3)*theta_dot*pow(cos(psi), 3)*pow(cos(theta), 2)*pow(sin(theta), 2) - pow(L, 3)*g*m2*pow(r1, 2)*cos(psi)*pow(cos(theta), 3)*pow(sin(psi), 2)*sin(theta) + 4*pow(L, 2)*m2*psi_dot*pow(r1, 3)*x_dot*cos(psi)*cos(theta)*pow(sin(psi), 2)*pow(sin(theta), 2) + 4*pow(L, 3)*m2*psi_dot*pow(r1, 3)*theta_dot*cos(psi)*pow(cos(theta), 2)*pow(sin(psi), 2)*pow(sin(theta), 2) - 2*pow(L, 4)*m2*psi_dot*pow(r1, 2)*theta_dot*cos(psi)*pow(cos(theta), 3)*pow(sin(psi), 3)*pow(sin(theta), 2) - 2*pow(L, 4)*m2*psi_dot*pow(r1, 2)*theta_dot*pow(cos(psi), 3)*pow(cos(theta), 3)*sin(psi)*pow(sin(theta), 2) - 2*pow(L, 3)*m2*psi_dot*pow(r1, 2)*x_dot*cos(psi)*pow(cos(theta), 2)*pow(sin(psi), 3)*pow(sin(theta), 2) - 2*pow(L, 3)*m2*psi_dot*pow(r1, 2)*x_dot*pow(cos(psi), 3)*pow(cos(theta), 2)*sin(psi)*pow(sin(theta), 2) + 2*pow(L, 3)*m2*pow(r1, 2)*theta_dot*x_dot*pow(cos(psi), 2)*cos(theta)*pow(sin(psi), 2)*pow(sin(theta), 3))/(pow(r1, 2)*(4*L*m1*pow(r1, 2)*pow(cos(theta), 2) + pow(L, 3)*m1*pow(cos(theta), 4)*pow(sin(psi), 2) - 4*pow(L, 2)*m1*r1*pow(cos(theta), 3)*sin(psi) + pow(L, 3)*m2*pow(cos(theta), 2)*pow(sin(psi), 4)*pow(sin(theta), 2) + L*m1*pow(r1, 2)*pow(cos(psi), 2)*pow(sin(theta), 2) + 4*L*m2*pow(r1, 2)*pow(cos(psi), 2)*pow(sin(theta), 2) + L*m1*pow(r1, 2)*pow(sin(psi), 2)*pow(sin(theta), 2) + 4*L*m2*pow(r1, 2)*pow(sin(psi), 2)*pow(sin(theta), 2) - 4*pow(L, 2)*m2*r1*cos(theta)*pow(sin(psi), 3)*pow(sin(theta), 2) + pow(L, 3)*m2*pow(cos(psi), 2)*pow(cos(theta), 2)*pow(sin(psi), 2)*pow(sin(theta), 2) - 4*pow(L, 2)*m2*r1*pow(cos(psi), 2)*cos(theta)*sin(psi)*pow(sin(theta), 2)));
	
	double phi_ddot = (tau_phi - k_truck*phi)/Ib;
	
	
	double theta_ddot = (m1+m2)*fy*sin(psi)*sin(theta)/(m2*pow(L, 2)*(m1+m2*pow(sin(theta), 2))) - fx*cos(theta)/(L*(m1+m2*pow(sin(theta), 2))) - (pow(L, 6)*pow(m2, 2)*pow(theta_dot, 2)*pow(cos(theta), 5)*pow(sin(psi), 4)*sin(theta) - 4*m2*pow(r1, 4)*tau_theta - m1*pow(r1, 4)*tau_theta + pow(L, 4)*pow(m2, 2)*pow(x_dot, 2)*pow(cos(theta), 3)*pow(sin(psi), 4)*sin(theta) + 4*L*m2*pow(r1, 3)*tau_theta*cos(theta)*sin(psi) - pow(L, 2)*m2*pow(r1, 2)*tau_theta*pow(cos(theta), 2)*pow(sin(psi), 2) - 4*L*g*pow(m2, 2)*pow(r1, 4)*cos(psi)*sin(theta) - 8*L*pow(m2, 2)*pow(r1, 3)*pow(x_dot, 2)*sin(psi)*sin(theta) + 2*pow(L, 5)*pow(m2, 2)*theta_dot*x_dot*pow(cos(theta), 4)*pow(sin(psi), 4)*sin(theta) - 4*pow(L, 2)*m1*m2*pow(r1, 4)*pow(theta_dot, 2)*cos(theta)*sin(theta) + 4*pow(L, 2)*pow(m2, 2)*pow(psi_dot, 2)*pow(r1, 4)*pow(cos(psi), 2)*cos(theta)*sin(theta) + 4*pow(L, 2)*pow(m2, 2)*pow(r1, 4)*pow(theta_dot, 2)*pow(cos(psi), 2)*cos(theta)*sin(theta) - L*g*m1*m2*pow(r1, 4)*cos(psi)*sin(theta) + 4*pow(L, 2)*pow(m2, 2)*pow(psi_dot, 2)*pow(r1, 4)*cos(theta)*pow(sin(psi), 2)*sin(theta) + 4*pow(L, 2)*pow(m2, 2)*pow(r1, 4)*pow(theta_dot, 2)*cos(theta)*pow(sin(psi), 2)*sin(theta) - 8*pow(L, 3)*pow(m2, 2)*pow(r1, 3)*pow(theta_dot, 2)*pow(cos(theta), 2)*sin(psi)*sin(theta) - 6*pow(L, 5)*pow(m2, 2)*r1*pow(theta_dot, 2)*pow(cos(theta), 4)*pow(sin(psi), 3)*sin(theta) + 12*pow(L, 2)*pow(m2, 2)*pow(r1, 2)*pow(x_dot, 2)*cos(theta)*pow(sin(psi), 2)*sin(theta) - 6*pow(L, 3)*pow(m2, 2)*r1*pow(x_dot, 2)*pow(cos(theta), 2)*pow(sin(psi), 3)*sin(theta) - 4*pow(L, 3)*pow(m2, 2)*pow(psi_dot, 2)*pow(r1, 3)*pow(cos(theta), 2)*pow(sin(psi), 3)*sin(theta) + pow(L, 4)*pow(m2, 2)*pow(psi_dot, 2)*pow(r1, 2)*pow(cos(theta), 3)*pow(sin(psi), 4)*sin(theta) - 4*pow(L, 3)*pow(m2, 2)*pow(r1, 3)*pow(theta_dot, 2)*pow(cos(theta), 2)*pow(sin(psi), 3)*sin(theta) + 12*pow(L, 4)*pow(m2, 2)*pow(r1, 2)*pow(theta_dot, 2)*pow(cos(theta), 3)*pow(sin(psi), 2)*sin(theta) + pow(L, 4)*pow(m2, 2)*pow(r1, 2)*pow(theta_dot, 2)*pow(cos(theta), 3)*pow(sin(psi), 4)*sin(theta) - 2*L*m1*m2*pow(r1, 3)*pow(x_dot, 2)*sin(psi)*sin(theta) + pow(L, 4)*pow(m2, 2)*pow(psi_dot, 2)*pow(r1, 2)*pow(cos(psi), 2)*pow(cos(theta), 3)*pow(sin(psi), 2)*sin(theta) + pow(L, 4)*pow(m2, 2)*pow(r1, 2)*pow(theta_dot, 2)*pow(cos(psi), 2)*pow(cos(theta), 3)*pow(sin(psi), 2)*sin(theta) - 16*pow(L, 2)*pow(m2, 2)*pow(r1, 3)*theta_dot*x_dot*cos(theta)*sin(psi)*sin(theta) - pow(L, 3)*g*pow(m2, 2)*pow(r1, 2)*cos(psi)*pow(cos(theta), 2)*pow(sin(psi), 2)*sin(theta) + pow(L, 2)*m1*m2*pow(psi_dot, 2)*pow(r1, 4)*pow(cos(psi), 2)*cos(theta)*sin(theta) + pow(L, 2)*m1*m2*pow(r1, 4)*pow(theta_dot, 2)*pow(cos(psi), 2)*cos(theta)*sin(theta) + 4*pow(L, 2)*g*pow(m2, 2)*pow(r1, 3)*cos(psi)*cos(theta)*sin(psi)*sin(theta) + pow(L, 2)*m1*m2*pow(psi_dot, 2)*pow(r1, 4)*cos(theta)*pow(sin(psi), 2)*sin(theta) + pow(L, 2)*m1*m2*pow(r1, 4)*pow(theta_dot, 2)*cos(theta)*pow(sin(psi), 2)*sin(theta) + 6*pow(L, 3)*m1*m2*pow(r1, 3)*pow(theta_dot, 2)*pow(cos(theta), 2)*sin(psi)*sin(theta) + pow(L, 2)*m1*m2*pow(r1, 2)*pow(x_dot, 2)*cos(theta)*pow(sin(psi), 2)*sin(theta) - 12*pow(L, 4)*pow(m2, 2)*r1*theta_dot*x_dot*pow(cos(theta), 3)*pow(sin(psi), 3)*sin(theta) - 4*pow(L, 3)*m1*m2*psi_dot*pow(r1, 3)*theta_dot*cos(psi)*pow(cos(theta), 3) - 4*pow(L, 2)*m1*m2*psi_dot*pow(r1, 3)*x_dot*cos(psi)*pow(cos(theta), 2) - 4*pow(L, 3)*pow(m2, 2)*pow(psi_dot, 2)*pow(r1, 3)*pow(cos(psi), 2)*pow(cos(theta), 2)*sin(psi)*sin(theta) - 4*pow(L, 3)*pow(m2, 2)*pow(r1, 3)*pow(theta_dot, 2)*pow(cos(psi), 2)*pow(cos(theta), 2)*sin(psi)*sin(theta) - 2*pow(L, 4)*m1*m2*pow(r1, 2)*pow(theta_dot, 2)*pow(cos(theta), 3)*pow(sin(psi), 2)*sin(theta) + 24*pow(L, 3)*pow(m2, 2)*pow(r1, 2)*theta_dot*x_dot*pow(cos(theta), 2)*pow(sin(psi), 2)*sin(theta) + 2*pow(L, 4)*m1*m2*psi_dot*pow(r1, 2)*theta_dot*cos(psi)*pow(cos(theta), 4)*sin(psi) + 2*pow(L, 3)*m1*m2*psi_dot*pow(r1, 2)*x_dot*cos(psi)*pow(cos(theta), 3)*sin(psi))/(m2*pow(r1, 2)*(4*pow(L, 2)*m1*pow(r1, 2)*pow(cos(theta), 2) + pow(L, 4)*m1*pow(cos(theta), 4)*pow(sin(psi), 2) - 4*pow(L, 3)*m1*r1*pow(cos(theta), 3)*sin(psi) + pow(L, 4)*m2*pow(cos(theta), 2)*pow(sin(psi), 4)*pow(sin(theta), 2) + pow(L, 2)*m1*pow(r1, 2)*pow(cos(psi), 2)*pow(sin(theta), 2) + 4*pow(L, 2)*m2*pow(r1, 2)*pow(cos(psi), 2)*pow(sin(theta), 2) + pow(L, 2)*m1*pow(r1, 2)*pow(sin(psi), 2)*pow(sin(theta), 2) + 4*pow(L, 2)*m2*pow(r1, 2)*pow(sin(psi), 2)*pow(sin(theta), 2) - 4*pow(L, 3)*m2*r1*cos(theta)*pow(sin(psi), 3)*pow(sin(theta), 2) + pow(L, 4)*m2*pow(cos(psi), 2)*pow(cos(theta), 2)*pow(sin(psi), 2)*pow(sin(theta), 2) - 4*pow(L, 3)*m2*r1*pow(cos(psi), 2)*cos(theta)*sin(psi)*pow(sin(theta), 2)));
	
	double psi_ddot = -fy*cos(psi)/(m2*pow(L, 2)*pow(cos(theta), 2)) + (pow(r1, 2)*tau_psi + L*g*m2*pow(r1, 2)*cos(theta)*sin(psi) - 2*L*m2*r1*pow(x_dot, 2)*cos(psi)*cos(theta) + pow(L, 4)*m2*pow(theta_dot, 2)*cos(psi)*pow(cos(theta), 4)*sin(psi) + pow(L, 2)*m2*pow(x_dot, 2)*cos(psi)*pow(cos(theta), 2)*sin(psi) - 2*pow(L, 3)*m2*r1*pow(theta_dot, 2)*cos(psi)*pow(cos(theta), 3) + 2*pow(L, 3)*m2*theta_dot*x_dot*cos(psi)*pow(cos(theta), 3)*sin(psi) - 4*pow(L, 2)*m2*r1*theta_dot*x_dot*cos(psi)*pow(cos(theta), 2) + 2*pow(L, 2)*m2*psi_dot*pow(r1, 2)*theta_dot*pow(cos(psi), 2)*cos(theta)*sin(theta) + 2*pow(L, 2)*m2*psi_dot*pow(r1, 2)*theta_dot*cos(theta)*pow(sin(psi), 2)*sin(theta))/(m2*pow(r1, 2)*(pow(L, 2)*pow(cos(psi), 2)*pow(cos(theta), 2) + pow(L, 2)*pow(cos(theta), 2)*pow(sin(psi), 2)));
        

        // Update states using Euler integration 
        xdot += x_ddot*DT;
        phi += phi_dot*DT;
        phi_dot += phi_ddot*DT;
        theta += theta_dot*DT;
        theta_dot += theta_ddot*DT;
        psi += psi_dot*DT;
        psi_dot += psi_ddot*DT;

        // Store current state in global memory 
        state[idx*8 + 0] = x; 
        state[idx*8 + 1] = phi;
        state[idx*8 + 2] = theta; 
        state[idx*8 + 3] = psi;
        state[idx*8 + 4] = xdot; 
        state[idx*8 + 5] = phi_dot;
        state[idx*8 + 6] = theta_dot; 
        state[idx*8 + 7] = psi_dot;
    }

    // Store final state in global memory 
    final_state[idx*8 + 0] = x; 
    final_state[idx*8 + 1] = phi;
    final_state[idx*8 + 2] = theta; 
    final_state[idx*8 + 3] = psi;
    final_state[idx*8 + 4] = xdot;
    final_state[idx*8 + 5] = phi_dot;
    final_state[idx*8 + 6] = theta_dot; 
    final_state[idx*8 + 7] = psi_dot;
}

int main() {
    double *d_state;
    curandState_t *d_states;
    double *d_initial_state;
    double *d_final_state;
    
    int slice_x = 3;
    int slice_y = 4;
    
    cudaError_t error1, error2, error3, error4;
    error1 = cudaMalloc((void**)&d_state, NUM_SIMULATIONS*8*sizeof(double));
    error2 = cudaMalloc((void**)&d_states, NUM_SIMULATIONS*sizeof(curandState_t));
    error3 = cudaMalloc((void**)&d_initial_state, NUM_SIMULATIONS*8*sizeof(double));
    error4 = cudaMalloc((void**)&d_final_state, NUM_SIMULATIONS*8*sizeof(double));
    
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

    simulate <<< (NUM_SIMULATIONS + 255) / 256, 256 >>> (d_states, d_state, d_initial_state, d_final_state, slice_x, slice_y);

    double *h_initial_state = new double[NUM_SIMULATIONS*8];
    double *h_final_state = new double[NUM_SIMULATIONS*8];

    cudaMemcpy(h_initial_state, d_initial_state, NUM_SIMULATIONS*8*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_final_state, d_final_state, NUM_SIMULATIONS*8*sizeof(double), cudaMemcpyDeviceToHost);

    // Write initial states to binary file
    std::ofstream initial_file("initial_states.bin", std::ios::binary);
    initial_file.write(reinterpret_cast<char*>(h_initial_state), NUM_SIMULATIONS*8*sizeof(double));
    initial_file.close();

    // Write final states to binary file
    std::ofstream final_file("final_states.bin", std::ios::binary);
    final_file.write(reinterpret_cast<char*>(h_final_state), NUM_SIMULATIONS*8*sizeof(double));
    final_file.close();

    delete[] h_initial_state;
    delete[] h_final_state;
    cudaFree(d_state);
    cudaFree(d_states);
    cudaFree(d_initial_state);
    cudaFree(d_final_state);
    return 0;
}


