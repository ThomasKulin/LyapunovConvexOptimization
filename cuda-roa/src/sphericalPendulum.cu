#include <stdio.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <fstream>
#include <iostream>

#define NUM_SIMULATIONS 100000
#define NUM_TIMESTEPS 5000
#define DT 0.001

// Pendulum parameters
#define LENGTH 0.9
#define MASS 90.0
#define GRAVITY 9.81


__global__ void init(unsigned int seed, curandState_t* states) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= NUM_SIMULATIONS) return;

    curand_init(seed, idx, 0, &states[idx]);
}

__global__ void simulate(curandState_t* states, double *state, double *initial_state, double *final_state, int slice_x, int slice_y) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= NUM_SIMULATIONS) return;

    // Initial state
    double x = 0;
    double x_dot = 0;
    double y = 0; 
    double y_dot = 0;
    double theta = curand_uniform_double(&states[idx]) * 1 * M_PI - M_PI/2; 
    double theta_dot = (curand_uniform_double(&states[idx]) * 2 - 1)*4; 
    double phi = curand_uniform_double(&states[idx]) * 1 * M_PI - M_PI/2; 
    double phi_dot = (curand_uniform_double(&states[idx]) * 2 - 1)*4; 

    // Store initial state in global memory
    initial_state[idx*8 + 0] = x; 
    initial_state[idx*8 + 1] = y;
    initial_state[idx*8 + 2] = theta; 
    initial_state[idx*8 + 3] = phi;
    initial_state[idx*8 + 4] = x_dot; 
    initial_state[idx*8 + 5] = y_dot;
    initial_state[idx*8 + 6] = theta_dot; 
    initial_state[idx*8 + 7] = phi_dot;
    
    // Set initial states to only vary in 2D slice.
    for(int i = 0; i < 8; i++){
    	if(slice_x == i || slice_y == i){
    	    i = i;
    	}
    	else initial_state[idx*8 + i] = 0;
    }

    for (int t = 0; t < NUM_TIMESTEPS; t++) {
    x =0;
    y=0;
    x_dot=0;
    y_dot=0;
    	
        
        int u_x = (-0.5 * (10 * (0.00065661500017398424 * x - 0.00080225087514233892 * sin(theta) + 0.0016683368797064707 * x_dot + 1.4518680477777806e-06 * y_dot + 0.058978371630366608 * theta_dot - 7.4945293865825948e-06 * (x * y * y_dot) + 7.7210561517206814e-06 * (x * pow(y, 2)) - 2.8029590224201096e-06 * (x * sin(theta) * theta_dot) + 0.029899901170013898 * (x * pow(sin(theta), 2)) - 0.029164959133086626 * (x * cos(theta)) + 5.7488306232284243e-05 * (x * cos(theta) * cos(phi)) + 0.029696757611333963 * (x * pow(cos(theta), 2)) - 0.0080531373932314146 * (x * pow(sin(phi), 2)) + 0.0087232597957820936 * (x * cos(phi)) - 0.0081857188078538982 * (x * pow(cos(phi), 2)) + 2.7924729758463211e-05 * (x * pow(x_dot, 2)) + 1.9588731165035797e-05 * (x * pow(y_dot, 2)) - 3.2106065337478798e-06 * (pow(x, 2) * sin(theta)) + 5.4588930500767238e-06 * (pow(x, 2) * x_dot) + 4.0363821439098707e-05 * (y * x_dot * y_dot) - 6.8772074756823885e-06 * (pow(y, 2) * sin(theta)) + 1.1232416041136309e-05 * (pow(y, 2) * x_dot) - 1.8018385007499595e-05 * (sin(theta) * cos(theta)) + 6.5663170607518304e-05 * (sin(theta) * cos(theta) * cos(phi)) - 0.00026388618879473116 * (sin(theta) * pow(cos(theta), 2)) + 0.00047295387099069073 * (sin(theta) * sin(phi) * x_dot) - 6.6731416346599797e-05 * (sin(theta) * sin(phi) * y_dot) - 0.00028750819826818606 * (sin(theta) * pow(sin(phi), 2)) - 0.00017769742622505838 * (sin(theta) * cos(phi)) - 0.00025973672449486812 * (sin(theta) * pow(cos(phi), 2)) - 0.012472764563654707 * (sin(theta) * x_dot * theta_dot) - 1.4637529555413814e-05 * (sin(theta) * pow(x_dot, 2)) - 1.8618565611693312e-06 * (sin(theta) * pow(y_dot, 2)) - 0.00092641875332006858 * (pow(sin(theta), 2) * x_dot) - 1.1077739242823393e-05 * (pow(sin(theta), 2) * theta_dot) - 0.00028137423669705126 * (cos(theta) * cos(phi) * x_dot) - 3.0391127442231211e-06 * (cos(theta) * cos(phi) * y_dot) + 0.0002866840145857091 * (cos(theta) * cos(phi) * theta_dot) - 1.8264144557303146e-06 * (cos(theta) * cos(phi) * phi_dot) - 0.0024713720098702235 * (cos(theta) * x_dot) + 0.0019392983654265249 * (cos(theta) * theta_dot) + 0.061905386082212861 * (pow(cos(theta), 2) * x_dot) - 1.5576281063658549e-06 * (pow(cos(theta), 2) * phi_dot) + 0.010842375909422547 * (sin(phi) * x_dot * phi_dot) + 0.005378524562402183 * (pow(sin(phi), 2) * x_dot) - 1.3286142264880418e-06 * (pow(sin(phi), 2) * y_dot) + 7.7880449572068395e-05 * (pow(sin(phi), 2) * theta_dot) - 0.001135704775697393 * (cos(phi) * x_dot) + 0.0027139544542427963 * (cos(phi) * theta_dot) + 0.0047724492826776502 * (pow(cos(phi), 2) * x_dot) - 1.3637700748498653e-06 * (pow(cos(phi), 2) * y_dot) + 0.00015874166906438498 * (pow(cos(phi), 2) * theta_dot) + 1.3320124355716667e-05 * (x_dot * pow(y_dot, 2)) - 2.716831918257384e-06 * (x_dot * pow(phi_dot, 2)) + 0.0008026161307993113 * pow(x, 3) - 0.00031558878857442294 * pow(sin(theta), 3) + 1.4378288089182047e-05 * pow(x_dot, 3)) - 10 * (cos(phi) * (0.00080639736390177984 * x + 29.446131439279817 * sin(theta) + 0.058978371630366608 * x_dot + 1.7525671126709179 * theta_dot - 2.8029590224201096e-06 * (x * sin(theta) * x_dot) + 9.3436147681453737e-06 * (x * sin(theta) * theta_dot) - 2.2663027977351189e-05 * (x * pow(sin(theta), 2)) + 0.00038391186437287353 * (x * cos(theta)) + 5.1123602340588504e-05 * (x * cos(theta) * cos(phi)) + 1.1044744818446107e-05 * (x * pow(sin(phi), 2)) + 0.00037854192005954052 * (x * cos(phi)) + 3.1664729168049129e-05 * (x * pow(cos(phi), 2)) - 0.015915599447994946 * (pow(x, 2) * sin(theta)) - 0.00027021318103590263 * (y * sin(theta) * sin(phi)) - 3.8090439539374346e-06 * (y * sin(theta) * y_dot) - 5.9616203613823222e-06 * (y * sin(phi) * theta_dot) - 0.01581004159658456 * (pow(y, 2) * sin(theta)) + 0.21644311142882269 * (sin(theta) * cos(theta)) - 0.25364155855433279 * (sin(theta) * cos(theta) * cos(phi)) - 0.21223719291622942 * (sin(theta) * pow(cos(theta), 2)) - 0.00069493599685916112 * (sin(theta) * sin(phi) * y_dot) - 5.0132245142752832e-05 * (sin(theta) * sin(phi) * theta_dot) - 1.2263243522666615e-06 * (sin(theta) * sin(phi) * phi_dot) - 0.15710343211361938 * (sin(theta) * pow(sin(phi), 2)) + 6.1745542635810269 * (sin(theta) * cos(phi)) - 0.00042751580576567028 * (sin(theta) * pow(cos(phi), 2)) - 0.0062363822818273535 * (sin(theta) * pow(x_dot, 2)) - 0.0082411401038263218 * (sin(theta) * pow(y_dot, 2)) - 0.027838970576979427 * (sin(theta) * pow(theta_dot, 2)) - 0.0041833996164675697 * (sin(theta) * pow(phi_dot, 2)) - 1.1077739242823393e-05 * (pow(sin(theta), 2) * x_dot) - 0.00010568737444342824 * (pow(sin(theta), 2) * theta_dot) + 0.0002866840145857091 * (cos(theta) * cos(phi) * x_dot) - 2.379312529080742e-06 * (cos(theta) * cos(phi) * y_dot) + 8.5205573627874369e-06 * (cos(theta) * cos(phi) * theta_dot) - 5.2846907876329557e-06 * (cos(theta) * cos(phi) * phi_dot) + 0.0019392983654265249 * (cos(theta) * x_dot) - 4.4657768974570682e-05 * (cos(theta) * theta_dot) - 8.313096330193371e-06 * (cos(theta) * phi_dot) + 8.3227127592261156e-06 * (sin(phi) * y_dot * theta_dot) + 0.00078325527287836537 * (sin(phi) * theta_dot * phi_dot) + 7.7880449572068395e-05 * (pow(sin(phi), 2) * x_dot) + 0.0077234722628826344 * (pow(sin(phi), 2) * theta_dot) + 0.0027139544542427963 * (cos(phi) * x_dot) - 1.5913256952373918e-06 * (cos(phi) * y_dot) - 0.00021916007829132103 * (cos(phi) * theta_dot) + 0.00015874166906438498 * (pow(cos(phi), 2) * x_dot) + 0.0076445069932167117 * (pow(cos(phi), 2) * theta_dot) - 0.085543707918508474 * pow(sin(theta), 3)))));

	int u_y = (-0.5 * (10 * ( - 0.00047692517621020735 * y - 0.0069576206227312514 * sin(phi) + 1.4518680477777806e-06 * x_dot - 0.0026245353847030631 * y_dot - 0.00013178514589777339 * phi_dot - 7.4945293865825948e-06 * (x * y * x_dot) - 1.8011278080317973e-05 * (x * sin(theta) * sin(phi)) + 3.9177462330071594e-05 * (x * x_dot * y_dot) + 7.3521510928228101e-06 * (pow(x, 2) * y) + 9.640333341330666e-06 * (pow(x, 2) * y_dot) - 1.7409935759379833e-06 * (pow(x, 2) * phi_dot) - 3.8090439539374346e-06 * (y * sin(theta) * theta_dot) + 0.029272431440145491 * (y * pow(sin(theta), 2)) - 0.02872793935884594 * (y * cos(theta)) + 0.0010916563180772185 * (y * cos(theta) * cos(phi)) + 0.029190133418623435 * (y * pow(cos(theta), 2)) - 0.0080419806516425323 * (y * pow(sin(phi), 2)) + 0.0087040839963470947 * (y * cos(phi)) - 0.0081927309492042921 * (y * pow(cos(phi), 2)) + 2.0181910719549354e-05 * (y * pow(x_dot, 2)) + 2.7892765412627658e-05 * (y * pow(y_dot, 2)) + 4.5188176529300099e-06 * (pow(y, 2) * y_dot) - 2.0960535739799055e-06 * (pow(y, 2) * phi_dot) - 6.6731416346599797e-05 * (sin(theta) * sin(phi) * x_dot) + 0.00031447734864650513 * (sin(theta) * sin(phi) * y_dot) - 0.00069493599685916112 * (sin(theta) * sin(phi) * theta_dot) - 3.7237131223386624e-06 * (sin(theta) * x_dot * y_dot) - 0.016482280207652644 * (sin(theta) * y_dot * theta_dot) - 0.0002681688105832913 * (pow(sin(theta), 2) * sin(phi)) - 0.0024714670883438828 * (pow(sin(theta), 2) * y_dot) + 0.00012878020486873376 * (pow(sin(theta), 2) * phi_dot) - 0.0019166190665451522 * (cos(theta) * sin(phi)) + 0.0053592936741295805 * (cos(theta) * sin(phi) * cos(phi)) - 3.0391127442231211e-06 * (cos(theta) * cos(phi) * x_dot) - 5.409235438254908e-05 * (cos(theta) * cos(phi) * y_dot) - 2.379312529080742e-06 * (cos(theta) * cos(phi) * theta_dot) + 0.0073798038486033125 * (cos(theta) * cos(phi) * phi_dot) - 0.0025897133441517171 * (cos(theta) * y_dot) + 4.3240213824870824e-06 * (cos(theta) * phi_dot) - 0.00086207998345668238 * (pow(cos(theta), 2) * sin(phi)) + 0.00550751538300595 * (pow(cos(theta), 2) * y_dot) + 0.00049736096272567714 * (pow(cos(theta), 2) * phi_dot) + 9.9541166167223659e-05 * (sin(phi) * cos(phi)) + 0.0032246365558445604 * (sin(phi) * pow(cos(phi), 2)) + 0.011861392841316447 * (sin(phi) * y_dot * phi_dot) + 4.1613563796130578e-06 * (sin(phi) * pow(theta_dot, 2)) - 1.3286142264880418e-06 * (pow(sin(phi), 2) * x_dot) + 0.0086856155232263137 * (pow(sin(phi), 2) * y_dot) + 5.2166450471752628e-06 * (pow(sin(phi), 2) * phi_dot) - 0.00010726822382748937 * (cos(phi) * y_dot) - 1.5913256952373918e-06 * (cos(phi) * theta_dot) + 1.373306565675941e-06 * (cos(phi) * phi_dot) - 1.3637700748498653e-06 * (pow(cos(phi), 2) * x_dot) + 0.008361330622479889 * (pow(cos(phi), 2) * y_dot) + 8.5149240266092548e-06 * (pow(cos(phi), 2) * phi_dot) + 1.3320124355716667e-05 * (pow(x_dot, 2) * y_dot) - 2.9256232940476088e-06 * (y_dot * pow(phi_dot, 2)) + 0.00084067209529821137 * pow(y, 3) + 0.0021057423986484869 * pow(sin(phi), 3) + 1.0226513483828438e-05 * pow(y_dot, 3)) + 10 * (sin(theta) * sin(phi) * (0.00080639736390177984 * x + 29.446131439279817 * sin(theta) + 0.058978371630366608 * x_dot + 1.7525671126709179 * theta_dot - 2.8029590224201096e-06 * (x * sin(theta) * x_dot) + 9.3436147681453737e-06 * (x * sin(theta) * theta_dot) - 2.2663027977351189e-05 * (x * pow(sin(theta), 2)) + 0.00038391186437287353 * (x * cos(theta)) + 5.1123602340588504e-05 * (x * cos(theta) * cos(phi)) + 1.1044744818446107e-05 * (x * pow(sin(phi), 2)) + 0.00037854192005954052 * (x * cos(phi)) + 3.1664729168049129e-05 * (x * pow(cos(phi), 2)) - 0.015915599447994946 * (pow(x, 2) * sin(theta)) - 0.00027021318103590263 * (y * sin(theta) * sin(phi)) - 3.8090439539374346e-06 * (y * sin(theta) * y_dot) - 5.9616203613823222e-06 * (y * sin(phi) * theta_dot) - 0.01581004159658456 * (pow(y, 2) * sin(theta)) + 0.21644311142882269 * (sin(theta) * cos(theta)) - 0.25364155855433279 * (sin(theta) * cos(theta) * cos(phi)) - 0.21223719291622942 * (sin(theta) * pow(cos(theta), 2)) - 0.00069493599685916112 * (sin(theta) * sin(phi) * y_dot) - 5.0132245142752832e-05 * (sin(theta) * sin(phi) * theta_dot) - 1.2263243522666615e-06 * (sin(theta) * sin(phi) * phi_dot) - 0.15710343211361938 * (sin(theta) * pow(sin(phi), 2)) + 6.1745542635810269 * (sin(theta) * cos(phi)) - 0.00042751580576567028 * (sin(theta) * pow(cos(phi), 2)) - 0.0062363822818273535 * (sin(theta) * pow(x_dot, 2)) - 0.0082411401038263218 * (sin(theta) * pow(y_dot, 2)) - 0.027838970576979427 * (sin(theta) * pow(theta_dot, 2)) - 0.0041833996164675697 * (sin(theta) * pow(phi_dot, 2)) - 1.1077739242823393e-05 * (pow(sin(theta), 2) * x_dot) - 0.00010568737444342824 * (pow(sin(theta), 2) * theta_dot) + 0.0002866840145857091 * (cos(theta) * cos(phi) * x_dot) - 2.379312529080742e-06 * (cos(theta) * cos(phi) * y_dot) + 8.5205573627874369e-06 * (cos(theta) * cos(phi) * theta_dot) - 5.2846907876329557e-06 * (cos(theta) * cos(phi) * phi_dot) + 0.0019392983654265249 * (cos(theta) * x_dot) - 4.4657768974570682e-05 * (cos(theta) * theta_dot) - 8.313096330193371e-06 * (cos(theta) * phi_dot) + 8.3227127592261156e-06 * (sin(phi) * y_dot * theta_dot) + 0.00078325527287836537 * (sin(phi) * theta_dot * phi_dot) + 7.7880449572068395e-05 * (pow(sin(phi), 2) * x_dot) + 0.0077234722628826344 * (pow(sin(phi), 2) * theta_dot) + 0.0027139544542427963 * (cos(phi) * x_dot) - 1.5913256952373918e-06 * (cos(phi) * y_dot) - 0.00021916007829132103 * (cos(phi) * theta_dot) + 0.00015874166906438498 * (pow(cos(phi), 2) * x_dot) + 0.0076445069932167117 * (pow(cos(phi), 2) * theta_dot) - 0.085543707918508474 * pow(sin(theta), 3))) + 10 * (( - 5.0622906700366539e-05 * y + 0.04182671115568963 * sin(phi) - 0.00013178514589777339 * y_dot + 0.0038148629944226159 * phi_dot - 1.4985353421805451e-06 * (pow(x, 2) * y) + 0.0076269437393572132 * (pow(x, 2) * sin(phi)) - 1.7409935759379833e-06 * (pow(x, 2) * y_dot) - 3.8909911968601225e-06 * (pow(x, 2) * phi_dot) + 4.8537142473336263e-05 * (y * pow(sin(theta), 2)) - 1.0589061682491498e-05 * (y * cos(theta)) + 0.001103085083767813 * (y * cos(theta) * cos(phi)) + 0.00038906719004088163 * (y * pow(cos(theta), 2)) + 1.8653183506085377e-06 * (y * pow(sin(phi), 2)) + 1.1968731446424174e-06 * (y * cos(phi)) + 1.5634375792757255e-06 * (y * pow(cos(phi), 2)) + 0.0077158572212794856 * (pow(y, 2) * sin(phi)) - 2.0960535739799055e-06 * (pow(y, 2) * y_dot) - 3.0517481681388179e-06 * (pow(y, 2) * phi_dot) - 1.2263243522666615e-06 * (sin(theta) * sin(phi) * theta_dot) - 0.0083667992329351393 * (sin(theta) * theta_dot * phi_dot) + 0.0023831149242609677 * (pow(sin(theta), 2) * sin(phi)) + 0.00012878020486873376 * (pow(sin(theta), 2) * y_dot) - 0.003818122186671219 * (pow(sin(theta), 2) * phi_dot) + 6.1768828266613713 * (cos(theta) * sin(phi)) + 0.33199496574342274 * (cos(theta) * sin(phi) * cos(phi)) - 1.8264144557303146e-06 * (cos(theta) * cos(phi) * x_dot) + 0.0073798038486033125 * (cos(theta) * cos(phi) * y_dot) - 5.2846907876329557e-06 * (cos(theta) * cos(phi) * theta_dot) - 3.5325770867431863e-05 * (cos(theta) * cos(phi) * phi_dot) + 4.3240213824870824e-06 * (cos(theta) * y_dot) - 8.313096330193371e-06 * (cos(theta) * theta_dot) - 2.5932781565153513e-05 * (cos(theta) * phi_dot) + 26.099774846111259 * (pow(cos(theta), 2) * sin(phi)) - 1.5576281063658549e-06 * (pow(cos(theta), 2) * x_dot) + 0.00049736096272567714 * (pow(cos(theta), 2) * y_dot) + 1.1114550592934673 * (pow(cos(theta), 2) * phi_dot) + 0.0002127959564886226 * (sin(phi) * cos(phi)) + 0.0034202795058604508 * (sin(phi) * pow(cos(phi), 2)) + 0.0054211879547112736 * (sin(phi) * pow(x_dot, 2)) + 0.0059306964206582236 * (sin(phi) * pow(y_dot, 2)) + 0.00039162763643918268 * (sin(phi) * pow(theta_dot, 2)) + 0.0025691718425845862 * (sin(phi) * pow(phi_dot, 2)) + 5.2166450471752628e-06 * (pow(sin(phi), 2) * y_dot) - 2.5640267418186254e-06 * (pow(sin(phi), 2) * phi_dot) + 1.373306565675941e-06 * (cos(phi) * y_dot) + 8.5149240266092548e-06 * (pow(cos(phi), 2) * y_dot) - 2.716831918257384e-06 * (pow(x_dot, 2) * phi_dot) - 2.9256232940476088e-06 * (pow(y_dot, 2) * phi_dot) + 0.023212484184623197 * pow(sin(phi), 3)) * ((-1 * cos(phi)) / cos(theta)))));
	
        // Calculate second derivatives based on equations of motion
        double x_ddot = u_x;
        double y_ddot = u_y;
        double theta_ddot = -cos(theta)*sin(theta)*phi_dot*phi_dot - GRAVITY*cos(phi)*sin(theta)/LENGTH - u_x*cos(theta)/LENGTH + u_y*sin(phi)*sin(theta)/LENGTH;
        double phi_ddot = (GRAVITY*sin(phi) + 2*LENGTH*phi_dot*theta_dot*sin(theta) - u_y*cos(phi))/(LENGTH*cos(theta));
        

        // Update states using Euler integration - for improved stability you may want to switch to a more sophisticated integration scheme
        x += x_dot*DT;
        x_dot += x_ddot*DT;
        y += y_dot*DT;
        y_dot += y_ddot*DT;
        theta += theta_dot*DT;
        theta_dot += theta_ddot*DT;
        phi += phi_dot*DT;
        phi_dot += phi_ddot*DT;

        // Store current state in global memory 
        state[idx*8 + 0] = x; 
        state[idx*8 + 1] = y;
        state[idx*8 + 2] = theta; 
        state[idx*8 + 3] = phi;
        state[idx*8 + 4] = x_dot; 
        state[idx*8 + 5] = y_dot;
        state[idx*8 + 6] = theta_dot; 
        state[idx*8 + 7] = phi_dot;
    }

    // Store final state in global memory 
    final_state[idx*8 + 0] = x; 
    final_state[idx*8 + 1] = y;
    final_state[idx*8 + 2] = theta; 
    final_state[idx*8 + 3] = phi;
    final_state[idx*8 + 4] = x_dot; 
    final_state[idx*8 + 5] = y_dot;
    final_state[idx*8 + 6] = theta_dot; 
    final_state[idx*8 + 7] = phi_dot;
}

int main() {
    double *d_state;
    curandState_t *d_states;
    double *d_initial_state;
    double *d_final_state;
    
    int slice_x = 2;
    int slice_y = 6;
    
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


