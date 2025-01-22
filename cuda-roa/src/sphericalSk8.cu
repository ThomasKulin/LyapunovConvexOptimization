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
    	
    	double x_dot = 0;
    	double x = 0;
    	
    	// Ankle torque inputs
        double tau_theta = (-(Kpprop + Kpv) * theta - (Kdprop + Kdv) * theta_dot);
        double tau_psi = (-Kpy * (psi - Wprop * phi) - Kdy * (psi_dot - Wprop * phi_dot) - Ky * (psi - phi) - Cy * (psi_dot - phi_dot));
        double tau_phi = (-tau_psi);
    	
        // Control Inputs
        double y = 0;
        double y_dot = 0;
        
        double u_x = (-0.5 * (10 * ( - 0.0037159116747687375 * x + 1.4023738844608461e-05 * sin(theta) + 0.38472287336981148 * x_dot + 0.27112989963130152 * theta_dot - 1.7220370871065135e-05 * (x * y * y_dot) - 4.0252395016612092e-06 * (x * sin(theta) * theta_dot) + 0.0042719825773443257 * (x * pow(sin(theta),2)) - 0.0049244807251479427 * (x * cos(theta)) - 1.0510749271845307e-05 * (x * cos(theta) * cos(psi)) + 0.0042370590862427868 * (x * pow(cos(theta),2)) - 0.003104062055035666 * (x * pow(sin(psi),2)) + 0.0023773919568441408 * (x * cos(psi)) - 0.0031373531446148291 * (x * pow(cos(psi),2)) - 0.00012289624398264913 * (x * pow(x_dot,2)) - 8.6355221724694918e-06 * (x * pow(y_dot,2)) - 8.7588547593900578e-06 * (x * pow(psi_dot,2)) - 8.452460209171403e-06 * (x*x * sin(theta)) + 1.4423248254671469e-05 * (x*x * x_dot) - 1.7601884942173404e-05 * (y * x_dot * y_dot) - 1.1092314675350877e-05 * (y*y * sin(theta)) + 4.0383728288752354e-05 * (y*y * x_dot) + 5.4168133222958632e-05 * (sin(theta) * cos(theta)) + 7.3075937429615515e-06 * (sin(theta) * cos(theta) * cos(psi)) - 2.9565660650770546e-05 * (sin(theta) * pow(cos(theta), 2)) + 9.3421235284090739e-05 * (sin(theta) * sin(psi) * x_dot) - 1.3084340361366297e-05 * (sin(theta) * sin(psi) * y_dot) - 2.5181896151793497e-05 * (sin(theta) * pow(sin(psi),2)) - 2.4397597367843353e-05 * (sin(theta) * cos(psi)) - 1.7655938616705257e-05 * (sin(theta) * pow(cos(psi),2)) - 0.0027563307022754175 * (sin(theta) * x_dot * theta_dot) - 2.4614930642655812e-05 * (sin(theta) * pow(x_dot,2)) - 2.9785274336034646e-06 * (sin(theta) * pow(y_dot,2)) - 0.012107776175232114 * (pow(sin(theta),2) * x_dot) - 2.7906342705792594e-05 * (cos(theta) * cos(psi) * x_dot) - 1.2917540354345276e-06 * (cos(theta) * cos(psi) * y_dot) + 1.9496432809408566e-05 * (cos(theta) * cos(psi) * theta_dot) - 0.0001937547536280703 * (cos(theta) * x_dot) + 0.00032732819239863933 * (cos(theta) * theta_dot) - 0.008865409217548078 * (pow(cos(theta), 2) * x_dot) + 0.0028698390138483403 * (sin(psi) * x_dot * psi_dot) + 0.00096397869722262284 * (pow(sin(psi), 2) * x_dot) + 5.6356784654603626e-06 * (pow(sin(psi), 2) * theta_dot) - 0.00015594425373812305 * (cos(psi) * x_dot) + 0.00032181054116067146 * (cos(psi) * theta_dot) + 0.00086759526916253924 * (pow(cos(psi), 2) * x_dot) + 2.0115618697169719e-05 * (pow(cos(psi), 2) * theta_dot) + 0.00021314436129659439 * (x_dot * pow(y_dot, 2)) - 7.7738727603666289e-05 * pow(sin(theta), 3) + 0.015660473116175109 * pow(x_dot, 3)) + 10 * (( - 1.2997198165522089e-06 * x + 27.581337901855274 * sin(theta) + 0.27112989963130152 * x_dot + 1.6000911506269773 * theta_dot + 4.5719187732948569e-06 * psi_dot - 4.0252395016612092e-06 * (x * sin(theta) * x_dot) + 1.0269565595430621e-05 * (x * cos(theta)) + 1.7005164305446174e-06 * (x * cos(theta) * cos(psi)) + 8.5921783417120885e-06 * (x * cos(psi)) + 1.7494724577622071e-06 * (x * pow(cos(psi), 2)) - 0.0025384864091727733 * (pow(x, 2) * sin(theta)) - 1.8530064808273711e-06 * (y * sin(theta) * sin(psi)) - 3.9957213977432063e-06 * (y * sin(theta) * y_dot) - 0.0025762932757968848 * (pow(y, 2) * sin(theta)) + 0.044608011972154776 * (sin(theta) * cos(theta)) - 0.054946853062034694 * (sin(theta) * cos(theta) * cos(psi)) - 0.043255098335273183 * (sin(theta) * pow(cos(theta), 2)) - 6.4791567941641426e-05 * (sin(theta) * sin(psi) * y_dot) - 2.385755324515651e-05 * (sin(theta) * sin(psi) * theta_dot) - 0.026575608823083983 * (sin(theta) * pow(sin(psi), 2)) + 0.92703396801288029 * (sin(theta) * cos(psi)) - 0.00038492409547887088 * (sin(theta) * pow(cos(psi), 2)) - 0.0013781653511377087 * (sin(theta) * pow(x_dot, 2)) - 0.0014510142966381585 * (sin(theta) * pow(y_dot, 2)) - 0.0051287866047923973 * (sin(theta) * pow(theta_dot, 2)) - 0.00065873514589530487 * (sin(theta) * pow(psi_dot, 2)) - 1.4809367694594903e-05 * (pow(sin(theta), 2) * theta_dot) + 1.9496432809408566e-05 * (cos(theta) * cos(psi) * x_dot) + 8.7176572664917901e-06 * (cos(theta) * cos(psi) * theta_dot) - 1.3172505300270798e-06 * (cos(theta) * cos(psi) * psi_dot) + 0.00032732819239863933 * (cos(theta) * x_dot) + 3.1736092768038269e-05 * (sin(psi) * theta_dot * psi_dot) + 5.6356784654603626e-06 * (pow(sin(psi), 2) * x_dot) + 0.0007195122059124446 * (pow(sin(psi), 2) * theta_dot) + 0.00032181054116067146 * (cos(psi) * x_dot) - 2.7861824789980237e-05 * (cos(psi) * theta_dot) + 2.0115618697169719e-05 * (pow(cos(psi), 2) * x_dot) + 0.00072030105108404358 * (pow(cos(psi), 2) * theta_dot) - 0.01426770024873722 * pow(sin(theta), 3)) * ((-1 * cos(psi)) / 0.90000000000000002))));
        
        
double u_y = (-0.5 * (10 * ( - 0.0043962222008269323 * y - 0.002248924052850554 * sin(psi) + 0.071759280895727889 * y_dot - 0.00019966948021526342 * psi_dot + 1.7582084466641341e-06 * (x * y * sin(theta)) - 1.7220370871065135e-05 * (x * y * x_dot) - 1.7271044344938984e-05 * (x * x_dot * y_dot) - 1.3190444673823374e-06 * (pow(x, 2) * y) + 4.585527616619425e-05 * (pow(x, 2) * y_dot) - 3.9957213977432063e-06 * (y * sin(theta) * theta_dot) + 0.0045599469538505485 * (y * pow(sin(theta), 2)) - 0.0050026761822693335 * (y * cos(theta)) + 1.4491953998406169e-05 * (y * cos(theta) * cos(psi)) + 0.0045519605450496051 * (y * pow(cos(theta), 2)) - 0.0031386414791052535 * (y * pow(sin(psi), 2)) + 0.0022224811281667512 * (y * cos(psi)) - 0.0031727033055412926 * (y * pow(cos(psi), 2)) - 8.8009424710867018e-06 * (y * pow(x_dot, 2)) - 0.00033402252575842606 * (y * pow(y_dot, 2)) - 1.2372475833795746e-05 * (y * pow(psi_dot, 2)) + 9.6422026586060816e-06 * (pow(y, 2) * y_dot) - 1.3084340361366297e-05 * (sin(theta) * sin(psi) * x_dot) + 0.00010209687382108834 * (sin(theta) * sin(psi) * y_dot) - 6.4791567941641426e-05 * (sin(theta) * sin(psi) * theta_dot) - 5.9570548672069293e-06 * (sin(theta) * x_dot * y_dot) - 0.002902028593276317 * (sin(theta) * y_dot * theta_dot) - 4.3156400162610565e-05 * (pow(sin(theta), 2) * sin(psi)) - 0.0022829123890516852 * (pow(sin(theta), 2) * y_dot) + 0.00019566304440390896 * (pow(sin(theta), 2) * psi_dot) - 0.00038059119890682358 * (cos(theta) * sin(psi)) + 0.0018779788053770168 * (cos(theta) * sin(psi) * cos(psi)) - 1.2917540354345276e-06 * (cos(theta) * cos(psi) * x_dot) - 3.9395805320193342e-05 * (cos(theta) * cos(psi) * y_dot) + 0.15690804640814937 * (cos(theta) * cos(psi) * psi_dot) - 0.00021568349551485277 * (cos(theta) * y_dot) - 1.1636585817008835e-05 * (cos(theta) * psi_dot) - 0.00023791607781748568 * (pow(cos(theta), 2) * sin(psi)) + 0.0013743091027173418 * (pow(cos(theta), 2) * y_dot) - 8.1262551330892475e-05 * (pow(cos(theta), 2) * psi_dot) + 3.590297466657987e-05 * (sin(psi) * cos(psi)) + 0.00086668225652336214 * (sin(psi) * pow(cos(psi), 2)) + 0.0031083567383904751 * (sin(psi) * y_dot * psi_dot) + 0.17407459229813302 * (pow(sin(psi), 2) * y_dot) - 0.00012371090560908671 * (cos(psi) * y_dot) - 2.1926708815370802e-06 * (cos(psi) * psi_dot) + 0.17399693699167884 * (pow(cos(psi), 2) * y_dot) - 1.8502608782575985e-06 * (pow(cos(psi), 2) * psi_dot) + 0.00021314436129659439 * (pow(x_dot, 2) * y_dot) - 3.364545005721065e-06 * (y_dot * pow(psi_dot, 2)) + 1.0331161589917214e-05 * pow(y, 3) + 0.00056474831854676428 * pow(sin(psi), 3) + 0.01771415997211381 * pow(y_dot, 3) - 2.4909844859246939e-06 * pow(psi_dot, 3)) + 10 * (( - 1.2997198165522089e-06 * x + 27.581337901855274 * sin(theta) + 0.27112989963130152 * x_dot + 1.6000911506269773 * theta_dot + 4.5719187732948569e-06 * psi_dot - 4.0252395016612092e-06 * (x * sin(theta) * x_dot) + 1.0269565595430621e-05 * (x * cos(theta)) + 1.7005164305446174e-06 * (x * cos(theta) * cos(psi)) + 8.5921783417120885e-06 * (x * cos(psi)) + 1.7494724577622071e-06 * (x * pow(cos(psi), 2)) - 0.0025384864091727733 * (pow(x, 2) * sin(theta)) - 1.8530064808273711e-06 * (y * sin(theta) * sin(psi)) - 3.9957213977432063e-06 * (y * sin(theta) * y_dot) - 0.0025762932757968848 * (pow(y, 2) * sin(theta)) + 0.044608011972154776 * (sin(theta) * cos(theta)) - 0.054946853062034694 * (sin(theta) * cos(theta) * cos(psi)) - 0.043255098335273183 * (sin(theta) * pow(cos(theta), 2)) - 6.4791567941641426e-05 * (sin(theta) * sin(psi) * y_dot) - 2.385755324515651e-05 * (sin(theta) * sin(psi) * theta_dot) - 0.026575608823083983 * (sin(theta) * pow(sin(psi), 2)) + 0.92703396801288029 * (sin(theta) * cos(psi)) - 0.00038492409547887088 * (sin(theta) * pow(cos(psi), 2)) - 0.0013781653511377087 * (sin(theta) * pow(x_dot, 2)) - 0.0014510142966381585 * (sin(theta) * pow(y_dot, 2)) - 0.0051287866047923973 * (sin(theta) * pow(theta_dot, 2)) - 0.00065873514589530487 * (sin(theta) * pow(psi_dot, 2)) - 1.4809367694594903e-05 * (pow(sin(theta), 2) * theta_dot) + 1.9496432809408566e-05 * (cos(theta) * cos(psi) * x_dot) + 8.7176572664917901e-06 * (cos(theta) * cos(psi) * theta_dot) - 1.3172505300270798e-06 * (cos(theta) * cos(psi) * psi_dot) + 0.00032732819239863933 * (cos(theta) * x_dot) + 3.1736092768038269e-05 * (sin(psi) * theta_dot * psi_dot) + 5.6356784654603626e-06 * (pow(sin(psi), 2) * x_dot) + 0.0007195122059124446 * (pow(sin(psi), 2) * theta_dot) + 0.00032181054116067146 * (cos(psi) * x_dot) - 2.7861824789980237e-05 * (cos(psi) * theta_dot) + 2.0115618697169719e-05 * (pow(cos(psi), 2) * x_dot) + 0.00072030105108404358 * (pow(cos(psi), 2) * theta_dot) - 0.01426770024873722 * pow(sin(theta), 3)) * ((sin(theta) * sin(psi)) / 0.90000000000000002)) + 10 * ((1.2374641169535167e-06 * y + 0.0072675567832205844 * sin(psi) - 0.00019966948021526342 * y_dot + 4.5719187732948569e-06 * theta_dot + 0.00027023618734189778 * psi_dot - 1.7517709518780116e-05 * (x * x_dot * psi_dot) + 0.001744324528574501 * (pow(x, 2) * sin(psi)) - 3.7079895394481251e-06 * (y * cos(theta)) + 8.1379632711054879e-06 * (y * cos(theta) * cos(psi)) - 1.1409523020176136e-05 * (y * pow(cos(theta), 2)) - 2.4744951667591491e-05 * (y * y_dot * psi_dot) + 0.001688877450676342 * (pow(y, 2) * sin(psi)) - 0.0013174702917906097 * (sin(theta) * theta_dot * psi_dot) + 0.00034960778549953184 * (pow(sin(theta), 2) * sin(psi)) + 0.00019566304440390896 * (pow(sin(theta), 2) * y_dot) - 0.00036809781364749493 * (pow(sin(theta), 2) * psi_dot) + 0.92760405122556688 * (cos(theta) * sin(psi)) + 0.055952028469535335 * (cos(theta) * sin(psi) * cos(psi)) + 0.15690804640814937 * (cos(theta) * cos(psi) * y_dot) - 1.3172505300270798e-06 * (cos(theta) * cos(psi) * theta_dot) - 2.9314406464880309e-05 * (cos(theta) * cos(psi) * psi_dot) - 1.1636585817008835e-05 * (cos(theta) * y_dot) - 2.0785746565371699e-05 * (cos(theta) * psi_dot) + 28.585430040258345 * (pow(cos(theta), 2) * sin(psi)) - 8.1262551330892475e-05 * (pow(cos(theta), 2) * y_dot) + 1.2938850354523483 * (pow(cos(theta), 2) * psi_dot) + 2.4022144592087908e-05 * (sin(psi) * cos(psi)) + 0.0005349322902368465 * (sin(psi) * pow(cos(psi), 2)) + 0.0014349195069241702 * (sin(psi) * pow(x_dot, 2)) + 0.0015541783691952376 * (sin(psi) * pow(y_dot, 2)) + 1.5868046384019135e-05 * (sin(psi) * pow(theta_dot, 2)) + 0.00078273009193682679 * (sin(psi) * pow(psi_dot, 2)) - 4.129815863385302e-06 * (pow(sin(psi), 2) * psi_dot) - 2.1926708815370802e-06 * (cos(psi) * y_dot) - 1.4803636876432956e-05 * (cos(psi) * psi_dot) - 1.8502608782575985e-06 * (pow(cos(psi), 2) * y_dot) - 1.1500954849239717e-05 * (pow(cos(psi), 2) * psi_dot) - 7.4729534577740817e-06 * (y_dot * pow(psi_dot, 2)) - 3.364545005721065e-06 * (pow(y_dot, 2) * psi_dot) + 0.0064695651695254597 * pow(sin(psi), 3)) * ((-1 * cos(psi)) / (0.90000000000000002 * cos(theta))))));	

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


