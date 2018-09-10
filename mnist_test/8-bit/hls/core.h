#ifndef CORE_H
#define CORE_H

#include "../../../hls_nnet_lib/nnet.h"

#include <ap_axi_sdata.h>
#include <ap_fixed.h>
typedef ap_axis<32,2,5,6> int_32_side_channel__;

const int fixed_factor = 128;//2^7
typedef ap_uint<64> wide_var;

typedef ap_fixed<8,1,AP_RND,AP_SAT> fixed_point_8_7;
typedef ap_fixed<8,3,AP_RND,AP_SAT> fixed_point_8_5;
typedef ap_fixed<8,5,AP_RND,AP_SAT> fixed_point_8_3;
typedef ap_fixed<24,12,AP_RND,AP_SAT> fixed_point_wide;


#define KERNEL_DIM 5
#define ZERO_PADDING     (KERNEL_DIM-1)/2

#define INPUT_IMAGE_SIZE      			784
#define INPUT_IMAGE_WIDTH      			28
#define INPUT_IMAGE_HEIGHT      		28

#define FIRST_CONV_INPUT_CHANNELS  		1
#define FIRST_CONV_LAYER_OUTPUT_DEPTH  	32
#define FIRST_CONV_LAYER_FILTER_SIZE 	5
#define FIRST_CONV_LAYER_POOLED_WIDTH 	14
#define FIRST_CONV_LAYER_POOLED_HEIGHT 	14

#define SECOND_CONV_INPUT_CHANNELS  	FIRST_CONV_LAYER_OUTPUT_DEPTH
#define SECOND_CONV_LAYER_OUTPUT_DEPTH	64
#define SECOND_CONV_LAYER_FILTER_SIZE 	5
#define SECOND_CONV_LAYER_POOLED_WIDTH 	7
#define SECOND_CONV_LAYER_POOLED_HEIGHT	7

#define FIRST_FC_INPUT_SIZE 			SECOND_CONV_LAYER_POOLED_WIDTH * SECOND_CONV_LAYER_POOLED_HEIGHT * SECOND_CONV_LAYER_OUTPUT_DEPTH
#define FIRST_FC_OUTPUT_SIZE 			512

#define SECOND_FC_INPUT_SIZE 			FIRST_FC_OUTPUT_SIZE
#define SECOND_FC_OUTPUT_SIZE 			10

#define OUTPUT_CLASSES 					SECOND_FC_OUTPUT_SIZE


void deepMNIST(hls::stream<fixed_point_8_7> &inStream,
		hls::stream<int_32_side_channel__> &outStream,
		hls::stream<wide_var> &weightStream
);


#endif // CORE_H
