
#include <stdio.h>
#include <string.h>

#include "core.h"
#include "weights_biases.h"

// dataflow region #1, includes conv layers
void block1(hls::stream<fixed_point_8_7> &inStream,
		hls::stream<fixed_point_8_5> &outStream,
		hls::stream<wide_var> &weightStream)
{

#pragma HLS DATAFLOW
	hls::stream<fixed_point_8_5> tmpStream1;
	hls::stream<fixed_point_8_5> tmpStream2;
	hls::stream<fixed_point_8_5> tmpStream3;
	hls::stream<fixed_point_8_5> tmpStream4;
	hls::stream<fixed_point_8_5> tmpStream5;

#ifndef __SYNTHESIS__
	printf("======= CONV1 =======\n");
#endif
	nnet::conv2d_localweights
	<
	fixed_point_8_7,
	fixed_point_wide,
	fixed_point_8_7,
	fixed_point_8_7,
	fixed_point_8_5,
	INPUT_IMAGE_WIDTH,
	INPUT_IMAGE_HEIGHT,
	FIRST_CONV_INPUT_CHANNELS,
	FIRST_CONV_LAYER_OUTPUT_DEPTH,
	ZERO_PADDING,
	KERNEL_DIM
	>
	(weights_conv1,biases_conv1,inStream,tmpStream1);

#ifndef __SYNTHESIS__
	printf("======= RELU1 =======\n");
#endif
	nnet::relu<fixed_point_8_5,fixed_point_8_5,FIRST_CONV_LAYER_OUTPUT_DEPTH*INPUT_IMAGE_WIDTH*INPUT_IMAGE_HEIGHT>(tmpStream1,tmpStream2);

#ifndef __SYNTHESIS__
	printf("======= POOL1 =======\n");
#endif
	nnet::maxpool_2x<fixed_point_8_5, INPUT_IMAGE_WIDTH, INPUT_IMAGE_HEIGHT, FIRST_CONV_LAYER_OUTPUT_DEPTH>(tmpStream2,tmpStream3);

#ifndef __SYNTHESIS__
	printf("======= CONV2 =======\n");
#endif
	nnet::conv2d_stream
	<
	fixed_point_8_5,
	fixed_point_wide,
	fixed_point_8_7,
	fixed_point_8_7,
	fixed_point_8_5,
	wide_var,
	FIRST_CONV_LAYER_POOLED_WIDTH,
	FIRST_CONV_LAYER_POOLED_HEIGHT,
	SECOND_CONV_INPUT_CHANNELS,
	SECOND_CONV_LAYER_OUTPUT_DEPTH,
	ZERO_PADDING,
	KERNEL_DIM
	>
	(weightStream,biases_conv2,tmpStream3,tmpStream4);

#ifndef __SYNTHESIS__
	printf("======= RELU2 =======\n");
#endif
	nnet::relu<fixed_point_8_5,fixed_point_8_5,SECOND_CONV_LAYER_OUTPUT_DEPTH*FIRST_CONV_LAYER_POOLED_WIDTH*FIRST_CONV_LAYER_POOLED_HEIGHT>(tmpStream4,tmpStream5);

#ifndef __SYNTHESIS__
	printf("======= POOL2 =======\n");
#endif
	nnet::maxpool_2x<fixed_point_8_5, FIRST_CONV_LAYER_POOLED_WIDTH, FIRST_CONV_LAYER_POOLED_HEIGHT, SECOND_CONV_LAYER_OUTPUT_DEPTH>(tmpStream5,outStream);

}

// dataflow region #2, including fc layers
void block2(hls::stream<fixed_point_8_5> &inStream, hls::stream<fixed_point_8_3> &outStream,
		hls::stream<wide_var> &weightStream)
{
#pragma HLS DATAFLOW
	hls::stream<fixed_point_8_5> tmpStream7, tmpStream8;

#ifndef __SYNTHESIS__
	printf("======= FC1 with RELU =======\n");
#endif
	nnet::fc_layer_stream
	<
	fixed_point_8_5,
	fixed_point_wide,
	fixed_point_8_7,
	fixed_point_8_7,
	fixed_point_8_5,
	wide_var,
	FIRST_FC_INPUT_SIZE,
	FIRST_FC_OUTPUT_SIZE>
	(inStream,tmpStream7,biases_fc1,weightStream);

	nnet::relu<fixed_point_8_5,fixed_point_8_5,FIRST_FC_OUTPUT_SIZE>(tmpStream7,tmpStream8);

#ifndef __SYNTHESIS__
	printf("======= FC2 =======\n");
#endif
	nnet::compute_layer
	<fixed_point_8_5,
	fixed_point_wide,
	fixed_point_8_7,
	fixed_point_8_7,
	fixed_point_8_3,
	SECOND_FC_INPUT_SIZE,SECOND_FC_OUTPUT_SIZE>	(tmpStream8,outStream,weights_fc2,biases_fc2);

}

template
<class mem_T, int SZ1,int SZ2>
void split(hls::stream<mem_T> &in,hls::stream<mem_T> &out1,hls::stream<mem_T> &out2)
{
	for (int i=0;i<(SZ1+SZ2);i++)
	{
#pragma HLS PIPELINE
		mem_T tmp = in.read();
		if (i<SZ1)
			out1 << tmp;
		else
			out2 << tmp;
	}
}

void deepMNIST(hls::stream<fixed_point_8_7> &inStream,
		hls::stream<int_32_side_channel__> &outStream,
		hls::stream<wide_var> &weightStream
)
{
#pragma HLS INTERFACE axis port=inStream
#pragma HLS INTERFACE axis port=outStream
#pragma HLS INTERFACE axis port=weightStream
#pragma HLS INTERFACE s_axilite port=return bundle=CRTL_BUS

	hls::stream<fixed_point_8_5> tmp1;
	hls::stream<fixed_point_8_3> tmp2;

#pragma HLS DATAFLOW

#pragma HLS STREAM variable=tmp1 depth=3136 dim=1
#pragma HLS STREAM variable=tmp2 depth=10 dim=1

	hls::stream<wide_var> weight_block1,weight_block2;
	split<wide_var,51200/8,1605632/8>(weightStream,weight_block1,weight_block2);

	block1(inStream,tmp1,weight_block1);
	block2(tmp1,tmp2,weight_block2);

	const int outputsize = 10;
	for (int i = 0; i< outputsize; i++)
	{
		//#pragma HLS PIPELINE
		int_32_side_channel__ outstruct;
		outstruct.keep = (1<<sizeof(int))-1;
		outstruct.strb = (1<<sizeof(int))-1; 	//required for AXI DMA 4bytes
		outstruct.user = 0;
		outstruct.id = 0;
		outstruct.dest = 0;

		if (i == outputsize - 1)
			outstruct.last = 1;
		else
			outstruct.last = 0;

		float output = tmp2.read().to_float();

		outstruct.data = output* (float)1048576;// * 2^20 because using 32/20 fixed point
		outStream << outstruct;
	}
}
