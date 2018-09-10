//
//    rfnoc-hls-neuralnet: Vivado HLS code for neural-net building blocks
//
//    Copyright (C) 2017 EJ Kreinar
//
//    This program is free software: you can redistribute it and/or modify
//    it under the terms of the GNU General Public License as published by
//    the Free Software Foundation, either version 3 of the License, or
//    (at your option) any later version.
//
//    This program is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU General Public License
//    along with this program.  If not, see <http://www.gnu.org/licenses/>.
//


#ifndef NNET_LAYER_H_
#define NNET_LAYER_H_

#include "nnet_default.h"
#include "hls_stream.h"

#define MAX_BURST_SIZE 256

namespace nnet {

// Streamed weights FC layer

//template<class input_T, short IN_SIZE, short OUT_SIZE>
//void fc_layer_stream(hls::stream<input_T> &inStream, hls::stream<input_T> &outStream, input_T biases[OUT_SIZE], hls::stream<input_T> &inStreamWeights)
//{
//	input_T temp_weight[OUT_SIZE];
//	input_T data_cache;
//	input_T acc[OUT_SIZE];
//
//#pragma HLS ARRAY_PARTITION variable=temp_weight cyclic factor=16 dim=1
//#pragma HLS ARRAY_PARTITION variable=acc cyclic factor=16 dim=1
//#pragma HLS RESOURCE variable=acc core=RAM_2P_LUTRAM
//
//	Reset: for(int iacc = 0; iacc < OUT_SIZE; iacc++)
//	{
//#pragma HLS UNROLL factor=16
//		acc[iacc] = 0;
//	}
//
//	NewInput:
////	printf("%d ==============\r\n", OUT_SIZE*sizeof(input_T));
//	for(int ii = 0; ii < IN_SIZE; ii++)
//	{
//		int offset = ii * OUT_SIZE;
////		printf("%d ===== \t %d =========\r\n", offset, mem_addr+offset);
////		memcpy(temp_weight,(const input_T*)(mem_addr+offset),OUT_SIZE*sizeof(input_T));
//		nnet::streamStore<fixed_point,OUT_SIZE>(inStreamWeights,temp_weight);
//
//		data_cache = inStream.read();
//		Product: for(int jj = 0; jj < OUT_SIZE; jj++)
//		{
//#pragma HLS UNROLL factor=16
//#pragma HLS PIPELINE
//			acc[jj] += data_cache * temp_weight[jj];
//		}
//	}
//
//	Result: for(int ires = 0; ires < OUT_SIZE; ires++)
//	{
//#pragma HLS PIPELINE
//		input_T tmp = (input_T) (acc[ires] + (input_T) biases[ires]);
//		outStream << tmp;
////		if (ires < 64)
////			printf("%f\t",(float)tmp);
//
//	}
////	printf("\n");
//}



// DRAM weights FC layer - stream

template<
class input_T,
class acc_T,
class weight_T,
class bias_T,
class output_T,
class mem_T,
int IN_SIZE,
int OUT_SIZE>
void fc_layer_stream(
		hls::stream<input_T> &inStream,
		hls::stream<output_T> &outStream,
		bias_T biases[OUT_SIZE],
		hls::stream<mem_T> &weightStream)
{
	mem_T temp_weights[OUT_SIZE/8];
	input_T data_cache;
	acc_T acc[OUT_SIZE];

#pragma HLS ARRAY_PARTITION variable=temp_weights cyclic factor=16 dim=1
#pragma HLS ARRAY_PARTITION variable=acc cyclic factor=16 dim=1
#pragma HLS RESOURCE variable=acc core=RAM_2P_LUTRAM

	Reset: for(int iacc = 0; iacc < OUT_SIZE; iacc++)
	{
#pragma HLS UNROLL factor=16
		acc[iacc] = 0;
	}

	NewInput:
	for(int ii = 0; ii < IN_SIZE; ii++)
	{
		int offset = ii * OUT_SIZE;

		FETCH_WEIGHTS:
		for (int i = 0; i < OUT_SIZE/8; i++)
#pragma HLS PIPELINE
			temp_weights[i] = weightStream.read();

		data_cache = inStream.read();
		Product: for(int jj = 0; jj < OUT_SIZE; jj++)
		{
#pragma HLS UNROLL factor=16
#pragma HLS PIPELINE
			weight_T tmp;
			short high,low;


			high = (jj%8) * 8 + 7;
			low = (jj%8) * 8;

			tmp.V = temp_weights[jj/8].range(high, low);

			acc[jj] += data_cache * tmp;
		}
	}

	Result: for(int ires = 0; ires < OUT_SIZE; ires++)
	{
#pragma HLS PIPELINE
		input_T tmp = (input_T) (acc[ires] + (input_T) biases[ires]);
		outStream << tmp;
	}
}


// DRAM weights FC layer - axi_m

template<class input_T,class mem_T, int IN_SIZE, int OUT_SIZE>
void fc_layer_dram2(hls::stream<input_T> &inStream,
		hls::stream<input_T> &outStream,
		input_T biases[OUT_SIZE],
		volatile mem_T* mem_addr)
{
	mem_T temp_weight[OUT_SIZE/2];
	input_T data_cache;
	input_T acc[OUT_SIZE];

#pragma HLS ARRAY_PARTITION variable=temp_weight cyclic factor=16 dim=1
#pragma HLS ARRAY_PARTITION variable=acc cyclic factor=16 dim=1
#pragma HLS RESOURCE variable=acc core=RAM_2P_LUTRAM

	Reset: for(int iacc = 0; iacc < OUT_SIZE; iacc++)
	{
#pragma HLS UNROLL factor=16
		acc[iacc] = 0;
	}



	NewInput:
	for(int ii = 0; ii < IN_SIZE; ii++)
	{
		int offset = ii * OUT_SIZE/2;

		int burst_count = OUT_SIZE/2/MAX_BURST_SIZE; // max burst 256
		for (int burst_idx = 0; burst_idx < burst_count; burst_idx++)
		{
			memcpy(temp_weight+MAX_BURST_SIZE*burst_idx,
					(const mem_T*)(mem_addr+offset+burst_idx*MAX_BURST_SIZE),
					MAX_BURST_SIZE*sizeof(mem_T));
		}

		int remain = OUT_SIZE/2 - MAX_BURST_SIZE*burst_count;
		if (remain)
		{
			memcpy(temp_weight+MAX_BURST_SIZE*burst_count,
					(const mem_T*)(mem_addr+MAX_BURST_SIZE*burst_count),
					remain*sizeof(mem_T));
		}

		data_cache = inStream.read();
		Product: for(int jj = 0; jj < OUT_SIZE; jj++)
		{
#pragma HLS UNROLL factor=16
#pragma HLS PIPELINE
			input_T tmp;
			tmp.V = temp_weight[jj/2].range( (jj%2==0) ? 31 : 63,(jj%2==0 )? 0 : 32);
			acc[jj] += data_cache * tmp;
		}
	}

	Result: for(int ires = 0; ires < OUT_SIZE; ires++)
	{
#pragma HLS PIPELINE
		input_T tmp = (input_T) (acc[ires] + (input_T) biases[ires]);
		outStream << tmp;


	}
}



// old code

template<class data_T, class res_T, class weight_T, class bias_T, class acc_T, int N_IN, int N_OUT>
void compute_small_layer(
		hls::stream<data_T>    &data,
		hls::stream<res_T>     &res,
		weight_T  weights[N_IN][N_OUT],
		bias_T    biases[N_OUT]);

template<class data_T, class res_T, class weight_T, class bias_T, class acc_T, int N_IN, int N_OUT>
void compute_medium_layer(
		hls::stream<data_T>    &data,
		hls::stream<res_T>     &res,
		weight_T  weights[N_IN][N_OUT],
		bias_T    biases[N_OUT]);

template<class data_T, class res_T, class weight_T, class bias_T, class acc_T, int N_IN, int N_OUT>
void compute_large_layer(
		hls::stream<data_T>    &data,
		hls::stream<res_T>     &res,
		weight_T  weights[N_IN][N_OUT],
		bias_T    biases[N_OUT]);

// *************************************************
//       Entry Function
// *************************************************

template<class data_T, class acc_T, class weight_T, class bias_T, class res_T, int N_IN, int N_OUT>
void compute_layer(
		hls::stream<data_T>    &data,
		hls::stream<res_T>     &res,
		weight_T  weights[N_IN][N_OUT],
		bias_T    biases[N_OUT])
{
	if (N_OUT >= 512) {
		compute_large_layer<data_T, res_T, weight_T, bias_T, acc_T, N_IN, N_OUT>(data, res, weights, biases);
	}
	else if (N_OUT >= 32) {
		compute_medium_layer<data_T, res_T, weight_T, bias_T, acc_T, N_IN, N_OUT>(data, res, weights, biases);
	}
	else {
		compute_small_layer<data_T, res_T, weight_T, bias_T, acc_T, N_IN, N_OUT>(data, res, weights, biases);
	}
}

// *************************************************
//       Possible implementation options
// *************************************************


template<class data_T, class res_T, class weight_T, class bias_T, class acc_T, int N_IN, int N_OUT>
void compute_small_layer(
		hls::stream<data_T>    &data,
		hls::stream<res_T>     &res,
		weight_T  weights[N_IN][N_OUT],
		bias_T    biases[N_OUT])
{
	//printf("small\r\n\r\n");
	data_T data_cache;
	acc_T acc[N_OUT];

#pragma HLS ARRAY_RESHAPE variable=weights complete dim=2
#pragma HLS ARRAY_PARTITION variable=acc complete dim=1

	Reset: for(int iacc = 0; iacc < N_OUT; iacc++)
#pragma HLS UNROLL
		acc[iacc] = 0;

	NewInput: for(int ii = 0; ii < N_IN; ii++) {
#pragma HLS PIPELINE
		data_cache = data.read();
		Product: for(int jj = 0; jj < N_OUT; jj++) {
			acc[jj] += data_cache * weights[ii][jj];
		}
	}
	Result: for(int ires = 0; ires < N_OUT; ires++)
#pragma HLS PIPELINE
		res << (res_T) (acc[ires] + (acc_T) biases[ires]);
}

template<class data_T, class res_T, class weight_T, class bias_T, class acc_T, int N_IN, int N_OUT>
void compute_medium_layer(
		hls::stream<data_T>    &data,
		hls::stream<res_T>     &res,
		weight_T  weights[N_IN][N_OUT],
		bias_T    biases[N_OUT])
{
	data_T data_cache;
	acc_T acc[N_OUT];

#pragma HLS ARRAY_PARTITION variable=weights cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=acc cyclic factor=8 dim=1

	// Optional... Cuts down on a few of the BRAMs
#pragma HLS RESOURCE variable=acc core=RAM_2P_LUTRAM

	Reset: for(int iacc = 0; iacc < N_OUT; iacc++) {
#pragma HLS UNROLL factor=8
		acc[iacc] = 0;
	}

	NewInput: for(int ii = 0; ii < N_IN; ii++) {
		data_cache = data.read();
		Product: for(int jj = 0; jj < N_OUT; jj++) {
#pragma HLS UNROLL factor=8
#pragma HLS PIPELINE
			acc[jj] += data_cache * weights[ii][jj];
		}
	}

	Result: for(int ires = 0; ires < N_OUT; ires++)
#pragma HLS PIPELINE
		res << (res_T) (acc[ires] + (acc_T) biases[ires]);
}


template<class data_T, class res_T, class weight_T, class bias_T, class acc_T, int N_IN, int N_OUT>
void compute_large_layer(
		hls::stream<data_T>    &data,
		hls::stream<res_T>     &res,
		weight_T  weights[N_IN][N_OUT],
		bias_T    biases[N_OUT])
{
	data_T data_cache;
	acc_T acc[N_OUT];

#pragma HLS ARRAY_PARTITION variable=weights cyclic factor=16 dim=2
#pragma HLS ARRAY_PARTITION variable=acc cyclic factor=16 dim=1

	// Optional... Cuts down on a few of the BRAMs
#pragma HLS RESOURCE variable=acc core=RAM_2P_LUTRAM

	Reset: for(int iacc = 0; iacc < N_OUT; iacc++) {
#pragma HLS UNROLL factor=16
		acc[iacc] = 0;
	}

	NewInput: for(int ii = 0; ii < N_IN; ii++) {
		data_cache = data.read();
		Product: for(int jj = 0; jj < N_OUT; jj++) {
#pragma HLS UNROLL factor=16
#pragma HLS PIPELINE
			acc[jj] += data_cache * weights[ii][jj];
		}
	}

	Result: for(int ires = 0; ires < N_OUT; ires++)
#pragma HLS PIPELINE
		res << (res_T) (acc[ires] + (acc_T) biases[ires]);
}

}

#endif
