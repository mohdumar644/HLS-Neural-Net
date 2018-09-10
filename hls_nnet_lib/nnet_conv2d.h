#ifndef NNET_CONV2D_H_
#define NNET_CONV2D_H_

#include "ap_fixed.h"
#include "hls_stream.h"
#include "hls_video.h"

namespace nnet {

template<class input_T, int IMG_WIDTH_OR_COLS__, int IMG_HEIGHT_OR_ROWS__, int ZERO_PADDING__>
input_T padZeroImage(int idxRow, int idxCol, input_T* local_image, short in_layer)
{
	input_T tmpread=0;

	// upper padding
	if (idxRow < ZERO_PADDING__)
	{
		return tmpread;
	}
	// lower padding
	else if (idxRow > ZERO_PADDING__ + IMG_HEIGHT_OR_ROWS__ - 1)
	{
		return tmpread;
	}
	// middle padding left
	else if (idxCol < ZERO_PADDING__)
	{
		return tmpread;
	}
	// middle padding right
	else if (idxCol > ZERO_PADDING__ + IMG_WIDTH_OR_COLS__ -1)
	{
		return tmpread;
	}
	//actual image
	else
	{
		return local_image[in_layer*IMG_WIDTH_OR_COLS__*IMG_HEIGHT_OR_ROWS__ + IMG_WIDTH_OR_COLS__*(idxRow-ZERO_PADDING__) + idxCol-ZERO_PADDING__];
	}
}


// Sum all values inside window (Already multiplied by the kernel)
template<class input_T, int KERNEL_DIM__>
input_T sumWindow(hls::Window<KERNEL_DIM__,KERNEL_DIM__,input_T> *window)
{
	input_T accumulator = 0;

	// Iterate on the window multiplying and accumulating the kernel and sampling window
	for (int idxRow = 0; idxRow < KERNEL_DIM__; idxRow++)
	{
#pragma HLS PIPELINE
		for (int idxCol = 0; idxCol < KERNEL_DIM__; idxCol++)
		{
			accumulator = accumulator + (input_T)window->getval(idxRow,idxCol);
		}
	}
	return accumulator;
}


template <class input_T, short IN_SIZE>
void streamStore(hls::stream<input_T> &inStream,input_T* local_image)
{
	for(int i = 0; i < IN_SIZE; i++)
#pragma HLS PIPELINE
		local_image[i] = inStream.read();
}


template <
class input_T,
class output_T,
short IMG_WIDTH_OR_COLS__,
short IMG_HEIGHT_OR_ROWS__,
short IMG_CHANNELS_OR_DEPTH__,
short OUT_DEPTH_,
short ZERO_PADDING__,
short KERNEL_DIM__
>
void conv2d_dram_axi_m(
		volatile input_T *mem_addr,
		input_T biases[OUT_DEPTH_],
		hls::stream<input_T> &inStream,
		hls::stream<input_T> &outStream
)
{
	input_T local_image[IMG_CHANNELS_OR_DEPTH__*IMG_WIDTH_OR_COLS__*IMG_HEIGHT_OR_ROWS__];

	streamStore<input_T,IMG_CHANNELS_OR_DEPTH__*IMG_WIDTH_OR_COLS__*IMG_HEIGHT_OR_ROWS__>(inStream,local_image);

	input_T temp_weights[IMG_CHANNELS_OR_DEPTH__*KERNEL_DIM__*KERNEL_DIM__];
	input_T temp_out_image[IMG_WIDTH_OR_COLS__*IMG_HEIGHT_OR_ROWS__];
#pragma HLS ARRAY_PARTITION variable=temp_weights cyclic factor=16 dim=1
#pragma HLS ARRAY_PARTITION variable=temp_out_image cyclic factor=16 dim=1


	OUTER_LOOP:
	for (int out_layer = 0; out_layer < OUT_DEPTH_; out_layer++)
	{
		int offset = out_layer * IMG_CHANNELS_OR_DEPTH__*KERNEL_DIM__*KERNEL_DIM__;

		/************ HARDCODED *************/
		memcpy(temp_weights,(const input_T*)(mem_addr+offset),256*sizeof(input_T));
		memcpy(temp_weights+256,(const input_T*)(mem_addr+offset+256),256*sizeof(input_T));
		memcpy(temp_weights+512,(const input_T*)(mem_addr+offset+512),256*sizeof(input_T));
		memcpy(temp_weights+768,(const input_T*)(mem_addr+offset+768),32*sizeof(input_T));
		/************ TODO: NEEDS TO BE CHANGED *************/

		INIT_ACCUM_LOOP:
		for (int idx = 0; idx < IMG_WIDTH_OR_COLS__*IMG_HEIGHT_OR_ROWS__; idx++)
		{
#pragma HLS UNROLL factor = 16
#pragma HLS PIPELINE
			temp_out_image[idx] = 0;
		}

		IN_LAYER_LOOP:
		for (int in_layer = 0; in_layer < IMG_CHANNELS_OR_DEPTH__; in_layer++)
		{
			hls::LineBuffer<KERNEL_DIM__,IMG_WIDTH_OR_COLS__+2*ZERO_PADDING__,input_T> lineBuff;

			// Index used to keep track of row,col
			int idxCol = 0;
			int idxRow = 0;
			int pixConvolved = 0;
			// Calculate delay to fix line-buffer offset
			int waitTicks = ((IMG_WIDTH_OR_COLS__+2*ZERO_PADDING__)*(KERNEL_DIM__-1)+KERNEL_DIM__)/2;// 241;
			int countWait = 0;
			int sentPixels = 0;

			TRAVERSE_PIXEL_LOOP:
			for (int idxPixel = 0; idxPixel < ((IMG_WIDTH_OR_COLS__+2*ZERO_PADDING__)*(2*ZERO_PADDING__+IMG_HEIGHT_OR_ROWS__)); idxPixel++)
			{
#pragma HLS PIPELINE
				// Read and cache (Block here if FIFO sender is empty)
				input_T pixelIn= padZeroImage<input_T, IMG_WIDTH_OR_COLS__,IMG_HEIGHT_OR_ROWS__,ZERO_PADDING__>(idxRow,idxCol,local_image,in_layer);//inStream.read();

				// Put data on the LineBuffer
				lineBuff.shift_up(idxCol);
				lineBuff.insert_top(pixelIn,idxCol); // Will put in val[2] of line buffer (Check Debug)

				countWait++;

				if ((idxRow >= KERNEL_DIM__-1) && (idxCol >= KERNEL_DIM__-1))
				{
					input_T acc=0;
					// Put data on the window and multiply with the kernel
					MULITPLY_KERNEL:
					for (int idxWinRow = 0; idxWinRow < KERNEL_DIM__; idxWinRow++)
					{
#pragma HLS UNROLL
#pragma HLS PIPELINE
						MULITPLY_KERNEL_INNER:
						for (int idxWinCol = 0; idxWinCol < KERNEL_DIM__; idxWinCol++)
						{
#pragma HLS UNROLL
#pragma HLS PIPELINE
							input_T val = (input_T)lineBuff.getval(idxWinRow,idxWinCol+pixConvolved);

							// Multiply kernel by the sampling window
							val = (input_T)temp_weights[//out_layer*(KERNEL_DIM__*KERNEL_DIM__)*IMG_CHANNELS_OR_DEPTH__+
														in_layer *(KERNEL_DIM__*KERNEL_DIM__) +
														(idxWinRow*KERNEL_DIM__) + idxWinCol ] * val;
							acc+=val;

						}
					}

					pixConvolved++;
					if (countWait > waitTicks)
					{

						temp_out_image[sentPixels] += acc;
						sentPixels++;
					}
				}

				// Calculate row and col index
				if (idxCol < IMG_WIDTH_OR_COLS__ + 2*ZERO_PADDING__ - 1)
				{
					idxCol++;
				}
				else
				{
					// New line
					idxCol = 0;
					idxRow++;
					pixConvolved = 0;
				}

			}
		}


		OUT_WRITE_LOOP:
		for (int i = 0; i < IMG_WIDTH_OR_COLS__*IMG_HEIGHT_OR_ROWS__; i++)
		{
			outStream << temp_out_image[i] + biases[out_layer];
		}
	}
}


template <
class input_T,
class acc_T,
class weight_T,
class bias_T,
class output_T,
short IMG_WIDTH_OR_COLS__,
short IMG_HEIGHT_OR_ROWS__,
short IMG_CHANNELS_OR_DEPTH__,
short OUT_DEPTH_,
short ZERO_PADDING__,
short KERNEL_DIM__
>
void conv2d_localweights(	// on-chip weights
		weight_T kernel_weights[OUT_DEPTH_*IMG_CHANNELS_OR_DEPTH__*KERNEL_DIM__*KERNEL_DIM__],
		bias_T biases[OUT_DEPTH_],
		hls::stream<input_T> &inStream,
		hls::stream<output_T> &outStream
)
{
	input_T local_image[IMG_CHANNELS_OR_DEPTH__*IMG_WIDTH_OR_COLS__*IMG_HEIGHT_OR_ROWS__];

	streamStore<input_T,IMG_CHANNELS_OR_DEPTH__*IMG_WIDTH_OR_COLS__*IMG_HEIGHT_OR_ROWS__>(inStream,local_image);

	acc_T temp_out_image[IMG_WIDTH_OR_COLS__*IMG_HEIGHT_OR_ROWS__];

	weight_T temp_kernel[KERNEL_DIM__*KERNEL_DIM__] ;

#pragma HLS ARRAY_PARTITION variable=temp_kernel cyclic factor=16 dim=1
#pragma HLS ARRAY_PARTITION variable=temp_out_image cyclic factor=16 dim=1


	OUTER_LOOP:
	for (int out_layer = 0; out_layer < OUT_DEPTH_; out_layer++)
	{
		INIT_ACCUM_LOOP:
		for (int idx = 0; idx < IMG_WIDTH_OR_COLS__*IMG_HEIGHT_OR_ROWS__; idx++)
		{
#pragma HLS UNROLL factor = 16
#pragma HLS PIPELINE
			temp_out_image[idx] = 0;
		}

		IN_LAYER_LOOP:
		for (int in_layer = 0; in_layer < IMG_CHANNELS_OR_DEPTH__; in_layer++)
		{
			BUFFER_KERNEL:
			for (int ki = 0; ki < KERNEL_DIM__*KERNEL_DIM__; ki++)
			{
#pragma HLS PIPELINE
#pragma HLS UNROLL factor = 16
				temp_kernel[ki] = kernel_weights[out_layer*(KERNEL_DIM__*KERNEL_DIM__)*IMG_CHANNELS_OR_DEPTH__+
												 in_layer *(KERNEL_DIM__*KERNEL_DIM__) +ki];
			}

			hls::LineBuffer<KERNEL_DIM__,IMG_WIDTH_OR_COLS__+2*ZERO_PADDING__,input_T> lineBuff;

			// Index used to keep track of row,col
			int idxCol = 0;
			int idxRow = 0;
			int pixConvolved = 0;
			// Calculate delay to fix line-buffer offset
			int waitTicks = ((IMG_WIDTH_OR_COLS__+2*ZERO_PADDING__)*(KERNEL_DIM__-1)+KERNEL_DIM__)/2;// 241;
			int countWait = 0;
			int sentPixels = 0;

			TRAVERSE_PIXEL_LOOP:
			for (int idxPixel = 0; idxPixel < ((IMG_WIDTH_OR_COLS__+2*ZERO_PADDING__)*(2*ZERO_PADDING__+IMG_HEIGHT_OR_ROWS__)); idxPixel++)
			{
#pragma HLS PIPELINE
				#pragma HLS PIPELINE
				// Read and cache (Block here if FIFO sender is empty)
				input_T pixelIn= padZeroImage<input_T, IMG_WIDTH_OR_COLS__,IMG_HEIGHT_OR_ROWS__,ZERO_PADDING__>(idxRow,idxCol,local_image,in_layer);

				// Put data on the LineBuffer
				lineBuff.shift_up(idxCol);
				lineBuff.insert_top(pixelIn,idxCol); // Will put in val[2] of line buffer (Check Debug)

				countWait++;

				if ((idxRow >= KERNEL_DIM__-1) && (idxCol >= KERNEL_DIM__-1))
				{
					acc_T acc=0;
					// Put data on the window and multiply with the kernel
					MULITPLY_KERNEL:
					for (int idxWinRow = 0; idxWinRow < KERNEL_DIM__; idxWinRow++)
					{
#pragma HLS UNROLL
#pragma HLS PIPELINE
						MULITPLY_KERNEL_INNER:
						for (int idxWinCol = 0; idxWinCol < KERNEL_DIM__; idxWinCol++)
						{
#pragma HLS UNROLL
#pragma HLS PIPELINE
							input_T val = (input_T)lineBuff.getval(idxWinRow,idxWinCol+pixConvolved);

							input_T tmp = (input_T)temp_kernel[(idxWinRow*KERNEL_DIM__) + idxWinCol ];
							acc +=  tmp* val;
						}
					}

					pixConvolved++;
					if (countWait > waitTicks)
					{

						temp_out_image[sentPixels] += acc;
						sentPixels++;
					}
				}

				// Calculate row and col index
				if (idxCol < IMG_WIDTH_OR_COLS__ + 2*ZERO_PADDING__ - 1)
				{
					idxCol++;
				}
				else
				{
					// New line
					idxCol = 0;
					idxRow++;
					pixConvolved = 0;
				}
			}
		}

		OUT_WRITE_LOOP:
		for (int i = 0; i < IMG_WIDTH_OR_COLS__*IMG_HEIGHT_OR_ROWS__; i++)
		{
			#pragma HLS PIPELINE
			outStream << temp_out_image[i] + biases[out_layer];
		}

	}
}



template <
class input_T,
class acc_T,
class weight_T,
class bias_T,
class output_T,
class mem_T,
short IMG_WIDTH_OR_COLS__,
short IMG_HEIGHT_OR_ROWS__,
short IMG_CHANNELS_OR_DEPTH__,
short OUT_DEPTH_,
short ZERO_PADDING__,
short KERNEL_DIM__
>
void conv2d_stream(
		hls::stream<mem_T> &weightStream,
		bias_T biases[OUT_DEPTH_],
		hls::stream<input_T> &inStream,
		hls::stream<output_T> &outStream
)
{
	input_T local_image[IMG_CHANNELS_OR_DEPTH__*IMG_WIDTH_OR_COLS__*IMG_HEIGHT_OR_ROWS__];

	streamStore<input_T,IMG_CHANNELS_OR_DEPTH__*IMG_WIDTH_OR_COLS__*IMG_HEIGHT_OR_ROWS__>(inStream,local_image);

	mem_T temp_weights[IMG_CHANNELS_OR_DEPTH__*KERNEL_DIM__*KERNEL_DIM__/8];
	acc_T temp_out_image[IMG_WIDTH_OR_COLS__*IMG_HEIGHT_OR_ROWS__];

#pragma HLS ARRAY_PARTITION variable=temp_weights cyclic factor=16 dim=1
#pragma HLS ARRAY_PARTITION variable=temp_out_image cyclic factor=16 dim=1


	OUTER_LOOP:
	for (int out_layer = 0; out_layer < OUT_DEPTH_; out_layer++)
	{
		int offset = out_layer * IMG_CHANNELS_OR_DEPTH__*KERNEL_DIM__*KERNEL_DIM__;

		FETCH_WEIGHTS:
		for (int i = 0; i < IMG_CHANNELS_OR_DEPTH__*KERNEL_DIM__*KERNEL_DIM__/8; i++)
#pragma HLS PIPELINE
			temp_weights[i] = weightStream.read();

		INIT_ACCUM_LOOP:
		for (int idx = 0; idx < IMG_WIDTH_OR_COLS__*IMG_HEIGHT_OR_ROWS__; idx++)
		{
#pragma HLS UNROLL factor = 16
#pragma HLS PIPELINE
			temp_out_image[idx] = 0;
		}

		IN_LAYER_LOOP:
		for (int in_layer = 0; in_layer < IMG_CHANNELS_OR_DEPTH__; in_layer++)
		{
			hls::LineBuffer<KERNEL_DIM__,IMG_WIDTH_OR_COLS__+2*ZERO_PADDING__,input_T> lineBuff;

			// Index used to keep track of row,col
			int idxCol = 0;
			int idxRow = 0;
			int pixConvolved = 0;
			// Calculate delay to fix line-buffer offset
			int waitTicks = ((IMG_WIDTH_OR_COLS__+2*ZERO_PADDING__)*(KERNEL_DIM__-1)+KERNEL_DIM__)/2;// 241;
			int countWait = 0;
			int sentPixels = 0;

			TRAVERSE_PIXEL_LOOP:
			for (int idxPixel = 0; idxPixel < ((IMG_WIDTH_OR_COLS__+2*ZERO_PADDING__)*(2*ZERO_PADDING__+IMG_HEIGHT_OR_ROWS__)); idxPixel++)
			{
#pragma HLS PIPELINE
				// Read and cache (Block here if FIFO sender is empty)
				input_T pixelIn= padZeroImage<input_T, IMG_WIDTH_OR_COLS__,IMG_HEIGHT_OR_ROWS__,ZERO_PADDING__>(idxRow,idxCol,local_image,in_layer);//inStream.read();

				// Put data on the LineBuffer
				lineBuff.shift_up(idxCol);
				lineBuff.insert_top(pixelIn,idxCol); // Will put in val[2] of line buffer (Check Debug)

				countWait++;

				if ((idxRow >= KERNEL_DIM__-1) && (idxCol >= KERNEL_DIM__-1))
				{
					acc_T acc=0;
					// Put data on the window and multiply with the kernel
					MULITPLY_KERNEL:
					for (int idxWinRow = 0; idxWinRow < KERNEL_DIM__; idxWinRow++)
					{
#pragma HLS UNROLL
#pragma HLS PIPELINE
						MULITPLY_KERNEL_INNER:
						for (int idxWinCol = 0; idxWinCol < KERNEL_DIM__; idxWinCol++)
						{
#pragma HLS UNROLL
#pragma HLS PIPELINE
							input_T val = (input_T)lineBuff.getval(idxWinRow,idxWinCol+pixConvolved);

							input_T tmp;
							int idx = in_layer *(KERNEL_DIM__*KERNEL_DIM__) + (idxWinRow*KERNEL_DIM__) + idxWinCol;

							short high,low;

							high = (idx%8) * 8 + 7;
							low = (idx%8) * 8;
							tmp.V = temp_weights[idx/8].range(high, low);

							// Multiply kernel by the sampling window
							acc += tmp * val;
						}
					}

					pixConvolved++;
					if (countWait > waitTicks)
					{
						temp_out_image[sentPixels] += acc;
						sentPixels++;
					}
				}

				// Calculate row and col index
				if (idxCol < IMG_WIDTH_OR_COLS__ + 2*ZERO_PADDING__ - 1)
				{
					idxCol++;
				}
				else
				{
					// New line
					idxCol = 0;
					idxRow++;
					pixConvolved = 0;
				}

			}
		}


		OUT_WRITE_LOOP:
		for (int i = 0; i < IMG_WIDTH_OR_COLS__*IMG_HEIGHT_OR_ROWS__; i++)
		{
			outStream << temp_out_image[i] + biases[out_layer];
		}
	}
}

}

#endif
