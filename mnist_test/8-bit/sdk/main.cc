
#include <stdio.h>
#include <cmath>
#include <bitset>
#include <iostream>
#include "xil_printf.h"
#include "xparameters.h"
#include "xdeepmnist.h"
#include "xaxidma.h"
#include "xtmrctr.h"
#include "mnist_test_1000.h"

#include "wf1.h"  // load fc1 weights
#include "wc2.h"  // load fc1 weights

#define img_size 784*1
#define out_size 10*4
#define weight_size 1656832*1

char input_image[784];

typedef union
{
	long long int64;
	signed char int8[8];
} T64;  // needed for 64-bit alignment

T64 mem[1656832/8];

int* DDR_address = (int*)0x0160000;

int fixed_point_factor = 128;

int main()
{
	for (int i=0; i < 10;i++)
	{
		DDR_address[i]=0;
	}

	// AXI Timer
	XTmrCtr tmr;
	XTmrCtr_Initialize(&tmr,XPAR_TMRCTR_0_DEVICE_ID);

	xil_printf("\r\n\n\n======== New Run ==========\r\n");
	XDeepmnist xdm;
	XDeepmnist_Config *xdm_cfg = XDeepmnist_LookupConfig(XPAR_DEEPMNIST_0_DEVICE_ID);
	int status = XDeepmnist_CfgInitialize(&xdm,xdm_cfg);
	xil_printf("XDeepMNIST IP status: %d\t",status);

	// AXI DMA for input image and output scores
	XAxiDma axidma;
	XAxiDma_Config *axidmacfg=XAxiDma_LookupConfig(XPAR_AXI_DMA_0_DEVICE_ID);
	status = XAxiDma_CfgInitialize(&axidma,axidmacfg);
	xil_printf("XAxiDMA status %d \r\n",status);
	XAxiDma_IntrDisable(&axidma,XAXIDMA_IRQ_ALL_MASK,XAXIDMA_DEVICE_TO_DMA);
	XAxiDma_IntrDisable(&axidma,XAXIDMA_IRQ_ALL_MASK,XAXIDMA_DMA_TO_DEVICE);

	// AXI DMA for input image and output scores
	XAxiDma axidma2;
	XAxiDma_Config *axidmacfg2 = XAxiDma_LookupConfig(XPAR_AXI_DMA_1_DEVICE_ID );
	status = XAxiDma_CfgInitialize(&axidma2,axidmacfg2);
	xil_printf("XAxiDMA2 status %d \r\n",status);
	XAxiDma_IntrDisable(&axidma2,XAXIDMA_IRQ_ALL_MASK,XAXIDMA_DEVICE_TO_DMA);
	XAxiDma_IntrDisable(&axidma2,XAXIDMA_IRQ_ALL_MASK,XAXIDMA_DMA_TO_DEVICE);

	// convert to fixed point
	for (int i=0; i < 51200;i++)
	{
		mem[i/8].int8[i%8]= round((float)fixed_point_factor * weights_conv2[i]);

		if( i<22)
			printf("orig %f  big  %f  saved: %f    int %d\n\r",
					weights_conv2[i],
					round((float)fixed_point_factor * weights_conv2[i]),
					mem[i/8].int8[i%8]/(float)	fixed_point_factor,
					mem[i/8].int8[i%8] );


	}

	for (int i=0; i < 1605632;i++)
	{
		mem[51200/8+i/8].int8[i%8]= round((float)fixed_point_factor * weights_fc1[i]);
		if( i<22)
			printf("==orig %f  big  %f  saved: %f    int %d\n\r",
					weights_fc1[i],
					round((float)fixed_point_factor * weights_fc1[i]),
					mem[51200/8+i/8].int8[i%8]/(float)	fixed_point_factor,
					mem[51200/8+i/8].int8[i%8] );
	}

	int wrong = 0;

	Xil_DCacheFlushRange((u32)mem,weight_size);

	int image_test_count = 1000;

	for (int test_image_index = 0; test_image_index < image_test_count; test_image_index++)
	{
		printf("\nInput Image Index.....%d\r\n",test_image_index+1);
		for (int i=0; i < 784; i++)
		{
			// convert to fixed point
//			input_image[i]= (float)256 * input_matrix_input[test_image_index][i+1] / (float) 255;
			input_image[i]= 0.5*input_matrix_input[test_image_index][i+1];;
		}

		// Clear changes to memory
		Xil_DCacheFlushRange((u32)input_image,img_size);
		Xil_DCacheFlushRange((u32)DDR_address,out_size);

		// Start Timer
		XTmrCtr_Reset(&tmr,0);
		int tick1 = XTmrCtr_GetValue(&tmr,0), tick2;
		XTmrCtr_Start(&tmr,0);

		// Start IP CORE
		XDeepmnist_Start(&xdm);
		//				printf("1 \n");
		// Send and receive data
		XAxiDma_SimpleTransfer(&axidma,(u32)input_image,img_size,XAXIDMA_DMA_TO_DEVICE);
		XAxiDma_SimpleTransfer(&axidma,(u32)DDR_address,out_size,XAXIDMA_DEVICE_TO_DMA);
		XAxiDma_SimpleTransfer(&axidma2,(u32)mem,weight_size,XAXIDMA_DMA_TO_DEVICE);
		//				printf("2 \n");
		while(XAxiDma_Busy(&axidma,XAXIDMA_DMA_TO_DEVICE));
		//		printf("22 \n");
		while(XAxiDma_Busy(&axidma2,XAXIDMA_DMA_TO_DEVICE));
		//				printf("3 \n");
		while(XAxiDma_Busy(&axidma,XAXIDMA_DEVICE_TO_DMA));
		Xil_DCacheInvalidateRange((u32)DDR_address,out_size);
		//				printf("4 \n");
		while(!XDeepmnist_IsDone(&xdm));
		//
		//		Xil_DCacheInvalidateRange((u32)DDR_address,out_size);
		//		Xil_DCacheFlushRange((u32)DDR_address,out_size);

		// Calculate Time
		XTmrCtr_Stop(&tmr,0);
		tick2 = XTmrCtr_GetValue(&tmr,0);
		printf("Time: %.2f ms\r\n",(double)(tick2-tick1)*1000/(double)XPAR_AXI_TIMER_0_CLOCK_FREQ_HZ);

		// Display Time
		for (int i = 0; i< 10; i++)
		{
			float tmps = ((float)DDR_address[i]/(float)1048576);
			printf("OUT %i = %+f\n\r",i,tmps);
		}

		// Find digit classification / max score
		int max_index = 0;
		float maxed = -1e99;

		for (int i = 0; i< 10; i++)
		{
			if (DDR_address[i] > maxed)
			{
				maxed = DDR_address[i];
				max_index = i;
			}
		}
		printf("Correct is %d - Classification %d\n",(int)input_matrix_input[test_image_index][0],max_index);

		if (max_index != input_matrix_input[test_image_index][0])
			wrong++;

		printf("Mismatches so far: %d\n",wrong);
		printf("<----->\n\n");
	}

	printf("Final Incorrect Classifications %d / %d\n",wrong,image_test_count);

	return 0;
}
