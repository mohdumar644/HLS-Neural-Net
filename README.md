# HLS-Neural-Net

An extension of the HLS library in [RFNOC-HLS-NeuralNet](https://github.com/Xilinx/RFNoC-HLS-NeuralNet) to support 2-D convolutions. Also includes demo project(s).

## Demo Setup

Use [Ristretto-Caffe](http://lepsucd.com/?page_id=621) to train and fine-tine a quantized deep network on the MNIST dataset. 

Extract the weights using the [provided scripts](mnist_test/train_caffe_ristretto_quantize), and convert into a suitable format for HLS/SDK.

Use the provided [scripts](mnist_test/8-bit/hls) to generate an HLS project. Co-sim and verify your network.

The reference [designs](mnist_test/8-bit/vivado) for Vivado IP Integrator, and sample [code](mnist_test/8-bit/sdk) for Xilinx SDK are also provided.

## Acknowledgments

The code for the Conv2D layer uses a linebuffer approach, inspired from [here](https://www.youtube.com/watch?v=38lj0VQci7E).
