    ./build/tools/ristretto quantize --model=examples/mnist/lenet_train_test.prototxt \
      --weights=examples/mnist/lenet_iter_1500.caffemodel \
      --model_quantized=examples/mnist/quantized.prototxt \
      --iterations=200 --gpu=0 --trimming_mode=dynamic_fixed_point --error_margin=.5
