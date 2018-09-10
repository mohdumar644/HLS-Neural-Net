import caffe
import numpy as np


net = caffe.Net('/home/umar/Desktop/caffe/caffe/python/quantized.prototxt','/home/umar/Desktop/caffe/caffe/python/lenet_iter_1500.caffemodel',caffe.TEST);

print(net.params['ip1'][1].data.shape)

zz = np.reshape(net.params['conv1'][0].data,[1,800])
np.savetxt("w1.csv", zz, delimiter=",")

zz = np.reshape(net.params['conv1'][1].data,[1,32])
np.savetxt("b1.csv", zz, delimiter=",")

zz = np.reshape(net.params['conv2'][0].data,[1,51200])
np.savetxt("w2.csv", zz, delimiter=",")

zz = np.reshape(net.params['conv2'][1].data,[1,64])
np.savetxt("b2.csv", zz, delimiter=",")

ss = np.swapaxes(net.params['ip1'][0].data,0,1)
zz = np.reshape(ss,[1,1605632])
np.savetxt("wf1.csv", zz, delimiter=",")

zz = np.reshape(net.params['ip1'][1].data,[1,512])
np.savetxt("bf1.csv", zz, delimiter=",")

ss = np.swapaxes(net.params['ip2'][0].data,0,1)
zz = np.reshape(ss,[1,5120])
np.savetxt("wf2.csv", zz, delimiter=",")

zz = np.reshape(net.params['ip2'][1].data,[1,10])
np.savetxt("bf2.csv", zz, delimiter=",")
