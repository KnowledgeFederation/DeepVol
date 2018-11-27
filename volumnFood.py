#!/usr/bin/env python
#!/usr/bin/env python
import sys

caffe_path = ''#ToDo: add caffe path
sys.path.insert(0, caffe_path + 'python')
import numpy as np
import os
import caffe

model_path = os.getcwd() + '/'
iter_n = 37500
use_mean = True
NET_FILE = model_path + 'deploy.prototxt'    
MODEL_FILE = model_path + '/_iter_' + str(iter_n) + '.caffemodel'          
gpu_ind = 0
  
caffe.set_mode_gpu()
caffe.set_device(gpu_ind)
#caffe.set_mode_cpu()
net = caffe.Net(NET_FILE, MODEL_FILE, caffe.TEST)
 
if use_mean:
    MEAN_FILE = model_path + 'mean.binaryproto'
    blob = caffe.proto.caffe_pb2.BlobProto()  
    bin_mean = open(MEAN_FILE, 'rb' ).read()  
    blob.ParseFromString(bin_mean)  
    arr = np.array( caffe.io.blobproto_to_array(blob) )  
    npy_mean = arr[0]

transformer = caffe.io.Transformer({'data': net.blobs['data'].shape})  
transformer.set_transpose('data', (2,0,1))  
if use_mean:  
    transformer.set_mean('data', npy_mean.mean(1).mean(1))  
transformer.set_raw_scale('data', 255)   
transformer.set_channel_swap('data', (2,1,0))   

def predictVolumn(path):
    est_values = [] 
     
    im = caffe.io.load_image(path)
    net.blobs['data'].data[...] = transformer.preprocess('data',im)    
    output = net.forward()
    est_values.append(float(output["pred"]))
    
    print np.mean(est_values)

  
if __name__ == '__main__':
    predictVolumn('170.jpg')


