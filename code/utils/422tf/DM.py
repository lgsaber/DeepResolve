import matplotlib
matplotlib.use('Agg')
import sys,os,caffe,numpy as np,h5py
#import sys,os,numpy as np,h5py
from os.path import join,dirname,basename,exists
from os import system
import matplotlib.pyplot as plt
from os import makedirs
from keras.models import model_from_json
from keras import backend as K
import theano
import numpy as np,sys,h5py,cPickle,argparse,subprocess,seaborn as sns

basedir='/cluster/zeng/research/recomb/generic/'
IsAnneal=True
avg_iter=4
buff_range=10
arch='best_archit.json'
weight='bestmodel_weights.h5'
tryi=100

def main():
	if len(sys.argv) < 3:
		print "you must call program as: python filtermap.py <method> <resultdir><rootdir><expname>"
		sys.exit(1)
	base = sys.argv[1]
	arc=sys.argv[2]
	architecture_file=join(base,arc+arch)
	weight_file=join(base,arc+weight)
	resultdir=sys.argv[3]
	L2coeff=float(sys.argv[4])
	step=float(sys.argv[5])
	model = model_from_json(open(architecture_file).read())
	model.load_weights(weight_file)
	layer_input = model.layers[0].input
	layer_size=model.layers[0].input_shape
	print layer_size
	output_layer = model.layers[-2].output
	savename=join(resultdir,'inputmap-'+str(L2coeff)+'-'+str(step))
	tryi=11
	activation = tryi*K.mean(output_layer[:,0])
	activation_all=output_layer[:,0]
	loss = activation-L2coeff*K.sum((layer_input ** 2))# - L1coeff * K.sum(abs(layer_input))        
	loss_all=activation_all-L2coeff*K.sum(K.sum(K.sum((layer_input**2),1),1),1)
	grads = K.gradients(loss, layer_input)
	iterate = K.function([layer_input,K.learning_phase()], grads)
	iterate2=K.function([layer_input,K.learning_phase()],[activation_all,loss_all])
	best_map=np.asarray([]).reshape((0,layer_size[1],layer_size[2],layer_size[3]))
	input_data=np.asarray([]).reshape((0,layer_size[1],layer_size[2],layer_size[3]))
	input_data=np.append(input_data,np.ones((1,layer_size[1],layer_size[2],layer_size[3]))/4.0,axis=0)
	x_new=np.random.random((10,layer_size[1],layer_size[2],layer_size[3]))
	x_new=x_new/np.sum(x_new,1).reshape(10,1,layer_size[2],layer_size[3])
	input_data=np.append(input_data,x_new,axis=0)
	print input_data.shape
	best_input =np.zeros(input_data.shape)
	best_activation = np.asarray([-1000.0]*tryi)
	best_loss = np.asarray([-1000000000.0]*tryi)
	best_iter = np.asarray([-1]*tryi)
	loss_track = []
	activation_track = []
	count=0
	activation_init,loss_init=iterate2([input_data,0])
	holdcnt = np.asarray([0]*tryi)
	lr=step
	print activation_init,loss_init
	mask=np.array([False for i in range(tryi)])
	while True:
		if (count%50==0):
		    print 'Iteration',count
		grads_value = iterate([input_data,0])
		activation_all,loss_all=iterate2([input_data,0])
		loss_track.append(loss_all)
		activation_track.append(activation_all)
		new_activation=np.copy(activation_all)
		new_activation[mask]=-1
		improve=(new_activation>best_activation)
		if sum(improve)>0:
		    best_activation[improve] = activation_all[improve]
		    best_input[improve,:,:,:] = input_data[improve,:,:,:]
		    best_iter[improve]=count
		converge=(loss_all>best_loss)
		best_loss[converge] = loss_all[converge]
		#print loss_all
		#print converge
		#print best_loss
		holdcnt[converge] = 0
		holdcnt[~converge]=holdcnt[~converge]+1
		mask=(holdcnt>=buff_range)
		if (count%50==0):
		    print 'Activation',np.mean(activation_all)
		    print 'Converged',sum(mask)
		    print 'Loss',np.mean(loss_all)
		if count>1000:
		    lr=max(step*(count-1000)**(-0.2),1e-6)
		if sum(mask)==tryi or count>5000:
		    print 'Converge at',count
		    print 'Activation',np.mean(activation_all)
		    print 'Converged',sum(mask)
		    print 'Loss',np.mean(loss_all)
		    print 'Best score',np.mean(best_activation)
		    break
		grads_value[mask,:,:,:]=0
		#print grads_value[0,:,:,:]
		input_data+= grads_value*lr
		input_data[input_data<0]=0
		count=count + 1
	best_map=np.append(best_map,best_input,axis=0)
	print best_map.shape
	best=np.reshape(best_map,(best_map.shape[0],best_map.shape[1],best_map.shape[3]))
	f=h5py.File(savename,'w')
        f.create_dataset("inputmap",data=best)
	f.create_dataset("score",data=best_activation)
	f.create_dataset("iteration",data=count)
	f.close()

if __name__ == "__main__":
	main()

