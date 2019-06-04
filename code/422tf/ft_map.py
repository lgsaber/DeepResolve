import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt,pandas as pd
from os import makedirs
#from keras.models import model_from_json
#from keras import backend as K
#import theano
import numpy as np,sys,h5py,cPickle,argparse,subprocess,seaborn as sns
from os.path import join,dirname,basename,exists
from os import system

IsAnneal=True
avg_iter=10
buff_range=10
arch='best_archit.json'#'seq_best_archit.json'
weight='bestmodel_weights.h5'#'seq_bestmodel_weights.h5'

def main():
    if len(sys.argv) < 4:
        print "you must call program as: python get_weights.py <rootpath> <your_archit.json> <your_weights.h5> <resultdir> "
        sys.exit(1)
    base = sys.argv[1]
    arc=sys.argv[2]
    architecture_file=join(base,arc+arch)
    weight_file=join(base,arc+weight)
    resultdir=sys.argv[3]
    L2coeff=float(sys.argv[4])
    step=float(sys.argv[5])
    savename=join(resultdir,arc+'importance_map-fc1-'+str(L2coeff)+'-'+str(step))
    savename2=join(resultdir,arc+'importance_score-'+str(L2coeff)+'-'+str(step))
    #if exists(savename2):
#	print 'exist!skip'
#	return
    from keras.models import model_from_json
    from keras import backend as K
    import theano
    if not exists(resultdir):
        system('mkdir '+resultdir)
    model = model_from_json(open(architecture_file).read())
    model.load_weights(weight_file)
    layer_input = model.layers[3].input
    layer_size=model.layers[3].input_shape
    layer_output = model.layers[-2].output
    activation = K.mean(layer_output[0,0])
    loss = activation - L2coeff * K.sum((layer_input ** 2))# - L1coeff * K.sum(abs(layer_input))
    grads = K.gradients(loss, layer_input)[0]
    iterate = K.function([layer_input,K.learning_phase()], [activation,loss, grads])
    best_map=[]
    best_score=[]
    for trial in range(avg_iter):
    	input_data = np.random.randn((1,layer_size[1]))#abs(np.random.random((1,layer_size[1]))*1.4)#np.random.random((1,layer_size[1]))
    	best_input = np.asarray([])
    	best_activation = -100000
    	best_loss = -100000
    	best_iter = -1
    	loss_track = []
    	activation_track = []
    	i=0
    	holdcnt = 0
        lr=step
    	while True:
        	if (i%50==0):
		    print 'Iteration',i
        	activation,loss_value, grads_value = iterate([input_data,0])
        	loss_track.append(loss_value)
        	activation_track.append(activation)
        	if activation > best_activation:
        	    best_activation = activation
        	    best_input = input_data
        	    best_iter = i
        	if loss_value > best_loss:
        	    best_loss = loss_value
        	    holdcnt = 0
        	else:
        	    holdcnt += 1
        	if (holdcnt >= buff_range) or (i>=10000):
        	        #break
		    print 'Iteration',i
        	    print 'Activation',activation
        	    print 'Loss',loss_value
		    break
		if (i%50==0):
		    print 'Activation',activation
		    print 'Loss',loss_value
		if i>500:
		    lr=max(step*(i-500)**(-0.2),1e-6)	
        	input_data += grads_value *lr
        #input_data = input_data.clip(0.01,10000000000)
        	i = i + 1	
	converge_i = i-1
	print 'Best activation:',best_activation
	print 'Best iter:',best_iter
	best_map.append(best_input.reshape(layer_size[1]))
	best_score.append(best_activation)
    best_map=np.asarray(best_map)
    best_score=np.asarray(best_score)
    print best_map.shape
    savename=join(resultdir,arc+'importance_map-fc1-'+str(L2coeff)+'-'+str(step))
    np.savetxt(savename,best_map,delimiter='\t',fmt='%.5f')
    savename2=join(resultdir,arc+'importance_score-'+str(L2coeff)+'-'+str(step))
    np.savetxt(savename2,best_score,delimiter='\t',fmt='%.5f')

if __name__ == "__main__":
        main()

