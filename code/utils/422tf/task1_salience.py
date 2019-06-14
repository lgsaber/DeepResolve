import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt,pandas as pd
from os import makedirs
from keras.models import model_from_json
from keras import backend as K
import theano
import numpy as np,sys,h5py,cPickle,argparse,subprocess,seaborn as sns
from os.path import join,dirname,basename,exists
from os import system
import os

basedir='/cluster/zeng/research/recomb/generic/saber'#wgEncodeAwgTfbsUwWi38CtcfUniPk/CV0/data'
#modir='/cluster/zeng/research/recomb/generic/forSirajul/task1'#wgEncodeAwgTfbsUwWi38CtcfUniPk/CV0/128_G/mri-best/best_trial

def main():
    if len(sys.argv) < 3:
        print "you must call program as: python filtermap.py <method> <resultdir><rootdir><start><end>"
        sys.exit(1)
    savename ='saliencemap_10000'
    savename2='sainputmap_10000'
    method = sys.argv[3]
    rootdir=sys.argv[1]
    resultdir=sys.argv[2]
    dir_ls=os.listdir(rootdir)
    dir_ls.sort()
    weight_file=method+'bestmodel_weights.h5'
    architecture_file=method+'best_archit.json'
    for x in range (len(dir_ls)):
       if dir_ls[x][0:15]=='wgEncodeAwgTfbs':
         print(dir_ls[x])
         if os.path.exists(join(resultdir,dir_ls[x],savename)):
            print('Exist!skip!')
         else:
            if not exists(resultdir) or not os.path.exists(join(resultdir,dir_ls[x])):
                print('Directory not exits!')
            else:
		inputdir=join(basedir,dir_ls[x],'CV0','data')#,'train.h5.batch1')
		files=subprocess.check_output("ls "+inputdir+"/train.h5.batch*", shell=True).split('\n')[0:-1]
    		input_data=np.asarray([]).reshape((0,4,1,101))
    		label_all=np.asarray([]).reshape((0,1))
		for batchfile in files:
			fi = h5py.File(batchfile, 'r')
			dataset = np.asarray(fi['data'])
			label=np.asarray(fi['label'])
			input_data=np.append(input_data,dataset,axis=0)
			label_all=np.append(label_all,label,axis=0)
		label_all=label_all.reshape(len(label_all))
		dataset=input_data[label_all>0]
		print dataset.shape
		x_in=dataset[0:10000]
                model = model_from_json(open(join(rootdir,dir_ls[x],architecture_file)).read())
	        model.load_weights(join(rootdir,dir_ls[x],weight_file))
    		layer_input = model.layers[0].input
	 	filter_layer=model.layers[3].input
    		layer_output = model.layers[-2].output
	        filter_act=model.layers[2].output
    		activation = K.sum(layer_output)
   		grads = K.gradients(activation, filter_layer)
    		iterate = K.function([layer_input,K.learning_phase()], [filter_act,grads])
		act,salience= iterate([x_in,0])
		sa_act=salience*act
		print salience.shape
		print act.shape
		np.savetxt(join(resultdir,dir_ls[x],savename),salience,delimiter='\t',fmt='%.5f')
		np.savetxt(join(resultdir,dir_ls[x],savename2),sa_act,delimiter='\t',fmt='%.5f')
		#f=h5py.File(join(resultdir,dir_ls[x],savename),'w')
		#f.create_dataset("salience",data=salience)
 	     	#f.create_dataset("sa_act",data=sa_act)
		#sa_avg=np.mean(salience[0:10],axis=0)
		#sa_act_avg=np.mean(sa_act[0:10],axis=0)
		#f.create_dataset("salience_avg",data=sa_avg)
                #f.create_dataset("sa_act_avg",data=sa_act_avg)
		#f.close()

if __name__ == "__main__":
	main()

