import matplotlib as mpl
mpl.use('Agg')
#import matplotlib.pyplot as plt,pandas as pd
from os import makedirs
from keras.models import model_from_json
#from keras import backend as K
#import theano
import deeplift
from deeplift.conversion import keras_conversion as kc
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
    savename ='deepliftmap_all'
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
		x_in=dataset#[0:10000]
                model = model_from_json(open(join(rootdir,dir_ls[x],architecture_file)).read())
	        model.load_weights(join(rootdir,dir_ls[x],weight_file))
		keras_model=model
		deeplift_model = kc.convert_sequential_model(keras_model,nonlinear_mxts_mode=deeplift.blobs.NonlinearMxtsMode.DeepLIFT_GenomicsDefault)
		find_scores_layer_idx=2
		deeplift_contribs_func = deeplift_model.get_target_contribs_func(find_scores_layer_idx=find_scores_layer_idx,target_layer_idx=-2)	
		scores = np.array(deeplift_contribs_func(task_idx=0,input_data_list=[x_in], batch_size=10,progress_update=1000))
		s=scores.shape
		score=scores.reshape((s[0],s[1],s[3]))
		deepmap=np.max(score,axis=2)
		print deepmap.shape
		np.savetxt(join(resultdir,dir_ls[x],savename),deepmap,delimiter='\t',fmt='%.5f')

if __name__ == "__main__":
	main()

