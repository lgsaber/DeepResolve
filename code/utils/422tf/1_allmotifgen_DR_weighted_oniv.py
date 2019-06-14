import matplotlib
matplotlib.use('Agg')
import sys,os,numpy as np,h5py,pandas
from os.path import join,dirname,basename,exists
from os import system
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

def weighted_var(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values,axis=0,weights=weights)
    variance = np.average((values-average)**2,axis=0,weights=weights)  # Fast and numerically precise
    return (average,variance)

def main():
    if len(sys.argv) < 3:
        print "you must call program as: python filtermap.py <method> <rootdir><resultdir><filterfile>"
        sys.exit(1)
    method = sys.argv[5]
    rootdir=sys.argv[1]
    resultdir=sys.argv[2]
    filter_file=sys.argv[3]
    score_file=sys.argv[4]
    outfile=sys.argv[6]
    downsample=int(sys.argv[7])
    dir_ls=os.listdir(rootdir)
    dir_ls.sort()
    weight_file=method+'bestmodel_weights.h5'
    ONIV={'num_compo':{},'oniv':{},'var':{}}
    for x in range (len(dir_ls)):
       if dir_ls[x][0:15]=='wgEncodeAwgTfbs':
         print(dir_ls[x])
         if False:#os.path.exists(join(resultdir,dir_ls[x],outfile)):
            print('Exist!skip!')
         else:
            if not exists(resultdir) or not os.path.exists(join(resultdir,dir_ls[x])):
		print('Directory not exits!')
	    else:
		filtermap=np.loadtxt(join(resultdir,dir_ls[x],filter_file))
		scoremap=np.loadtxt(join(resultdir,dir_ls[x],score_file))
		#score=np.tile(scoremap,[filtermap.shape[1],1]).transpose()
		if downsample>1:
			filtermap=filtermap[0:downsample]
		if sum(scoremap<0)>0:
			scoremap=scoremap-np.min(scoremap)
		weight=h5py.File(join(resultdir,dir_ls[x],weight_file),'r')
		name=weight['model_weights'].keys()[1]
		conv1_weight=np.asarray(weight['model_weights'][name][name+'_W'])
		#ft_av=np.sum(filtermap*score,axis=0)/np.sum(scoremap)
		ft_av,ft_var=weighted_var(filtermap,scoremap)
		bic=[]
		aic=[]
		m=[]
		lab=[]
		fv=ft_var.reshape(len(ft_var),1)
		for com in range(1,4):
        		gmm = GaussianMixture(n_components=com, covariance_type='diag').fit(fv)
        		bic.append(gmm.bic(fv))
        		aic.append(gmm.aic(fv))
        		m.append(gmm.lower_bound_)
			lab.append(gmm.fit(fv))
		num_component=np.argmin(bic)+1
		if num_component==2:
			center=m[1]
			mask=np.ones(filtermap.shape)
			mask[:,(lab[1]==np.argmax(center))]=0
			mask[filtermap>0]=1
			ftmap2=filtermap*mask
			scmap2=np.tile(scoremap.reshape(len(scoremap),1),filtermap.shape[1])*mask
			ft_av,_=weighted_var(ftmap2,scmap2)
		ONIV['oniv'][dir_ls[x]]=ft_av
		ONIV['num_compo'][dir_ls[x]]=num_component
		ONIV['var'][dir_ls[x]]=ft_var
		rank=np.argsort(ft_av)
		rank=np.flipud(rank)
		out=open(join(resultdir,dir_ls[x],outfile),'w')
		for i in range(16):
			pwm=conv1_weight[rank[i]]
			pwm=pwm.reshape((pwm.shape[0],pwm.shape[2]))
			pwm=np.fliplr(pwm)
			pwm[pwm<0]=0
			pwm=pwm/np.sum(pwm,axis=0)
			pwm[np.isnan(pwm)]=0.25
			if i==0:
				out.writelines('Motif_top'+str(i+1)+'\n')
			else:
				out.writelines('\nMotif_top'+str(i+1)+'\n')				
			out.writelines('A:\t'+ '\t'.join(['%.10f' % y for y in pwm[0]])+'\n')
			out.writelines('C:\t'+ '\t'.join(['%.10f' % y for y in pwm[1]])+'\n')
			out.writelines('G:\t'+ '\t'.join(['%.10f' % y for y in pwm[2]])+'\n')
			out.writelines('T:\t'+ '\t'.join(['%.10f' % y for y in pwm[3]])+'\n')
		out.close()
    ONIV=pandas.DataFrame(ONIV)
    ONIV.to_pickle('ONIV-weighted')
		
if __name__ == "__main__":
        main()

