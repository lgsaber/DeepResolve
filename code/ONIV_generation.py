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
    filtermap_dir=sys.argv[1]
    scoremap_dir=sys.argv[2]
    weight_dir=sys.argv[3]
    resultdir=sys.argv[4]
    ONIV={}
    if not exists(filtermap_dir) or not exists(scoremap_dir):
	print('Directory not exits!')
    else:
	filtermap=np.loadtxt(filtermap_dir)
	scoremap=np.loadtxt(scoremap_dir)
	if sum(scoremap<0)>0:
		scoremap=scoremap-np.min(scoremap)
	weight=h5py.File(weight_dir,'r')
	name=weight['model_weights'].keys()[1]
	conv1_weight=np.asarray(weight['model_weights'][name][name+'_W'])
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
	ONIV['oniv']=ft_av
	ONIV['num_compo']=num_component
	ONIV['var']=ft_var
	rank=np.argsort(ft_av)
	rank=np.flipud(rank)
    ONIV=pandas.DataFrame(ONIV)
    ONIV.to_pickle(join(resultdir,'ONIV.plk'))
	
if __name__ == "__main__":
    main()

