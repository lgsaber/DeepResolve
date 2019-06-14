import matplotlib
matplotlib.use('Agg')
import sys,os,numpy as np,h5py,pandas
from os.path import join,dirname,basename,exists
from os import system
import matplotlib.pyplot as plt

def main():
    if len(sys.argv) < 5:
        print "you must call program as: python filtermap.py <rootdir><resultdir><matching result dir><output prefix>"
        sys.exit(1)
    rootdir=sys.argv[1]
    resultdir=sys.argv[2]
    matchdir=sys.argv[3]
    out_pref=sys.argv[4]
    dir_ls=os.listdir(rootdir)
    dir_ls.sort()
    for x in range (len(dir_ls)):
      if dir_ls[x][0:15]=='wgEncodeAwgTfbs':
	print(dir_ls[x])
	all_stat=pandas.read_csv(join(resultdir,dir_ls[x],matchdir,'tomtom.txt'),sep='\t',index_col=0)
	q_val=all_stat['q-value']
	e_val=all_stat['E-value']
	q_val.name=dir_ls[x]
	e_val.name=dir_ls[x]
	if x==0:
		Q=q_val
		E=e_val
	else:
		Q=pandas.concat([Q,q_val],axis=1)
		E=pandas.concat([E,e_val],axis=1)
    Q.to_csv('Q-value'+out_pref,sep='\t',na_rep='nan')
    E.to_csv('E-value'+out_pref,sep='\t',na_rep='nan')	
		
if __name__ == "__main__":
        main()

