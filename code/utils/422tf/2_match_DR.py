import matplotlib
matplotlib.use('Agg')
import sys,os,numpy as np,h5py,pandas
from os.path import join,dirname,basename,exists
from os import system
import matplotlib.pyplot as plt

#basedir='/cluster/zeng/research/recomb/generic/'

def main():
    if len(sys.argv) < 3:
        print "you must call program as: python NAME.py <method> <rootdir><resultdir><filtermap>"
        sys.exit(1)
    rootdir=sys.argv[1]
    resultdir=sys.argv[2]
    topmotif=sys.argv[3]
    dir_ls=os.listdir(rootdir)
    dir_ls.sort()
    MOTIF=pandas.read_csv('/cluster/geliu/CNN_visualization/motif/map',sep='\t',index_col=0,header=None)
    for x in range (len(dir_ls)):
       if dir_ls[x][0:15]=='wgEncodeAwgTfbs':
         print(dir_ls[x])
	 if os.path.exists(join(resultdir,dir_ls[x],topmotif)):
	    print('Exist!conveting!')
	    motif=MOTIF.loc[dir_ls[x]].get_values()
	    print join(resultdir,dir_ls[x],topmotif)
	    system('uniprobe2meme '+join(resultdir,dir_ls[x],topmotif)+' >'+join(resultdir,dir_ls[x],topmotif+'.meme'))
	    print('comparing!')
	    system('tomtom -incomplete-scores -thresh 1 -eps -o '+join(resultdir,dir_ls[x],'match_'+topmotif)+' '+join(resultdir,dir_ls[x],topmotif+'.meme')+' '+motif[0])
		
if __name__ == "__main__":
        main()

