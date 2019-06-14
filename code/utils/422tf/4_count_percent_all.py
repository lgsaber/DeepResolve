import pandas as pd
import matplotlib
matplotlib.use('Agg')
import sys,os,numpy as np,h5py
from os.path import join,dirname,basename,exists
from os import system
import matplotlib.pyplot as plt

def main():
    if len(sys.argv) < 3:
        print "you must call program as: python filtermap.py <valuefile><output>"
        sys.exit(1)
    valuedir=sys.argv[1]
    resultfile=sys.argv[2]
    chart=pd.read_csv(valuedir,sep='\t',index_col=0)
    judge05=(chart<=0.5)
    judge005=(chart<=0.05)
    top1_05=sum(judge05.loc['Motif_top1'])
    top3_05=sum(judge05.loc['Motif_top1']+judge05.loc['Motif_top2']+judge05.loc['Motif_top3'])
    top5_05=sum(judge05.loc['Motif_top1']+judge05.loc['Motif_top2']+judge05.loc['Motif_top3']+judge05.loc['Motif_top4']+judge05.loc['Motif_top5'])
    all_05=sum(judge05.sum(axis=0)>0)
    top1_005=sum(judge005.loc['Motif_top1'])
    top3_005=sum(judge005.loc['Motif_top1']+judge005.loc['Motif_top2']+judge005.loc['Motif_top3'])
    top5_005=sum(judge005.loc['Motif_top1']+judge005.loc['Motif_top2']+judge005.loc['Motif_top3']+judge005.loc['Motif_top4']+judge005.loc['Motif_top5'])
    all_005=sum(judge005.sum(axis=0)>0)
    fout=open(resultfile,'w')
    #fout.writelines("\tall")
    #print "q<0.5\t%d\n" % all_05
    #fout.writelines("q<0.5\t%d\n" % (all_05))
    #print "q<0.05\t%d\n" % (all_005)
    #fout.writelines("q<0.05\t%d\n" % (all_005))
    #fout.close()
    fout.writelines("\ttop1\ttop3\ttop5\nall")
    print "q<0.5\t%d\t%d\t%d\t%d\n" % (top1_05,top3_05,top5_05,all_05)
    fout.writelines("q<0.5\t%d\t%d\t%d\t%d\n" % (top1_05,top3_05,top5_05,all_05))
    print "q<0.05\t%d\t%d\t%d\t%d\n" % (top1_005,top3_005,top5_005,all_005)
    fout.writelines("q<0.05\t%d\t%d\t%d\t%d\n" % (top1_005,top3_005,top5_005,all_005))
    fout.close()

if __name__ == "__main__":
        main()

