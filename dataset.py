import numpy as np
import torch
from torch.utils.data import Dataset
from pyfasta import Fasta
from config import *
import random
import pandas as pd

class GenomicData(Dataset):
    def __init__(self, cellline_trn_num,path='/home/liuqiao/software/DeepCAGE/data/encode'):
        self.path = path
        self.genome = Fasta('%s/genome.fa'%path)
        pd_openness = pd.read_csv('%s/readscount_normalized_filtered.csv'%path,header=0,index_col=[0],sep='\t')
        self.pd_openness = np.log(pd_openness+1)

        pd_tf_gexp = pd.read_csv('%s/preprocessed/tf_gexp.csv'%path ,sep='\t',header=0,index_col=[0])
        self.pd_tf_gexp = np.log(pd_tf_gexp+1)

        self.pd_tf_bs = pd.read_csv('%s/preprocessed/tf_motif_score.csv'%path,sep='\t',header=0,index_col=[0])
        np.random.seed(123)
        train_idx = np.random.choice(np.arange(self.pd_openness.shape[1]),size=cellline_trn_num,replace=False)
        self.train_dseq_celllines = np.array(list(self.pd_openness.columns))[train_idx]
        self.test_dseq_celllines = [item for item in self.pd_openness.columns if item not in self.train_dseq_celllines]

        self.train_rseq_celllines = np.array(list(self.pd_tf_gexp.index))[train_idx]
        self.test_rseq_celllines = [item for item in self.pd_tf_gexp.index if item not in self.train_rseq_celllines]
        assert self.pd_openness.shape[1] == len(self.train_dseq_celllines)+len(self.test_dseq_celllines)
        assert self.pd_tf_gexp.shape[0] == len(self.train_rseq_celllines)+len(self.test_rseq_celllines)


    def get_seq_meta(self,file_path):
        pd_openness = pd.read_csv(file_path,header=0,index_col=[0],sep='\t',dtype='float16')
        seq_info, cellline_info = openness.index, openness.columns
        openness_mat = pd_openness.values
        return openness_mat,seq_info,cellline_info

    def get_tf_meta(self, gexp_file, bs_file):
        pd_tf_gexp = pd.read_csv(gexp_file ,sep='\t',header=0,index_col=[0],dtype='float16')
        pd_tf_bs = pd.read_csv(bs_file,sep='\t',header=0,index_col=[0],dtype='float16')
        tf_names = pd_tf_gexp.columns

    def get_seq_from_meta(self,region_idx):
        seq_info = self.pd_openness.index[region_idx*INPUT_LEN:(region_idx+1)*INPUT_LEN]
        #print(seq_info)
        seq_list = []
        for info in seq_info:
            chrom, start, end = info.split(':')[0],int(info.split(':')[1].split('-')[0]),int(info.split(':')[1].split('-')[1])
            seq_list.append(self.genome[chrom][start:end])
        return seq_list
        

    def get_seq_embeds(self, seq, k=4):
        dic = {'a':0, 'A':0, 'c':1, 'C':1, 'g':2, 'G':2, 't':3, 'T':3, 'N':4, 'n':4}
        assert len(seq) > k
        kmer_feat = np.zeros((4**k,),dtype='float16')
        for i in range(len(seq)-k+1):
            sub_seq = seq[i:(i+k)]
            if 'N' not in sub_seq and 'n' not in sub_seq:
                idx = sum([dic[char]*4**(k-i-1) for i,char in enumerate(sub_seq)])
                kmer_feat[idx]+=1
        return kmer_feat
    
    def get_tf_state(self,region_idx, cellline_idx):
        tf_gexp_vec = self.pd_tf_gexp.loc[self.train_rseq_celllines[cellline_idx]].values #(711,)
        tf_gexp_feat = np.tile(tf_gexp_vec,(INPUT_LEN,1))
        seq_info = self.pd_openness.index[region_idx*INPUT_LEN:(region_idx+1)*INPUT_LEN]
        seq_extend_info=[]
        for info in seq_info:
            chrom, start, end = info.split(':')[0],int(info.split(':')[1].split('-')[0]),int(info.split(':')[1].split('-')[1])
            seq_extend_info.append('%s:%d-%d'%(chrom,start-400,end+400))
        tf_bs_feat = self.pd_tf_bs.loc[seq_extend_info].values  # (50, 711)
        #print('a',tf_gexp_feat[0,:],tf_gexp_feat[1,:])
        #print('b',tf_bs_feat[0,:])
        return tf_gexp_feat*tf_bs_feat



    def __getitem__(self, index):
        region_idx = index // len(self.train_dseq_celllines)
        cellline_idx = index % len(self.train_dseq_celllines)
        seq_list = self.get_seq_from_meta(region_idx)
        assert len(seq_list) == INPUT_LEN
        seq_embeds_list = map(self.get_seq_embeds,seq_list)
        seq_embeds = np.stack(seq_embeds_list)
        tf_feats = self.get_tf_state(region_idx,cellline_idx)
        inputs_embeds = np.concatenate([seq_embeds,tf_feats],axis=1)
        #print(self.train_dseq_celllines[cellline_idx])
        #print(self.train_rseq_celllines[cellline_idx])
        targets_openness = self.pd_openness[self.train_dseq_celllines[cellline_idx]].values[region_idx*INPUT_LEN:(region_idx+1)*INPUT_LEN]
        inputs_embeds = np.pad(inputs_embeds,((0, 0), (0, 1)))
        inputs_embeds = np.array(inputs_embeds,dtype='float16')#(50,967+1)
        targets_openness = np.array(targets_openness,dtype='float16')#(50,)
        targets_openness = targets_openness[:,np.newaxis]
        inputs_embeds = torch.from_numpy(inputs_embeds)
        targets_openness = torch.from_numpy(targets_openness)
        return (inputs_embeds, targets_openness)

    def __len__(self):
        return len(self.train_dseq_celllines)*(self.pd_openness.shape[0] // INPUT_LEN)


if __name__=="__main__":
    Mydataset = GenomicData(100)
    print(Mydataset.get_seq_embeds('AAAATTTT'))
    print(len(Mydataset))
    a,b = Mydataset[0]
    print(a.shape,b.shape)
    print(a[0,:])
    print(b)
