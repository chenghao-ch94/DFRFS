import torch
import numpy as np
import random


class CategoriesSampler_SetA():

    def __init__(self, label, n_batch, n_cls, n_per, domain, d_id, s_id):
        self.n_batch = n_batch# the number of iterations in the dataloader
        self.n_cls = n_cls
        self.n_per = n_per  # 5+1 or 5+15
        self.domain = domain
        
        if isinstance(d_id,int):           # always true for test step
            self.d_id = d_id
        else:
            self.d_id = np.array(d_id)
        self.s_id = np.array(s_id)

        label = np.array(label)#all data label
        self.domain = np.array(self.domain)
        self.m_ind = []#the data index of each class
        assert len(label)==len(self.domain)
        
        label_u = np.unique(label)

        for i in range(len(label_u)):
            ind = np.argwhere(label == label_u[i]).reshape(-1)# all data index of this class
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)

    def __len__(self):
        return self.n_batch
    
    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = []
            classes = torch.randperm(len(self.m_ind))[:self.n_cls]#random sample num_class indexs,e.g. 5-way
            # print(classes)
            d_ind = 0
            s_id = [j for j in range(len(self.s_id))]
            random.shuffle(s_id)
            for c in classes:
                l = self.m_ind[c]#all data indexs of this class
                dl = self.domain[l]#all domain label of this class

                # select support samples from train domains (5) to construct support set
                assert (isinstance(self.d_id, int) == True)

                if self.n_per == 16:    #5-way 1-shot: random select 1 support sample from random domain
                    domain_s = s_id[d_ind]   #selected domain for this class
                    l2 = np.argwhere(dl==domain_s)
                    assert len(l2)>0
                    pos1 = torch.randperm(len(l2))[:1]

                elif self.n_per == 20:  # 5-way 5-shot 
                    temp=[]
                    for i in range(len(self.s_id)):
                        l2 = np.argwhere(dl==self.s_id[i])
                        pos1 = torch.randperm(len(l2))[:1] #sample 1 support index of each domain

                        if len(temp) == 0:
                            temp = pos1
                        else:
                            temp = torch.cat([temp, pos1],dim=0)
                    pos1 = temp                    

                else:
                    assert 1==0
                
                # select query samples from test domains to construct query set (same query setting)
                l2 = np.argwhere(dl==self.d_id)  #all data indexs of this class with certain test domain
                pos2 = torch.randperm(len(l2))[:15] #sample 15 query data index of this class
                pos1 = torch.cat([pos1,pos2],dim=0)
                batch.append(l[pos1])
                d_ind += 1
    
            batch = torch.stack(batch).t().reshape(-1)
            # print(batch)
            # .t() transpose,
            # due to it, the label is in the sequence of abcdabcdabcd form after reshape,
            # instead of aaaabbbbccccdddd
            yield batch

class CategoriesSampler_Dom():

    def __init__(self, label, n_batch, n_cls, n_per, domain, d_id):
        self.n_batch = n_batch# the number of iterations in the dataloader
        self.n_cls = n_cls
        self.n_per = n_per  # 5+1 or 5+15
        self.domain = domain
        
        if isinstance(d_id,int):
            self.d_id = d_id
        else:
            self.d_id = np.array(d_id)

        label = np.array(label)#all data label
        self.domain = np.array(self.domain)
        self.m_ind = []#the data index of each class
        assert len(label)==len(self.domain)
        
        label_u = np.unique(label)

        for i in range(len(label_u)):
            ind = np.argwhere(label == label_u[i]).reshape(-1)# all data index of this class
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)

    def __len__(self):
        return self.n_batch
    
    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = []
            classes = torch.randperm(len(self.m_ind))[:self.n_cls]#random sample num_class indexs,e.g. 5
            for c in classes:
                l = self.m_ind[c]#all data indexs of this class
                dl = self.domain[l]#all domain label of this class

                if isinstance(self.d_id, int):      # test set with only one domain, same setting with original few-shot
                    pos = torch.randperm(len(l))[:self.n_per] #sample n_per data index of this class
                    batch.append(l[pos])

                else:                # train and val set with five domains
                    assert(len(self.d_id)==5)
                    if self.n_per == 16:   # 1-shot, 1 support for 1 random domain and 3 query for each domain
                        in_d = [j for j in range(len(self.d_id))]
                        random.shuffle(in_d)
                        d_n = self.d_id[in_d]
                        domain_s = d_n[0]            # the selected domain for 1 support sample
                        # print('domain label',dl)
                        # print('choose', domain_s)
                        l2 = np.argwhere(dl==domain_s)
                        assert len(l2)>0
                        pos1 = torch.randperm(len(l2))[:1]

                    elif self.n_per == 20: # 5-shot, 1 support and 3 query for each domain
                        temp=[]
                        for i in range(len(self.d_id)):
                            l2 = np.argwhere(dl==self.d_id[i])
                            pos1 = torch.randperm(len(l2))[:1] #sample 1 support index of each domain

                            if len(temp) == 0:
                                temp = pos1
                            else:
                                temp = torch.cat([temp, pos1],dim=0)
                        pos1 = temp
                    
                    for i in range(len(self.d_id)):
                        l2 = np.argwhere(dl==self.d_id[i])  #all data indexs of this class with certain domain
                        pos2 = torch.randperm(len(l2))[:3] #sample 1 support and 3 query data index of this class
                        pos1 = torch.cat([pos1,pos2],dim=0)

                    batch.append(l[pos1])

            batch = torch.stack(batch).t().reshape(-1)
            # .t() transpose,
            # due to it, the label is in the sequence of abcdabcdabcd form after reshape,
            # instead of aaaabbbbccccdddd
            yield batch