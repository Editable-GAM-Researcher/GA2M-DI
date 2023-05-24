import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict



class res_block(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(res_block, self).__init__()
        self.linear_layer = nn.Linear(input_size, output_size)
        self.res_layer = nn.Linear(input_size, output_size)
        self.bn = nn.BatchNorm1d(output_size)

    def forward(self, x):
        x = F.relu(self.bn(self.linear_layer(x))) + self.res_layer(x)
        return x


class subNet(torch.nn.Module):
    def __init__(self, outputsize, layerSizeList):
        super(subNet, self).__init__()
        full_dim_list = [1]
        full_dim_list.extend(layerSizeList)
        full_dim_list.append(outputsize)
        blk_list = []
        for idx in range(len(full_dim_list)-1):
            res_blk = res_block(full_dim_list[idx],full_dim_list[idx+1])
            blk_list.append(res_blk)
        
        self.model = nn.Sequential(*blk_list)
        
    def forward(self, x):
        return self.model(x)

    
    
class GA2M(torch.nn.Module):
    def __init__(self, feature_num, layerSizeList, basis_num=128,interaction_list=None):
        super(GA2M, self).__init__()
        self.basis_num = basis_num
        self.feature_num=feature_num
        if interaction_list is None:
            self.permutation_num = int(self.feature_num*(self.feature_num-1)/2)
        else:
            self.permutation_num = len(interaction_list)
        self.interaction_map = self._generate_interaction_map(interaction_list)
        self.comb_layer = nn.Linear(self.permutation_num, 1)
        self.first_order_attention_weights = nn.Parameter(torch.zeros((1,self.feature_num,self.basis_num+1),requires_grad=True))
        self.second_order_attention_weights = nn.Parameter(torch.zeros((1,self.permutation_num,self.basis_num+1),requires_grad=True))
        self.netList = nn.ModuleList([subNet(outputsize=self.basis_num, layerSizeList=layerSizeList)
                                      for i in range(self.feature_num)])
        self.final_bn = nn.BatchNorm1d(self.feature_num)
    
    def _generate_interaction_map(self, interaction_list):
        if interaction_list is None:
            interaction_list = []
            for ii in range(self.feature_num-1):
                for jj in range(ii+1,self.feature_num):
                    interaction_list.append((ii,jj))
        interaction_list_permed = []
        for (ii,jj) in interaction_list:
            if np.random.randn(1) > 0:
                interaction_list_permed.append((ii,jj))
            else:
                interaction_list_permed.append((jj,ii))
        interaction_map = OrderedDict()
        for (ii,jj) in interaction_list_permed:
            if ii not in interaction_map.keys():
                interaction_map[ii] = [jj]
            else:
                interaction_map[ii].append(jj)
        return interaction_map

    def forward(self, x):
        latent = []
        for idx, subnet in enumerate(self.netList):
            latent.append(subnet(x[:, idx].unsqueeze(1)).unsqueeze(1))
        self.latent = torch.cat(latent, axis=1)
        self.latent = self.final_bn(self.latent)
            
        First_order_attention=F.softmax(self.first_order_attention_weights,dim=-1)
        main_effect = (self.latent)*First_order_attention[:,:,1:]
        x_main_effect = torch.mean(torch.sum(main_effect,axis=-1),dim=-1)

        Second_order_attention=F.softmax(self.second_order_attention_weights,dim=-1)
        x_interactions = 0
        attent_count = 0
        for idx, related_list in self.interaction_map.items():
            x_interactions += self.latent[:,idx,:]*torch.sum(Second_order_attention[:,attent_count:attent_count+len(related_list),1:]*self.latent[:,related_list,:],dim=1)
            attent_count += len(related_list)
        x_interactions = torch.sum(x_interactions,axis=-1)/self.permutation_num

        x_out = x_main_effect + x_interactions
        return x_out
    