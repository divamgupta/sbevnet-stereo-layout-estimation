import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import time

import cv2
from tqdm import tqdm
import os 
import numpy as np

from pytorch_propane.callbacks  import Callback 


    
def is_degenerate( model , data_loader , n_iter=100 ):
    i = 0
    count_vec = np.zeros((300))
    im_size = 0 
    for batch_idx, (imgCrp_old , targets ) in tqdm(enumerate(data_loader)):
        
        imgCrp={}
        for k in imgCrp_old:
            if k == 'input_imgs':
                imgCrp[k] = list( map( lambda x :Variable(torch.FloatTensor(x[:1].float() )).cuda() , imgCrp_old[k]  ))
            else:
                imgCrp[k] = Variable(torch.FloatTensor(imgCrp_old[k][:1].float() )).cuda() 
        
        output_seg = model.network(imgCrp )
        if len(output_seg) == 2:
            output_seg , output_hmap = output_seg 
        imm = output_seg['top_seg'][0].argmax(0).cpu().numpy().astype("uint8")
        im_size = imm.shape[0]
        count_vec += np.bincount( imm.flatten() , minlength=300 )
        i += 1 
        
        if i > n_iter :
            break
    
    count_vec.sort()
    mx = count_vec[-1]
    mx2 =  count_vec[-2]
        
    
    if mx > 0.85*float(im_size*im_size*n_iter):
        print("mmm" , mx , mx2 )
        return True
    else:
        return False
    
    



class IsDegenerateCallback( Callback ):
    def __init__(self , n_iter=100    ):
        super(IsDegenerateCallback, self).__init__()
        
        self.n_iter = n_iter 
        
        
    def on_epoch_end( self , epoch , logs=None):

        dataloader = self.model._trainer_args['dataloader'] 

        n_iter = self.n_iter
        # if self.model._trainer_args['sanity'] :
        #     n_iter = 2 
        
        if is_degenerate( self.model , dataloader , n_iter=n_iter  ):
            raise ValueError("The model training collapsed. Please restart the training. ")
        else:
            print("Model did not degenerate this time! ")
            
   
        


    