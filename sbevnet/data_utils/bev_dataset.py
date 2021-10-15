#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import torch
import torch.utils.data as data
import torch
import torchvision.transforms as transforms
import random
from PIL import Image, ImageOps
import numpy as np
from . import preprocess
import cv2
import json 
import inspect 

from pytorch_propane.data_utils import ComposeDatasetDict 
from pytorch_propane.registry import registry



IMG_EXTENSIONS = [
    '.jpg',
    '.JPG',
    '.jpeg',
    '.JPEG',
    '.png',
    '.PNG',
    '.ppm',
    '.PPM',
    '.bmp',
    '.BMP',
    ]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in
               IMG_EXTENSIONS)


def default_loader(path):
    return Image.open(path).convert('RGB')


def disparity_loader(path):
    return Image.open(path)



def load_mask( x ):
    return torch.from_numpy(cv2.imread(x, cv2.IMREAD_UNCHANGED).T > 0 ).long()

# rgb_imgs - list of lists [ left_imgs , right_imgs , center_imgs  ]


class ImgsLoader(data.Dataset):
    
    def __init__(self , rgb_imgs  , th , tw , loader=default_loader ):
        self.rgb_imgs = rgb_imgs
        self.loader = loader
        self.th = th
        self.tw = tw
        
    def __getitem__(self, index):
        
        rgb = []
        for rr in self.rgb_imgs:
            rgb.append(rr[index])

        rgb_imgs = list(map(lambda x: self.loader(x), rgb))
        
        (w, h) = rgb_imgs[0].size
        th = self.th
        tw = self.tw 
        
        assert w/h == tw / th , (w/h , tw/th) # ratio shall not change 
        
        if w != h:
            rgb_imgs = list(map(lambda x: x.resize(( tw , th )), rgb_imgs))

        processed = preprocess.get_transform(augment=False)
        rgb_imgs = list(map(lambda x: processed(x).float(), rgb_imgs))

        return rgb_imgs

    def __len__(self):
        return len( self.rgb_imgs[0])
            
            
        
class IPMLoader(data.Dataset):
    
    def __init__(self , ipm_img,  loader=default_loader):
        self.ipm_img = ipm_img 
        self.loader = loader
        
    def __getitem__(self, index):
        imp_img  = self.loader( self.ipm_img[ index ] )
        processed = preprocess.get_transform(augment=False)
        imp_img = processed( imp_img ).permute(0 , 2 , 1)
        
        return imp_img.float()
    
    def __len__(self):
        return len( self.ipm_img  )
        

class NPArrayLoader(data.Dataset):
    
    def __init__(self , f_list ):
        self.f_list = f_list
        
    def __getitem__(self, index):
        return np.load( self.f_list[ index] ).astype('float32')
    
    def __len__(self):
        return len( self.f_list  )
    
    
class MaskLoader(data.Dataset):
    
    def __init__(self , f_list ):
        self.f_list = f_list
        
    def __getitem__(self, index):
        return load_mask(self.f_list[index])
    
    def __len__(self):
        return len( self.f_list  )
    
    
class HMapLoader(data.Dataset):
    
    def __init__(self , f_list , hmap_max  ):
        self.f_list = f_list
        self.hmap_max = hmap_max
        
    def __getitem__(self, index):
        hmap = self.f_list[index]
        hmap = cv2.imread(  hmap , cv2.IMREAD_UNCHANGED ) 
        hmap = hmap.astype(np.float32)
        hmap = np.clip(hmap , 0 ,  self.hmap_max )
        hmap = hmap/self.hmap_max
        hmap = hmap.T
    
    def __len__(self):
        return len( self.f_list  )

        

class SegLoader(data.Dataset):
    
    def __init__(self , f_list ,mask_segs=False , explicit_mask=None , resize=None , do_transpose=False  ):
        self.f_list = f_list
        self.mask_segs = mask_segs
        self.explicit_mask = explicit_mask
        self.resize = resize
        self.do_transpose = do_transpose
        
    def __getitem__(self, index):
        
        seg_img = cv2.imread( self.f_list[index] , cv2.IMREAD_UNCHANGED)

        if len( seg_img.shape ) == 3 :
            seg_img = seg_img[: , : , 2 ]

        assert len(seg_img.shape) == 2 
        
        if self.do_transpose:
            seg_img = seg_img.T 
        
        if not self.resize is None:
            seg_img = cv2.resize( seg_img , self.resize , interpolation=cv2.INTER_NEAREST)
        
        seg_img = torch.from_numpy( seg_img ).long()
        
        
        if self.mask_segs:
            if not self.explicit_mask is None:
                mask = load_mask(self.explicit_mask[index])
                seg_img[mask<0.5] = -100
            else:
                seg_img[seg_img<0.5] = -100
                
        return seg_img

    def __len__(self):
        return len( self.f_list  )
        
    
    

        
def disparity_loader_hccc(path):
    
    r =  cv2.imread(path , cv2.IMREAD_UNCHANGED)
    r = np.ascontiguousarray(r,dtype=np.float32)
    r = (r - 32767)/16
    r = r/4 # as we had reduced the img size by that!!!! 
    return r # as the data loader will mul it by 256 after this LOL



class DispImgLoader(data.Dataset):
    
    def __init__(self , f_list ,dploader=disparity_loader , resize=None  ):
        self.f_list = f_list
        self.dploader = dploader 
        self.resize = resize 
        
    def __getitem__(self, index):
        dataL = self.dploader(self.f_list[index])
        
        if not self.resize is None:
            if dataL.shape[0] == self.resize[1] and dataL.shape[1] == self.resize[0]:
                pass
            else:
                assert dataL.shape[0]/ dataL.shape[1] == self.resize[1]/self.resize[0] ,  "ratio shall not change "
                dataL = cv2.resize(dataL , self.resize , interpolation=cv2.INTER_NEAREST )
                ratio =  dataL.shape[0]/self.resize[1] 
                dataL = dataL/ratio 
                
        dataL = np.ascontiguousarray(dataL,dtype=np.float32)
        
        return dataL

    def __len__(self):
        return len( self.f_list  )
        
        
    



@registry.register_dataset("sbevnet_dataset_main")
def sbevnet_dataset(

    datapath , dataset_split  , do_ipm_rgb=False , 
    do_ipm_feats=False , fixed_cam_confs=True , 
    do_mask=True ,  do_top_seg=True ,  
    zero_mask=False ,
    image_w = 512 , image_h=288 

    ):
    

    localss = locals()
    print( "dataset argsss : " ,  { arg: localss[arg] for arg in inspect.getfullargspec(sbevnet_dataset ).args if arg != 'self'}) 


    jj = json.loads(open( datapath ).read()) 

    rootp = os.path.dirname( datapath )

    for s in ['train' , 'test']:
        for k in jj[s]:
            jj[s][k] = list( map( lambda x: os.path.join(rootp , x)  ,   jj[s][k]  ))

    sub_datasets = {}

    
    sub_datasets['input_imgs'] = ImgsLoader( [ jj[dataset_split]["rgb_left"]  ,jj[dataset_split]["rgb_right"] ] , tw=image_w , th=image_h , loader=default_loader )

    
    if do_mask:
        mask = jj[dataset_split]["mask"] 
        sub_datasets['mask'] = MaskLoader(mask)
    else:
        mask = None 

    mask_imgs=(zero_mask or do_mask)
        
    if do_top_seg:
        sub_datasets['top_seg'] = SegLoader(jj[dataset_split]["top_seg"]  ,mask_segs=mask_imgs , explicit_mask=mask , resize=None , do_transpose=True  )
    
    
        
    if  do_ipm_rgb:
        sub_datasets['ipm_rgb'] =  IPMLoader( jj[dataset_split]["top_ipm"]  )

    if do_ipm_feats:
        sub_datasets['ipm_feats_m']= NPArrayLoader(  jj[dataset_split]["top_ipm_m"]  )

    if not fixed_cam_confs:
        sub_datasets['cam_confs']= NPArrayLoader(   jj[dataset_split]["confs"]  ) 

   
    return ComposeDatasetDict( sub_datasets , ret_double=True )
    