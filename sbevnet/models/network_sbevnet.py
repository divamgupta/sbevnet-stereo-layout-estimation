

from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import skimage
import skimage.io
import skimage.transform
import numpy as np
import time
import math
import cv2
import torchgeometry
from pytorch_propane.registry import registry


import inspect 

from .submodule import feature_extraction , convbn_3d ,convbn  
from .unet import UNet 
from .bev_costvol_utils import get_grid_one , pt_costvol_to_hmap  , warp_p_scale  , build_cost_volume 
    
    
class CostVolRefine(nn.Module):
    def __init__(self , n_inp=64):
        
        super(CostVolRefine, self).__init__()
        
        self.dres0 = nn.Sequential(convbn_3d(n_inp, 32, 3, 1, 1),
                                     nn.ReLU(inplace=True),
                                     convbn_3d(32, 32, 3, 1, 1),
                                     nn.ReLU(inplace=True))

        self.dres1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1)) 

        self.dres2 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1))
 
        self.dres3 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1)) 

        self.dres4 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1)) 
        
    
    def forward(self, cost ):
        cost0 = self.dres0(cost)
        cost0 = self.dres1(cost0) + cost0
        cost0 = self.dres2(cost0) + cost0 
        cost0 = self.dres3(cost0) + cost0 
        cost0 = self.dres4(cost0) + cost0
        return cost0 

        

        
        
class StereoBEVFeats(nn.Module):
    def __init__(self , img_h , n_ch=32 , reduce_mode='concat'):
        super(StereoBEVFeats, self).__init__()

        self.reduce_mode = reduce_mode

        if reduce_mode == 'concat':
            self.dres4_2d_top = nn.Sequential(convbn(n_ch*img_h//4  ,  256 , 3, 1, 1 , 1 ),
                                    nn.ReLU(inplace=True),
                                    convbn( 256 ,  256 , 3, 1, 1 , 1 )) 
        elif reduce_mode == 'sum':
            self.dres4_2d_top = nn.Sequential(convbn(n_ch  ,  256 , 3, 1, 1 , 1 ),
                                    nn.ReLU(inplace=True),
                                    convbn( 256 ,  256 , 3, 1, 1 , 1 )) 
        
        self.dres4_2d_top_2 = nn.Sequential(convbn( 256  ,  256 , 3, 1, 1 , 1 ),
                                   nn.ReLU(inplace=True),
                                   convbn( 256 ,  256 , 3, 1, 1 , 1 )) 
        self.seg_up = nn.UpsamplingBilinear2d(scale_factor=4 )


        
    def forward(self , cost0 , sys_confs  , cam_confs ):
        
        fea = cost0
        fea = fea.permute(0 , 1 , 3 , 4 , 2 )
        fea = fea.contiguous()
        if self.reduce_mode == 'concat':
            fea = fea.view( fea.size(0) , fea.size(1)*fea.size(2) , fea.size(3) , fea.size(4) )
        elif self.reduce_mode == "sum":
            fea = fea.sum( 2 )
        else:
            assert False 
        
        fea = self.dres4_2d_top( fea )
        fea = self.seg_up( fea )
        fea = self.dres4_2d_top_2( fea )
        
        fea = pt_costvol_to_hmap( fea , cam_confs , sys_confs=sys_confs  )
        
        return fea 
        
        
        
        

class BEVSegHead(nn.Module):
    def __init__(self , nnn , n_classes_seg , bev_size=128   ):
        super(BEVSegHead, self).__init__()
        
        self.dres_2d_seg_1 = nn.Sequential(convbn( nnn  ,  nnn , 3, 1, 1 , 1 ),
                                   nn.ReLU(inplace=True),
                                   convbn( nnn ,  nnn , 3, 1, 1 , 1 )) 
        
        self.dres_2d_seg_2 = nn.Sequential(convbn( nnn  ,  nnn , 3, 1, 1 , 1 ),
                                   nn.ReLU(inplace=True),
                                   convbn( nnn ,  nnn , 3, 1, 1 , 1 )) 
        
        self.dres_2d_seg_3 = nn.Sequential(convbn( nnn  ,  nnn , 3, 1, 1 , 1 ),
                                   nn.ReLU(inplace=True),
                                   convbn( nnn ,  nnn , 3, 1, 1 , 1 )) 
        
        self.unet = UNet(inp_shape=bev_size , n_channels=nnn )
        
        
        self.classify_seg =  nn.Conv2d(nnn  , n_classes_seg , kernel_size=3, padding=1, stride=1,bias=False)

        
    def forward(self , fea ):
        
        fea = self.unet( fea )
        fea = self.dres_2d_seg_1(fea) + fea
        fea = self.dres_2d_seg_2(fea) + fea
        fea = self.dres_2d_seg_3(fea) + fea
        
        fea1 = self.classify_seg(fea)
        pred_seg2 = F.log_softmax(fea1)
        return pred_seg2 
        

        



        



@registry.register_network("sbevnet")
class SBEVNet(nn.Module):
    def __init__(self, image_w , image_h , xmin , xmax , ymin , ymax  , n_hmap , 
                 max_disp=64 , cx=None , cy=None , f=None , tx=None , 
                 camera_ext_x=None , camera_ext_y=None , 
                 n_classes_seg = 25 , 
                 do_ipm_rgb=True , 
                 do_ipm_feats=True , 
                 fixed_cam_confs=False  , 
                 reduce_mode='concat' 
                  ):
        super(SBEVNet, self).__init__()


        sys_confs = {
            "img_w": image_w  , 
            "img_h" : image_h   , 
            "xmin" : xmin  ,
            "xmax" : xmax  , 
            "ymin" : ymin ,
            "ymax" : ymax  , 
            "max_disp" : max_disp  , 
            
            "cx" : cx , 
            "cy" : cy , 
            "f" : f, 
            "tx" : tx  , 
            "camera_ext_x": camera_ext_x  , 
            "camera_ext_y": camera_ext_y , 
            "n_hmap": n_hmap  
            
        }

        maxdisp = max_disp 
        
        localss = locals()
        self.netowork_config = { arg: localss[arg] for arg in inspect.getfullargspec(self.__init__).args if arg != 'self'}
        print( "Network args: " , self.netowork_config  )
        
        self.maxdisp = maxdisp
        
        assert maxdisp == sys_confs['max_disp']
        
        
        self.feature_extraction = feature_extraction()
        
        self.n_classes_seg = n_classes_seg

        self.do_ipm_rgb = do_ipm_rgb 
        self.do_ipm_feats = do_ipm_feats
        
        self.sys_confs = sys_confs 
        self.fixed_cam_confs = fixed_cam_confs 



       
        nnn = 256
        
        if do_ipm_rgb:
            nnn += 3  
            
        if do_ipm_feats:
            nnn += 32
            
            
        self.bev_seg_head = BEVSegHead( nnn , n_classes_seg , bev_size=sys_confs['n_hmap']    )


      
        self.cost_vol_refine = CostVolRefine(n_inp=64)

        sterr_n_ch = 32

        self.ster_bev_feats = StereoBEVFeats(img_h = sys_confs['img_h'], reduce_mode=reduce_mode , n_ch=sterr_n_ch  )

        self.down4x = nn.AvgPool3d( 4 )

    def forward(self, data , feats_ret={} ):

        ret = {}
        
        imgs = data['input_imgs']

        left = imgs[0]
        right = imgs[1]

        
        if self.do_ipm_rgb:
            img_ipm = data['ipm_rgb']
            
        if not self.fixed_cam_confs:
            cam_confs = data['cam_confs']
            assert cam_confs.shape[-1] == 4 
            assert len(cam_confs.shape) == 2 
        else:
            cam_conf = [self.sys_confs['f'] , self.sys_confs['cx'] , self.sys_confs['cy'] , self.sys_confs['tx']]
            bs = left.shape[0]
            cam_confs = [cam_conf]*bs 
            
        if self.do_ipm_feats: 
            ipm_m = data['ipm_feats_m']
            assert ipm_m.shape[-1] == 3*3 + 2 
            assert len(ipm_m.shape) == 2 
                        
        refimg_fea     = self.feature_extraction(left)
        targetimg_fea  = self.feature_extraction(right)

        
        if self.do_ipm_feats:
            feat_ipm = warp_p_scale( refimg_fea , ipm_m , self.sys_confs  )
        
       
        cost = build_cost_volume(refimg_fea , targetimg_fea , self.maxdisp  )
       
        cost0 = self.cost_vol_refine(cost)
        
        fea = self.ster_bev_feats(cost0 , sys_confs=self.sys_confs ,cam_confs=cam_confs )

        if self.do_ipm_rgb:
            fea = torch.cat( [ fea ,  img_ipm ] , dim=1 )
        
        if self.do_ipm_feats:
            fea = torch.cat( [ fea , feat_ipm  ] , dim=1 )
    

        pred_seg = self.bev_seg_head(fea)
        ret['top_seg'] =  pred_seg
           
        
        return ret 

        