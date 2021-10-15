
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import inspect
import torch.nn.functional as F
import json

from pytorch_propane.models import Model 
from pytorch_propane.callbacks import ModelCheckpoint

from pytorch_propane.registry import registry

from ..train_utils import IsDegenerateCallback


@registry.register_model("sbevnet_model")
def sbevnet_model( network  , check_degenerate=False   ):
    
    
    localss = locals()
    model_config = { arg: localss[arg] for arg in inspect.getfullargspec( sbevnet_model ).args if arg != 'self' and arg != 'network'}
    model_config["model_name"] = "sbevnet_model"
    print( "Model args: " , model_config  )
    
    model = Model( network=network )
    model.compile( 'adam' , cuda=True )
    
    
    loss_fn =  nn.NLLLoss2d( ignore_index = -100 ) 
    model.add_loss(  loss_fn , output_key='top_seg' , display_name='bev_seg_nll')

    if check_degenerate:
        model.add_callback( IsDegenerateCallback( )  )
        
    return model   


