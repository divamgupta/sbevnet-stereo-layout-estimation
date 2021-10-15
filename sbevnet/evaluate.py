
from pytorch_propane.callbacks import InferenceCallback
import numpy as np

from pytorch_propane.registry import registry
from pytorch_propane.function import Function 

import cv2 
import os 

class EvaluateIOU(InferenceCallback):
    
    def __init__( self , n_classes , ignore_zero=False , classes={} , seg_key=None , mask_key=None ) :
        # ignore_zero , ignoring the zero in both gt and pred 
        self.n_classes = n_classes
        self.ignore_zero = ignore_zero
        self.classes = classes 
        self.seg_key = seg_key
        self.mask_key = mask_key 
    
    def on_start( self ):
        self.tp = np.zeros(self.n_classes)
        self.fp = np.zeros(self.n_classes)
        self.fn = np.zeros(self.n_classes)
        self.n_pixels = np.zeros(self.n_classes)
    
    def __call__( self , data_x=None , data_y=None , model_output=None ):
        
        
        if not self.seg_key is None:
            data_x_or = data_x
            data_x = data_x[ self.seg_key  ]
            data_y = data_y[ self.seg_key  ]
            model_output = model_output[ self.seg_key  ]
            
        assert data_y.shape[0] == 1 # batch size should be one 
        
        gt = data_y[0].cpu().numpy().flatten()
        
        if not self.mask_key is None:
            assert not self.ignore_zero 
        
        if self.ignore_zero:
            pr = model_output[0].cpu().detach().numpy()[1:].argmax(0).flatten() + 1 
            mask = gt > 0 
            pr = pr[ mask ]
            gt = gt[ mask ]
        elif  not self.mask_key is None:
            pr = model_output[0].cpu().detach().numpy().argmax(0).flatten()
            mask = data_x_or[self.mask_key].cpu().detach().numpy().flatten() > 0.5
            pr = pr[ mask ]
            gt = gt[ mask ]
            
        else:
            pr = model_output[0].cpu().detach().numpy().argmax(0).flatten()
        
        for cl_i in range(self.n_classes):
            
            self.tp[cl_i] += np.sum((pr == cl_i) * (gt == cl_i))
            self.fp[cl_i] += np.sum((pr == cl_i) * ((gt != cl_i)))
            self.fn[cl_i] += np.sum((pr != cl_i) * ((gt == cl_i)))
            self.n_pixels[cl_i] += np.sum(gt == cl_i)
            
            
    def on_end(self):
        cl_wise_score = self.tp / (self.tp + self.fp + self.fn + 0.000000000001)
        n_pixels_norm = self.n_pixels / np.sum(self.n_pixels)
        frequency_weighted_IU = np.sum(cl_wise_score*n_pixels_norm)
        mean_IU = np.mean(cl_wise_score)
        
        for cl_name in self.classes:
            print( cl_name + " -> " + str(cl_wise_score[self.classes[cl_name]]))
        
        print( " cl_wise_score " , cl_wise_score )
        
        
        

@registry.register_function("eval_iou")
class EvaluateIOUFn(Function):
    def __init__(self):
        pass

    def execute(self , model  , eval_dataloader , dataset_type=None    ):

        if dataset_type == 'kitti':
            model.run_inference_callback( eval_dataloader , EvaluateIOU(25 , seg_key='top_seg' , mask_key='mask' 
                                                    , classes={ "road":2  , "Vegetation":4 ,"cars":5 ,  "sidewalk":7,  "buildings":8  } ) ) 
        elif dataset_type == 'carla':
            model.run_inference_callback( eval_dataloader , EvaluateIOU(25 , seg_key='top_seg' , mask_key='mask' 
                                                  , classes={ "road":7 , "sidewalk":8 , "Vegetation":9 ,"cars":10 , "buildings":1  } ) )
        else:
            model.run_inference_callback( eval_dataloader , EvaluateIOU(25 , seg_key='top_seg' , mask_key='mask' 
                                                  , classes={ } ) )






def get_colored_segmentation_image(seg_arr):
    output_height = seg_arr.shape[0]
    output_width = seg_arr.shape[1]
    n_classes = np.max(seg_arr) + 1 

    colors = [(197, 215, 20), (132, 248, 207), (155, 244, 183), (111, 71, 144), (71, 48, 128), (75, 158, 50), (37, 169, 241), (51, 181, 222), (161, 104, 244), (226, 133, 31), (7, 47, 204), (0, 252, 170), (124, 166, 32), (97, 113, 122), (72, 229, 46), (41, 163, 250), (55, 154, 149), (63, 170, 104), (147, 227, 46), (197, 162, 123), (148, 94, 96), (95, 16, 133), (243, 35, 45), (66, 76, 19), (41, 200, 141), (120, 110, 214), (140, 230, 252), (182, 42, 166), (59, 249, 171), (97, 124, 8), (138, 59, 112), (190, 87, 170), (218, 31, 51), (74, 112, 23), (37, 13, 63), (96, 61, 200), (46, 189, 59), (18, 11, 99), (94, 63, 245), (107, 31, 11), (217, 51, 133), (35, 113, 36), (154, 179, 223), (92, 31, 239), (20, 51, 200), (102, 133, 183), (240, 86, 104), (29, 81, 82), (175, 128, 60), (226, 89, 6), (241, 209, 159), (182, 198, 128), (78, 6, 234), (40, 171, 23), (143, 69, 122), (246, 180, 147), (183, 67, 158), (198, 212, 41), (0, 98, 171), (81, 122, 114), (229, 193, 212), (16, 205, 214), (23, 84, 228), (32, 132, 80), (228, 249, 0), (19, 253, 166), (159, 239, 25), (212, 96, 42), (66, 7, 205), (213, 161, 1), (109, 7, 1), (50, 97, 60), (101, 154, 143), (93, 51, 243), (203, 41, 11), (140, 231, 59), (131, 68, 177), (58, 79, 142), (9, 21, 20), (105, 132, 161), (187, 21, 253), (234, 222, 190), (91, 106, 192), (149, 4, 70), (77, 138, 170), (172, 188, 47), (173, 18, 21), (138, 83, 76), (148, 184, 202), (66, 150, 58), (244, 122, 24), (157, 91, 36), (154, 206, 168), (153, 212, 55), (50, 246, 242), (172, 175, 63), (245, 59, 254), (218, 19, 154), (171, 79, 85), (192, 44, 33), (43, 101, 113), (31, 197, 4), (50, 201, 148), (229, 250, 111), (216, 42, 188), (112, 133, 85), (220, 98, 183), (58, 32, 14), (231, 103, 60), (254, 203, 131), (106, 21, 110), (74, 53, 101), (234, 193, 185), (77, 53, 249), (75, 207, 216), (253, 165, 255), (255, 103, 112), (4, 174, 162), (164, 18, 75), (131, 79, 194), (150, 240, 33), (43, 20, 33), (115, 66, 20), (153, 7, 229), (169, 82, 76), (235, 190, 195), (17, 46, 39), (218, 105, 148), (213, 246, 198), (119, 10, 0), (93, 154, 130), (170, 33, 252), (134, 155, 208), (196, 196, 31), (83, 65, 122), (146, 171, 28), (18, 246, 213), (72, 251, 41), (77, 180, 210), (18, 238, 197), (234, 24, 51), (241, 77, 10), (16, 67, 165), (53, 177, 99), (196, 251, 56), (30, 239, 172), (63, 151, 65), (198, 150, 62), (96, 19, 200), (227, 190, 97)]


    seg_img = np.zeros((output_height, output_width, 3))

    for c in range(n_classes):
        seg_arr_c = seg_arr[:, :] == c
        seg_img[:, :, 0] += ((seg_arr_c)*(colors[c][0])).astype('uint8')
        seg_img[:, :, 1] += ((seg_arr_c)*(colors[c][1])).astype('uint8')
        seg_img[:, :, 2] += ((seg_arr_c)*(colors[c][2])).astype('uint8')

    return seg_img



class SaveSegmentationPredictions(InferenceCallback):
    
    def __init__( self , save_dir , seg_key ) :
        
        self.save_dir = save_dir 
        self.seg_key = seg_key 


    def on_start( self ):
        self.i = 0 
    

    def __call__( self , data_x=None , data_y=None , model_output=None ):
        
        
        if not self.seg_key is None:
            model_output = model_output[ self.seg_key  ]

        assert model_output.shape[0] == 1 , "The model output should be 1 "
        pr = model_output[0].cpu().detach().numpy().argmax(0).astype("uint8")
        
        pr_color = get_colored_segmentation_image( pr )
        cv2.imwrite( os.path.join(self.save_dir , "%d.png"%self.i )  , pr )
        cv2.imwrite( os.path.join(self.save_dir , "%d_color.png"%self.i )  , pr_color )

        self.i += 1 

        

@registry.register_function("save_preds")
class SavePredsFn(Function):
    def __init__(self):
        pass

    def execute(self , model  , eval_dataloader , output_dir     ):


        model.run_inference_callback( eval_dataloader , SaveSegmentationPredictions(save_dir=output_dir ,seg_key='top_seg'  ) )



        