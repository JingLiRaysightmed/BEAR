#!/usr/bin/python
from __future__ import division
# from skimage.transform import resize
import configparser
# import nibabel as nib
import numpy as np


def LoadIniFile(ini_file):
    # initialize
    cf = configparser.ConfigParser() # Noah comment: ConfigParser.ConfigParser() is a function of Python.
    cf.read(ini_file)
    # dictionary list
    param_sections = []

    s = cf.sections() # Noah comment: sections() Return a list of the sections available.
    for d in range(len(s)):
        # create dictionary
        level_dict = dict(batch_size    = cf.getint(s[d], "batch_size"),
                          inputI_size_x = cf.getint(s[d], "inputI_size_x"),
                          inputI_size_y = cf.getint(s[d], "inputI_size_y"),
                          inputI_size_z = cf.getint(s[d], "inputI_size_z"),
                          inputI_chn    = cf.getint(s[d], "inputI_chn"),
                          outputI_size  = cf.getint(s[d], "outputI_size"),
                          output_chn    = cf.getint(s[d], "output_chn"),
                          rename_map    = cf.get(s[d], "rename_map"),
                          resize_r      = cf.getfloat(s[d], "resize_r"),
                          traindata_dir = cf.get(s[d], "traindata_dir"),
                          chkpoint_dir  = cf.get(s[d], "chkpoint_dir"),
                          lr            = cf.getfloat(s[d], "lr"),
                          momentum      = cf.getfloat(s[d], "momentum"),
                          epochs        = cf.getint(s[d], "epochs"),
                          model_name    = cf.get(s[d], "model_name"),
                          save_intval   = cf.getint(s[d], "save_intval"),
                          testdata_dir  = cf.get(s[d], "testdata_dir"),
                          labeling_dir  = cf.get(s[d], "labeling_dir"),
                          gt_dir        = cf.get(s[d], "gt_dir"),
                          ovlp_ita      = cf.getint(s[d], "ovlp_ita"),
                          fusion_dir    = cf.get(s[d], "fusion_dir"),
                          no_cuda       = cf.get(s[d], "no_cuda"),
                          seed=cf.get(s[d], "seed"),
                          log_interval=cf.get(s[d], "log_interval")
                          )
        # add to list
        param_sections.append(level_dict)

    return param_sections


def LoadDataPairs(pair_list, resize_r, rename_map):

    img_clec = []
    label_clec = []

    # rename_map = [0, 205, 420, 500, 550, 600, 820, 850]
    for k in range(0, len(pair_list), 2):
        img_path = pair_list[k]
        lab_path = pair_list[k+1]
        # print("img_path = %s" % img_path) #
        # print("lab_path = %s" % lab_path) #
        img_data = nib.load(img_path).get_data().copy() 
        lab_data = nib.load(lab_path).get_data().copy()
        # print("before img_data.shape") # 
        # print(img_data.shape) #
        # print("before lab_data.shape") #
        # print(lab_data.shape) #

        ###preprocessing
        # resize
        resize_dim = (np.array(img_data.shape) * resize_r).astype('int') # Noah comment: numpy.ndarray.astype: Copy of the array, cast to a specified type
        # print("resize_dim") #
        # print(resize_dim) #
        img_data = resize(img_data, resize_dim, order=1, preserve_range=True)
        lab_data = resize(lab_data, resize_dim, order=0, preserve_range=True)
        
        # print("after img_data.shape") #
        # print(img_data.shape) #
        # print("after ab_data.shape") #
        # print(lab_data.shape) #
        
        lab_r_data = np.zeros(lab_data.shape, dtype='int32')

        # rename labels 
        for i in range(len(rename_map)):
            lab_r_data[lab_data == rename_map[i]] = i # Noah comment: rename the label to [0, 7]

        # print("np.amax(lab_r_data)") #
        # print(np.amax(lab_r_data)) #

        # for s in range(img_data.shape[2]):
        #     cv2.imshow('img', np.concatenate(((img_data[:,:,s]).astype('uint8'), (lab_r_data[:,:,s]*30).astype('uint8')), axis=1))
        #     cv2.waitKey(20)
        
        # print("np.amax(img_data)") #
        # print(np.amax(img_data)) #

        img_clec.append(img_data)
        label_clec.append(lab_r_data)

    # print("len(img_clec)") #
    # print(len(img_clec)) #
    # print("len(label_clec)") #
    # print(len(label_clec)) #
    
    return img_clec, label_clec