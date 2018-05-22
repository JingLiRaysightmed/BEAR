
# Core is the program used easily and algorithm is advanced!

import os
# import tensorflow as tf
import argparse
from common import fileUtils # Noah comment: "ini_file_io" is from "ini_file_io.py"
from segment import MultiPlanar
# from model import fcn_2D_xy # Noah comment: "model" is from "model.py"

# set cuda visable device
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3' # Noah comment: Tensorflow's default is that using all GPUs


'''
Goal: load the config file and parameters
Step: 1. load the arguments
      2. load the config file and parameters
      3. transfer the params to the data-flow part
'''

def print_usage:

    print("[ train a classification model ]")
    print("phthon[3] MPUNetSeg2D.py --platform [tensorflow|pytorch] --phase [train|test] --file iniFile\n")

    print("[ create an example ini file ]")
    print("phthon[3] MPUNetSeg2D.py --example iniFile\n")



def main(_):

    # 1. load the arguments
    parser = argparse.ArgumentParser(description='MPUNetSeg2D parameters')

    parser.add_argument('--platform', type=str, default="tensorflow",
                        help='deelp learning platform')
    parser.add_argument('--phase', type=str, default="train",
                        help='train or test')
    parser.add_argument('--file', type=str, default="./MPUNetSeg2D.ini",
                        help='configure file')

    args = parser.parse_args()
    print("Read params          DONE")



    print '====== Phase >>> %s <<< ======' % args.phase

    if not os.path.exists(param_set['chkpoint_dir']):
        os.makedirs(param_set['chkpoint_dir'])
    if not os.path.exists(param_set['labeling_dir']):
        os.makedirs(param_set['labeling_dir'])
    
    # config file
    ini_file = args.file
    print(ini_file)
    
    param_sets = fileUtils.LoadIniFile(ini_file)
    param_set = param_sets[0]

    #platform
    if args.platform == "tensorflow":
        import tensorflow as tf
        print("platform is tensorflow")

        # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95, allow_growth=True) 

        # with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)) as sess:
            """
            Noah comment: 
            "allow_soft_placement=True":
            If you would like TensorFlow to automatically choose an existing and supported device to run the operations
            in case the specified one doesn't exist, you can set allow_soft_placement to True in the configuration option
            when creating the session.
            """
            # model = fcn_2D_xy(sess, param_set, args.phase) # Noah comment: "fcn_2D_xy()" is from "model.py"
            
            
            if args.phase == 'train':
                print("args.phase == train")
                trainer = MultiPlanarTrainer(param_set)
                trainer.train()

            elif args.phase  == 'test':
                # MultiPlanar.test()
            # elif args.phase  == 'crsv':
            #     model.test4crsv()
            # elif args.phase  == 'evaluation':
            #     model.evaluation()
            # elif args.phase  == 'fusion':
            #     model.fusion_probmap()
            #     # model.fusion_3state()

    elif args.platform == "pytorch":
        import torch
        print("platform is pytorch")



    



            

if __name__ == '__main__': # Noah comment: "if __name__ == '__main__'" make a script both importable and executable.
    # tf.app.run() # ??? why does not run the main()? A: "tf.app.run" runs the program with an optional 'main' function and 'argv' list.
    print("#=====    BEAR start    =====#")
    main("BEAR")