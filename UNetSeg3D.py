# Core is the program used easily and algorithm is advanced!

import os
# import tensorflow as tf
import argparse
from common import fileUtils  # Noah comment: "ini_file_io" is from "ini_file_io.py"
from segment import UNet3DTrainer

# from model import fcn_2D_xy # Noah comment: "model" is from "model.py"

'''
Goal: load the config file and parameters
Step: 1. load the arguments
      2. load the config file and parameters
      3. transfer the params to the data-flow part
'''


def print_usage():
    print("[ train a classification model ]")
    print("phthon[3] UNetSeg3D.py --platform [tensorflow|pytorch] --phase [train|test] --file iniFile\n")

    print("[ create an example ini file ]")
    print("phthon[3] UNetSeg3D.py --example iniFile\n")


def main(_):
    # 1. load the arguments
    parser = argparse.ArgumentParser(description='MPUNetSeg2D parameters')

    parser.add_argument('--platform', type=str, default='pytorch',
                        help='deep learning platform')
    parser.add_argument('--phase', type=str, default='train',
                        help='train or test')
    parser.add_argument('--file', type=str, default='./MPUNetSeg2D.ini',
                        help='configure file')
    # parser.add_argument('--help', help='help information')

    args = parser.parse_args()
    print("Read params          DONE")

    # if args.help == True:
    #     print_usage()

    print('====== Phase >>> %s <<< ======' % args.phase)

    # config file
    ini_file = args.file
    print(ini_file)

    param_sets = fileUtils.LoadIniFile(ini_file)
    param_set = param_sets[0]

    # if not os.path.exists(param_set['chkpoint_dir']):
    #     os.makedirs(param_set['chkpoint_dir'])
    # if not os.path.exists(param_set['labeling_dir']):
    #     os.makedirs(param_set['labeling_dir'])

    # platform
    if args.platform == "tensorflow":
        import tensorflow as tf
        print("platform is tensorflow")

        # set cuda visable device
        os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'

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
            trainer = UNet3DTrainer(param_set)
            trainer.train()

        elif args.phase == 'test':
            print("test")
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
        import torch.nn as nn
        import torch.optim as optim
        import torch.nn.functional as F
        from torchvision import datasets, transforms

        print("platform is pytorch")

        if args.phase == 'train':
            print("args.phase == train")

            device = torch.device("cuda" if param_set['no_cuda'] else "cpu")

            use_cuda = not param_set['no_cuda'] and torch.cuda.is_available()

            print(device)

            kwargs = {'num_workers': 1,
                      'pin_memory': True} if use_cuda else {}  # ??? what's meaning about num_workers and pin_memory?

            print(kwargs)

            train_loader = torch.utils.data.DataLoader(
                datasets.MNIST('../data', train=True, download=True,
                               transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.1307,), (0.3081,))
                               ])),
                batch_size=param_set['batch_size'], shuffle=True, **kwargs)

            model = UNet3DTrainer.Net().to(device)
            optimizer = optim.SGD(model.parameters(), lr=param_set['lr'], momentum=param_set['momentum'])

            model.train()  # I guess it use nn.Module.train()

            # ??? what's meaning about enumerate?  Answer: feedback the index and relative
            for batch_idx, (data, target) in enumerate(train_loader):
                # content
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = F.nll_loss(output, target)  # ??? nll_loss? Answer:CrossEntropyLoss = log_softmax + nll_loss
                loss.backward()
                optimizer.step()
                if batch_idx % int(param_set['log_interval']) == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        # ???how to use format?  Answer:like print in C language.
                        param_set['epochs'], batch_idx * len(data), len(train_loader.dataset),
                                             100. * batch_idx / len(train_loader), loss.item()))

            # trainer = UNet3DTrainer(param_set)  # type: class
            #
            # for epoch in range(1, args.epochs + 1):
            #
            #     trainer.TrainNetwork()

        elif args.phase == 'test':
            print('test')


if __name__ == '__main__':  # Noah comment: "if __name__ == '__main__'" make a script both importable and executable.
    # tf.app.run() # ??? why does not run the main()?
    # A: "tf.app.run" runs the program with an optional 'main' function and 'argv' list.
    print("#=====    BEAR start    =====#")
    main("BEAR")
