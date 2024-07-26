# -*- coding: UTF-8 -*-
'''
@Project : ProPos 
@File    : main.py
@Author  : Zhizhong Huang from Fudan University
@Homepage: https://hzzone.github.io/
@Email   : zzhuang19@fudan.edu.cn
@Date    : 2022/10/19 9:24 PM 
'''

import torch
import torch.distributed as dist
import numpy as np
import random
from PIL import ImageFile
import sys
import yaml
import os.path as osp
from datetime import datetime
from torchsummary import summary

ImageFile.LOAD_TRUNCATED_IMAGES = True

from models.basic_template import TrainTask
from models import model_dict

print("I have imported everything required")

if __name__ == '__main__':
    print("I am inside __main__function")
    config_path = sys.argv[1]
    print(config_path)
    with open(config_path) as f:
        print(f)
        if hasattr(yaml, 'FullLoader'):
            configs = yaml.load(f.read(), Loader=yaml.FullLoader)
            print('configs --> ', configs)
            print(configs['batch_size'])
        else:
            configs = yaml.load(f.read())
            print(" second time printing configs")
            print(configs)
    MODEL = model_dict[configs['model_name']]
    print('\n')
    print(MODEL)
    # print(MODEL.build_options())
    print('\n')
    default_parser = TrainTask.build_default_options()
    print(default_parser)
    default_opt, unknown_opt = default_parser.parse_known_args('')
    print(default_opt)
    print(unknown_opt)
    private_parser = MODEL.build_options()
    # print(private_parser.parse_known_args(''))
    opt = private_parser.parse_args(unknown_opt, namespace=default_opt)
    print('Printing OPT')
    print(opt)
    print('\n')
    print('opt.run_name-->', opt.run_name)
    if opt.run_name is None:
        print(f'opt.run_name is {opt.run_name}')
        print('osp.basename(config_path) --->', osp.basename(config_path)[:-4])
        opt.run_name = osp.basename(config_path)[:-4]
    opt.run_name = '{}-{}'.format(datetime.now().strftime("%Y_%m_%d_%H_%M_%S"), opt.run_name)
    print('opt.run_name again for logging-->', opt.run_name)
    for k in configs:
        setattr(opt, k, configs[k])

    print(opt)
    # if opt.dist:
    #     dist.init_process_group(backend='nccl', init_method='env://')
    #     torch.cuda.set_device(dist.get_rank())
    print('opt.num_devices --> ', opt.num_devices)
    if opt.num_devices > 0:
        assert opt.num_devices == torch.cuda.device_count()  # total batch size
        print('opt.num_devices --> ', opt.num_devices)
    
    print('opt.seed --> ', opt.seed)
    seed = opt.seed
    print('torch.backends.cudnn.deterministic ', torch.backends.cudnn.deterministic)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    model = MODEL(opt)
    print('model --> ', model)
    # print('model summary --> ', summary(model, (3, 32, 32), 512))
    model.fit()
