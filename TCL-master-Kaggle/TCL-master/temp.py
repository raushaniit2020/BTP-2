#main.py:
# -*- coding: UTF-8 -*-

import torch
import torch.distributed as dist
import numpy as np
import random
from PIL import ImageFile
import sys
import yaml
import os.path as osp
from datetime import datetime
ImageFile.LOAD_TRUNCATED_IMAGES = True
from models.basic_template import TrainTask
from models import model_dict
if __name__ == '__main__':
    config_path = sys.argv[1]
    with open(config_path) as f:
        if hasattr(yaml, 'FullLoader'):
            configs = yaml.load(f.read(), Loader=yaml.FullLoader)
        else:
            configs = yaml.load(f.read())
    MODEL = model_dict[configs['model_name']]
    default_parser = TrainTask.build_default_options()
    default_opt, unknown_opt = default_parser.parse_known_args('')
    private_parser = MODEL.build_options()
    opt = private_parser.parse_args(unknown_opt, namespace=default_opt)
    if opt.run_name is None:
        opt.run_name = osp.basename(config_path)[:-4]
    opt.run_name = '{}-{}'.format(datetime.now().strftime("%Y_%m_%d_%H_%M_%S"), opt.run_name)
    for k in configs:
        setattr(opt, k, configs[k])
    if opt.num_devices > 0:
        assert opt.num_devices == torch.cuda.device_count()  # total batch size
    seed = opt.seed
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    model = MODEL(opt)
    model.fit()

#models/basic_template.py:
from __future__ import print_function
import os
import os.path as osp
import argparse
import warnings
import torch
import torchvision.datasets
from torchvision import transforms
import numpy as np
import torch.distributed as dist
import tqdm
from utils import TwoCropTransform, extract_features
from utils.ops import convert_to_cuda, is_root_worker, dataset_with_indices
from utils.grad_scaler import NativeScalerWithGradNormCount
from utils.loggerx import LoggerX
import torch_clustering

class TrainTask(object):
    single_view = False
    l2_normalize = True
    def __init__(self, opt):
        self.opt = opt
        self.verbose = is_root_worker()
        total_batch_size = opt.batch_size * opt.acc_grd_step
        if dist.is_initialized():
            total_batch_size *= dist.get_world_size()
        opt.learning_rate = opt.learning_rate * (total_batch_size / 256)
        if opt.resume_epoch > 0:
            opt.run_name = opt.resume_name
        self.cur_epoch = 1
        self.logger = LoggerX(save_root=osp.join('/kaggle/working/', opt.run_name),
                              enable_wandb=opt.wandb,
                              config=opt,
                              project=opt.project_name,
                              entity=opt.entity,
                              name=opt.run_name)
        self.feature_extractor = None
        self.set_loader()
        self.set_model()
        self.scaler = NativeScalerWithGradNormCount(amp=opt.amp)
        self.logger.append(self.scaler, name='scaler')

    @staticmethod
    def build_default_options():
        parser = argparse.ArgumentParser('Default arguments for training of different methods')
        parser.add_argument('--save_freq', type=int, default=50, help='save frequency')
        parser.add_argument('--test_freq', type=int, default=50, help='test frequency')
        parser.add_argument('--wandb', help='wandb', action='store_true')
        parser.add_argument('--project_name', help='wandb project_name', type=str, default='Clustering')
        parser.add_argument('--entity', help='wandb project_name', type=str, default='Hzzone')
        parser.add_argument('--run_name', type=str, help='each run name')
        parser.add_argument('--num_workers', type=int, default=16, help='num of workers to use')
        parser.add_argument('--resume_epoch', type=int, default=0, help='number of training epochs')
        parser.add_argument('--resume_name', type=str)
        parser.add_argument('--local_rank', default=0, type=int)
        parser.add_argument('--seed', default=0, type=int)
        parser.add_argument('--eval_metric', nargs='+', type=str,
                            default=['nmi', 'acc', 'ari'], help='evaluation metric NMI ACC ARI')
        # optimization
        parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
        parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
        parser.add_argument('--amp', help='amp', action='store_true')
        parser.add_argument('--encoder_name', type=str, help='the type of encoder', default='bigresnet18')
        parser.add_argument('--batch_size', type=int, default=256, help='batch_size')
        parser.add_argument('--epochs', type=int, default=1000, help='number of training epochs')
        # learning rate
        parser.add_argument('--learning_rate', type=float, default=0.05, help='base learning rate')
        parser.add_argument('--learning_eta_min', type=float, default=0.01, help='base learning rate')
        parser.add_argument('--lr_decay_gamma', type=float, default=0.1)
        parser.add_argument('--lr_decay_milestone', nargs='+', type=int, default=[60, 80])
        parser.add_argument('--step_lr', help='step_lr', action='store_true')
        parser.add_argument('--acc_grd_step', help='acc_grd_step', type=int, default=1)
        parser.add_argument('--warmup_epochs', type=int, default=0, help='warmup epochs')
        parser.add_argument('--dist', help='use  for clustering', action='store_true')
        parser.add_argument('--num_devices', type=int, default=-1, help='warmup epochs')
        # dataset
        parser.add_argument('--whole_dataset', action='store_true', help='use whole dataset')
        parser.add_argument('--pin_memory', action='store_true', help='pin_memory for dataloader')
        parser.add_argument('--dataset', type=str, default='cifar10', help='dataset')
        parser.add_argument('--data_folder', type=str, default='/kaggle/working/dataset/',
                            help='path to custom dataset')
        parser.add_argument('--label_file', type=str, default=None, help='path to label file (numpy format)')
        parser.add_argument('--img_size', type=int, default=32, help='parameter for RandomResizedCrop')
        parser.add_argument('--num_cluster', type=int, help='num_cluster')
        parser.add_argument('--test_resized_crop', action='store_true', help='imagenet test transform')
        parser.add_argument('--resized_crop_scale', type=float, help='randomresizedcrop scale', default=0.08)
        parser.add_argument('--model_name', type=str, help='the type of method', default='supcon')
        parser.add_argument('--use_gaussian_blur', help='use_gaussian_blur', action='store_true')
        parser.add_argument('--save_checkpoints', help='save_checkpoints', action='store_true')
        # SSL setting
        parser.add_argument('--feat_dim', type=int, default=2048, help='projection feat_dim')
        parser.add_argument('--data_resample', help='data_resample', action='store_true')
        parser.add_argument('--reassign', type=int, help='reassign kmeans', default=1)

        return parser

    @staticmethod
    def build_options():
        pass

    @staticmethod
    def create_dataset(data_root, dataset_name, train, transform=None, memory=False, label_file=None,
                       ):
        has_subfolder = False
        if dataset_name in ['cifar10', 'cifar20', 'cifar100']:
            dataset_type = {'cifar10': torchvision.datasets.CIFAR10,
                            'cifar20': torchvision.datasets.CIFAR100,
                            'cifar100': torchvision.datasets.CIFAR100}[dataset_name]
            has_subfolder = True
            dataset = dataset_type(data_root, train, transform, download=True)
            if dataset_name == 'cifar20':
                targets = np.array(dataset.targets)
                super_classes = [
                    [72, 4, 95, 30, 55],
                    [73, 32, 67, 91, 1],
                    [92, 70, 82, 54, 62],
                    [16, 61, 9, 10, 28],
                    [51, 0, 53, 57, 83],
                    [40, 39, 22, 87, 86],
                    [20, 25, 94, 84, 5],
                    [14, 24, 6, 7, 18],
                    [43, 97, 42, 3, 88],
                    [37, 17, 76, 12, 68],
                    [49, 33, 71, 23, 60],
                    [15, 21, 19, 31, 38],
                    [75, 63, 66, 64, 34],
                    [77, 26, 45, 99, 79],
                    [11, 2, 35, 46, 98],
                    [29, 93, 27, 78, 44],
                    [65, 50, 74, 36, 80],
                    [56, 52, 47, 59, 96],
                    [8, 58, 90, 13, 48],
                    [81, 69, 41, 89, 85],
                ]
                import copy
                copy_targets = copy.deepcopy(targets)
                for i in range(len(super_classes)):
                    for j in super_classes[i]:
                        targets[copy_targets == j] = i
                dataset.targets = targets.tolist()
        else:
            data_path = osp.join(data_root, dataset_name)
            dataset_type = torchvision.datasets.ImageFolder
            if 'train' in os.listdir(data_path):
                has_subfolder = True
                dataset = dataset_type(
                    osp.join(data_root, dataset_name, 'train' if train else 'val'), transform=transform)
            else:
                dataset = dataset_type(osp.join(data_root, dataset_name), transform=transform)
        if label_file is not None:
            new_labels = np.load(label_file).flatten()
            assert len(dataset.targets) == len(new_labels)
            noise_ratio = (1 - np.mean(np.array(dataset.targets) == new_labels))
            dataset.targets = new_labels.tolist()
            print(f'load label file from {label_file}, possible noise ratio {noise_ratio}')
        return dataset, has_subfolder

    def build_dataloader(self,
                         dataset_name,
                         transform,
                         batch_size,
                         drop_last=False,
                         shuffle=False,
                         sampler=False,
                         train=False,
                         memory=False,
                         data_resample=False,
                         label_file=None):

        opt = self.opt
        data_root = opt.data_folder
        dataset, has_subfolder = self.create_dataset(data_root, dataset_name,
                                                     train, transform=transform,
                                                     memory=memory,
                                                     label_file=label_file)
        labels = dataset.targets
        labels = np.array(labels)
        if opt.whole_dataset and has_subfolder:
            ano_dataset = self.create_dataset(data_root, dataset_name, not train, transform=transform,
                                              memory=memory)[0]
            labels = np.concatenate([labels, ano_dataset.targets], axis=0)
            dataset = torch.utils.data.ConcatDataset([dataset, ano_dataset])
        with_indices = train and (not memory)
        if with_indices:
            dataset = dataset_with_indices(dataset)
        if sampler:
            if dist.is_initialized():
                from utils.sampler import RandomSampler
                if shuffle and data_resample:
                    num_iter = len(dataset) // (batch_size * dist.get_world_size())
                    sampler = RandomSampler(dataset=dataset, batch_size=batch_size, num_iter=num_iter, restore_iter=0,
                                            weights=None, replacement=True, seed=0, shuffle=True)
                else:
                    sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=shuffle)
            else:
                # memory loader
                sampler = None
        else:
            sampler = None
        prefetch_factor = 2
        persistent_workers = True
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(False if sampler is not None else shuffle),
            num_workers=opt.num_workers,
            pin_memory=opt.pin_memory,
            sampler=sampler,
            drop_last=drop_last,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor
        )
        return dataloader, labels, sampler

    def train_transform(self, normalize):
        opt = self.opt
        train_transform = [
            transforms.RandomResizedCrop(size=opt.img_size, scale=(opt.resized_crop_scale, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
        ]
        if opt.use_gaussian_blur:
            train_transform.append(
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))], 0.5)
            )
        train_transform += [transforms.ToTensor(), normalize]
        train_transform = transforms.Compose(train_transform)
        if not self.single_view:
            train_transform = TwoCropTransform(train_transform)
        return train_transform

    def test_transform(self, normalize):
        opt = self.opt
        def resize(image):
            size = (opt.img_size, opt.img_size)
            if image.size == size:
                return image
            return image.resize(size)
        test_transform = []
        if opt.test_resized_crop:
            test_transform += [transforms.Resize(256), transforms.CenterCrop(224)]
        test_transform += [
            resize,
            transforms.ToTensor(),
            normalize
        ]
        test_transform = transforms.Compose(test_transform)
        return test_transform

    @staticmethod
    def normalize(dataset_name):
        normalize_params = {
            'mnist': [(0.1307,), (0.3081,)],
            'cifar10': [(0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)],
            'cifar20': [(0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)],
            'cifar100': [(0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)],
            'imagenet': [(0.485, 0.456, 0.406), (0.229, 0.224, 0.225)],
            'stl10': [(0.485, 0.456, 0.406), (0.229, 0.224, 0.225)],
        }
        if dataset_name not in normalize_params.keys():
            mean, std = normalize_params['imagenet']
            print(f'Dataset {dataset_name} does not exist in normalize_params,'
                  f' use default normalizations: mean {str(mean)}, std {str(std)}.')
        else:
            mean, std = normalize_params[dataset_name]
        normalize = transforms.Normalize(mean=mean, std=std, inplace=True)
        return normalize

    def set_loader(self):
        opt = self.opt
        normalize = self.normalize(opt.dataset)
        train_transform = self.train_transform(normalize)
        self.logger.msg_str(f'set train transform... \n {str(train_transform)}')
        train_loader, labels, sampler = self.build_dataloader(
            dataset_name=opt.dataset,
            transform=train_transform,
            batch_size=opt.batch_size,
            shuffle=True,
            drop_last=True,
            sampler=True,
            train=True,
            data_resample=opt.data_resample,
            label_file=opt.label_file)
        if sampler is None:
            sampler = train_loader.sampler
        self.logger.msg_str(f'set train dataloader with {len(train_loader)} iterations...')
        test_transform = self.test_transform(normalize)
        self.logger.msg_str(f'set test transform... \n {str(test_transform)}')
        if 'imagenet' in opt.dataset:
            if not opt.test_resized_crop:
                warnings.warn('ImageNet should center crop during testing...')
        test_loader = self.build_dataloader(opt.dataset,
                                            test_transform,
                                            train=False,
                                            sampler=True,
                                            batch_size=opt.batch_size)[0]
        self.logger.msg_str(f'set test dataloader with {len(test_loader)} iterations...')
        memory_loader = self.build_dataloader(opt.dataset,
                                              test_transform,
                                              train=True,
                                              batch_size=opt.batch_size,
                                              sampler=True,
                                              memory=True,
                                              label_file=opt.label_file)[0]
        self.logger.msg_str(f'set memory dataloader with {len(memory_loader)} iterations...')
        self.test_loader = test_loader
        self.memory_loader = memory_loader
        self.train_loader = train_loader
        self.sampler = sampler
        self.iter_per_epoch = len(train_loader)
        self.num_classes = len(np.unique(labels[labels >= 0]))
        self.num_samples = len(labels)
        self.gt_labels = torch.from_numpy(labels).cuda()
        self.num_cluster = self.num_classes if opt.num_cluster is None else opt.num_cluster
        opt.num_cluster = self.num_cluster
        self.psedo_labels = torch.zeros((self.num_samples,)).long().cuda()
        self.logger.msg_str('load {} images...'.format(self.num_samples))

    def fit(self):
        opt = self.opt
        if opt.resume_epoch > 0:
            self.logger.load_checkpoints(opt.resume_epoch)

        n_iter = self.iter_per_epoch * opt.resume_epoch + 1
        self.cur_epoch = int(opt.resume_epoch + 1)
        # training routine
        self.progress_bar = tqdm.tqdm(total=self.iter_per_epoch * opt.epochs, disable=not self.verbose, initial=n_iter)

        # if n_iter == 1:
        #     self.psedo_labeling(n_iter)
        #     self.test(n_iter)
        self.psedo_labeling(n_iter)
        print("Before training starts")
        while True:
            # self.sampler.set_epoch(self.cur_epoch)
            print("inside while loop")
            for inputs in self.train_loader:
                inputs, indices = convert_to_cuda(inputs)
                self.adjust_learning_rate(n_iter)
                self.train(inputs, indices, n_iter)
                self.progress_bar.refresh()
                self.progress_bar.update()
                n_iter += 1

            cur_epoch = self.cur_epoch
            self.logger.msg([cur_epoch, ], n_iter)

            apply_kmeans = (self.cur_epoch % opt.reassign) == 0
            if (self.cur_epoch % opt.test_freq == 0) or (self.cur_epoch % opt.save_freq == 0) or apply_kmeans:
                if opt.save_checkpoints:
                    self.logger.checkpoints(int(self.cur_epoch))

            if apply_kmeans:
                self.psedo_labeling(n_iter)

            if (self.cur_epoch % opt.test_freq == 0) or apply_kmeans:
                self.test(n_iter)
                torch.cuda.empty_cache()

            self.cur_epoch += 1

            if self.cur_epoch > opt.epochs:
                break

    def set_model(opt):
        pass

    @torch.no_grad()
    def test(self, n_iter):
        opt = self.opt
        if opt.whole_dataset:
            return

        test_features, test_labels = extract_features(self.feature_extractor, self.test_loader)
        if hasattr(self, 'mem_data') and self.mem_data['epoch'] == self.cur_epoch:
            mem_features, mem_labels = self.mem_data['features'], self.mem_data['labels']
        else:
            mem_features, mem_labels = extract_features(self.feature_extractor, self.memory_loader)
            if self.l2_normalize:
                mem_features.div_(torch.linalg.norm(mem_features, dim=1, ord=2, keepdim=True))
        if self.l2_normalize:
            test_features.div_(torch.linalg.norm(test_features, dim=1, ord=2, keepdim=True))

        from utils.knn_monitor import knn_monitor
        knn_acc = knn_monitor(
            mem_features,
            mem_labels,
            test_features,
            test_labels,
            knn_k=20,
            knn_t=0.07)
        self.logger.msg([knn_acc, ], n_iter)

    def train(self, inputs, indices, n_iter):
        pass

    def cosine_annealing_LR(self, n_iter):
        opt = self.opt
        epoch = n_iter / self.iter_per_epoch
        max_lr = opt.learning_rate
        min_lr = max_lr * opt.learning_eta_min
        # warmup
        if epoch < opt.warmup_epochs:
            # lr = (max_lr - min_lr) * epoch / opt.warmup_epochs + min_lr
            lr = opt.learning_rate * epoch / opt.warmup_epochs
        else:
            lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + np.cos((epoch - opt.warmup_epochs) * np.pi / opt.epochs))
        return lr

    def step_LR(self, n_iter):
        opt = self.opt
        lr = opt.learning_rate
        epoch = n_iter / self.iter_per_epoch
        if epoch < opt.warmup_epochs:
            # lr = (max_lr - min_lr) * epoch / opt.warmup_epochs + min_lr
            lr = opt.learning_rate * epoch / opt.warmup_epochs
        else:
            for milestone in opt.lr_decay_milestone:
                lr *= opt.lr_decay_gamma if epoch >= milestone else 1.
        return lr

    def adjust_learning_rate(self, n_iter):
        opt = self.opt
        if opt.step_lr:
            lr = self.step_LR(n_iter)
        else:
            lr = self.cosine_annealing_LR(n_iter)
        for i in range(len(self.optimizer.param_groups)):
            self.optimizer.param_groups[i]['lr'] = lr
        self.logger.msg([lr, ], n_iter)

    def clustering(self, features, n_clusters):
        opt = self.opt
        kwargs = {
            'metric': 'cosine' if self.l2_normalize else 'euclidean',
            'distributed': True,
            'random_state': 0,
            'n_clusters': n_clusters,
            'verbose': True
        }
        clustering_model = torch_clustering.PyTorchKMeans(init='k-means++', max_iter=300, tol=1e-4, **kwargs)
        psedo_labels = clustering_model.fit_predict(features)
        cluster_centers = clustering_model.cluster_centers_
        return psedo_labels, cluster_centers

    @torch.no_grad()
    def psedo_labeling(self, n_iter):
        opt = self.opt
        torch.cuda.empty_cache()
        self.logger.msg_str('Generating the psedo-labels')
        mem_features, mem_labels = extract_features(self.feature_extractor, self.memory_loader)
        if self.l2_normalize:
            mem_features.div_(torch.linalg.norm(mem_features, dim=1, ord=2, keepdim=True))
        psedo_labels, cluster_centers = self.clustering(mem_features, self.num_cluster)
        dist.barrier()
        global_std = torch.std(mem_features, dim=0).mean()
        self.logger.msg_str(torch.unique(psedo_labels.cpu(), return_counts=True))
        self.logger.msg_str(torch.unique(mem_labels.long().cpu(), return_counts=True))
        results = torch_clustering.evaluate_clustering(mem_labels.cpu().numpy(),
                                                       psedo_labels.cpu().numpy(),
                                                       eval_metric=opt.eval_metric,
                                                       phase='ema_train')
        results['global_std'] = global_std
        self.logger.msg(results, n_iter)
        dist.broadcast(psedo_labels, src=0)
        dist.broadcast(cluster_centers, src=0)
        self.psedo_labels.copy_(psedo_labels)
        self.cluster_centers = cluster_centers
        self.mem_data = {
            'features': mem_features,
            'labels': mem_labels,
            'epoch': self.cur_epoch
        }
        if opt.data_resample:
            counts = torch.unique(psedo_labels.cpu(), return_counts=True)[1]
            weights = torch.zeros(psedo_labels.size()).float()
            for l in range(counts.size(0)):
                weights[psedo_labels == l] = psedo_labels.size(0) / counts[l]
            self.sampler.set_weights(weights)
            self.logger.msg_str(f'set the weights of train dataloader as {weights.cpu().numpy()}')
        torch.cuda.empty_cache()

    def collect_params(self, *models, exclude_bias_and_bn=True):
        param_list = []
        for model in models:
            for name, param in model.named_parameters():
                param_dict = {
                    'name': name,
                    'params': param,
                }
                if exclude_bias_and_bn and any(s in name for s in ['bn', 'bias']):
                    param_dict.update({'weight_decay': 0., 'lars_exclude': True})
                param_list.append(param_dict)
        return param_list

#models/__init__.py:
import importlib
import os
import os.path as osp
from utils.model_register import import_models, Register
model_dict = Register('model_dict')
import_models(osp.dirname(__file__), 'models')
import_models(osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))), 'experiment'), 'experiment')

#utils/model_register.py:
# -*- coding: UTF-8 -*-

import logging
import importlib
import os
import pathlib
import re


class Register:
    def __init__(self, registry_name, baseclass=None):
        self._dict = {}
        self._name = registry_name
        self._baseclass = baseclass

    def __setitem__(self, key, value):
        if not callable(value):
            raise Exception(f"Value of a Registry must be a callable!\nValue: {value}")
        if key is None:
            key = value.__name__
        if key in self._dict:
            logging.warning("Key %s already in registry %s." % (key, self._name))
        self._dict[key] = value

    def register(self, target):
        """Decorator to register a function or class."""
        def add(key, value):
            self[key] = value
            return value
        if target in self._dict:
            logging.warning(f'Cannot register duplicate ({target})')
        if callable(target):
            # @reg.register
            return add(None, target)

        # @reg.register('alias')
        def class_rebuilder(cls):
            if self._baseclass is not None:
                for p in dir(self._baseclass):
                    if p in dir(cls):
                        continue
                    setattr(cls, p, getattr(self._baseclass, p))
            return add(target, cls)

        return class_rebuilder

    def __getitem__(self, key):
        return self._dict[key]

    def __contains__(self, key):
        return key in self._dict

    def keys(self):
        """key"""
        return self._dict.keys()

    def __repr__(self):
        return str(self._dict)

def import_models(root, prefix):
    root = os.path.abspath(root)
    for p in pathlib.Path(root).rglob('*.py'):
        p = str(p)
        flag = False
        for x in p.split(os.sep):
            if x.startswith('.'):
                flag = True
        if flag:
            continue
        lib = re.sub(root, prefix, p)
        lib = re.sub(os.sep, '.', lib)[:-3]
        importlib.import_module(lib)

#network/resnet.py:

import torch.nn as nn
from torchvision.models import resnet
from torchvision.models.resnet import conv3x3, conv1x1, BasicBlock, Bottleneck
from torch.hub import load_state_dict_from_url
from typing import Type, Any, Callable, Union, List, Optional
from torch import Tensor
settings = {
    'resnet18': [[2, 2, 2, 2], BasicBlock],
    'resnet34': [[3, 4, 6, 3], BasicBlock],
    'resnet50': [[3, 4, 6, 3], Bottleneck],
}

model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-f37072fd.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-b627a593.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-0676ba61.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-63fe2227.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-394f9c45.pth",
    "resnext50_32x4d": "https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth",
    "resnext101_32x8d": "https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth",
    "wide_resnet50_2": "https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth",
    "wide_resnet101_2": "https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth",
}

def _resnet(
        arch: str,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        pretrained: bool,
        progress: bool,
        **kwargs: Any,
) -> resnet.ResNet:
    model = resnet.ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model

class ResNet(object):
    def __init__(self,
                 net_name,
                 cifar=False,
                 preact=False):
        self.net_name = net_name
        self.cifar = cifar
        self.preact = preact

    def __call__(self, pretrained: bool = False, progress: bool = True, **kwargs):
        layers, block = settings[self.net_name]
        kwargs.update({
            'arch': self.net_name,
            'layers': layers,
            'block': block,
        })
        if self.preact:
            kwargs['block'] = PreActBasicBlock
        model = _resnet(pretrained=pretrained, progress=progress, **kwargs)
        nets = []
        for name, module in model.named_children():
            if self.cifar:
                if name == 'conv1':
                    module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
                if isinstance(module, nn.MaxPool2d):
                    continue
            if isinstance(module, nn.Linear):
                nets.append(nn.Flatten(1))
                continue
            nets.append(module)
        model = nn.Sequential(*nets)
        return model

class PreActBasicBlock(BasicBlock):
    expansion = 1
    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super(PreActBasicBlock, self).__init__(inplanes, planes, stride, downsample, groups, base_width, dilation,
                                               norm_layer)
        self.bn1 = norm_layer(inplanes)
        if self.downsample is not None:
            self.downsample = self.downsample[0]  # remove norm

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.bn1(x)
        out = self.relu(out)
        if self.downsample is not None:
            identity = self.downsample(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out += identity
        return out

if __name__ == '__main__':
    model = ResNet('resnet18',
                   cifar=True,
                   preact=True)
    model = model()
    print(model)
    import torch

    inputs = torch.randn(2, 3, 32, 32)
    print(model(inputs))

#network/preact_resnet.py:
# -*- coding: UTF-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x), inplace=True)
        shortcut = self.shortcut(out)
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out), inplace=True))
        out += shortcut
        return out

class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4
    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x), inplace=True)
        shortcut = self.shortcut(out)
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out), inplace=True))
        out = self.conv3(F.relu(self.bn3(out), inplace=True))
        out += shortcut
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = conv3x3(3, 64)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.bn = nn.BatchNorm2d(512)
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    def forward(self, x):
        out = x
        out = self.conv1(out)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.relu(self.bn(out), inplace=True)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return out

def ResNet18():
    return ResNet(PreActBlock, [2, 2, 2, 2])
if __name__ == '__main__':
    net = ResNet18()
    y = net(torch.randn(2, 3, 32, 32))
    print(y.size())


#network/__init__.py:
# -*- coding: UTF-8 -*-

from . import resnet, preact_resnet

backbone_dict = {
    'bigresnet18': [resnet.ResNet('resnet18', cifar=True), 512],
    'bigresnet34': [resnet.ResNet('resnet18', cifar=True), 512],
    'bigresnet50': [resnet.ResNet('resnet18', cifar=True), 2048],
    'bigresnet18_preact': [preact_resnet.ResNet18, 512],
    'resnet18': [resnet.ResNet('resnet18'), 512],
    'resnet34': [resnet.ResNet('resnet34'), 512],
    'resnet50': [resnet.ResNet('resnet50'), 2048],
}

#utils/ops.py:

import math
from torch import nn
import torch
from typing import Union
import torch.distributed as dist
import collections.abc as string_classes
import torch.nn.functional as F
import collections.abc as container_abcs
from PIL import Image

@torch.no_grad()
def topk_accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append((correct_k / batch_size).item())
    return res

def concat_all_gather(tensor):
    dtype = tensor.dtype
    tensor = tensor.float()
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    output = torch.cat(tensors_gather, dim=0)
    output = output.to(dtype)
    return output

class dataset_with_indices(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        outs = self.dataset[idx]
        return [outs, idx]

def convert_to_cuda(data):
    r"""Converts each NumPy array data field into a tensor"""
    if isinstance(data, torch.Tensor):
        return data.cuda(non_blocking=True)  # Ensure tensors are on the GPU
    elif isinstance(data, container_abcs.Mapping):
        return {key: convert_to_cuda(data[key]) for key in data}
    elif isinstance(data, tuple) and hasattr(data, '_fields'):  # namedtuple
        return data._replace(**{key: convert_to_cuda(data[key]) for key in data._fields})
    elif isinstance(data, container_abcs.Sequence) and not isinstance(data, str):
        return [convert_to_cuda(d) for d in data]
    else:
        return data

def is_root_worker():
    verbose = True
    if dist.is_initialized():
        if dist.get_rank() != 0:
            verbose = False
    return verbose

def load_network(state_dict):
    if isinstance(state_dict, str):
        state_dict = torch.load(state_dict, map_location='cpu')
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        namekey = k.replace('module.', '')  # remove `module.`
        new_state_dict[namekey] = v
    return new_state_dict

def convert_to_ddp(modules: Union[list, nn.Module], **kwargs):
    if isinstance(modules, list):
        modules = [x.cuda() for x in modules]
    else:
        modules = modules.cuda()
    if dist.is_initialized():
        device = torch.cuda.current_device()
        if isinstance(modules, list):
            modules = [torch.nn.parallel.DistributedDataParallel(x,
                                                                 device_ids=[device, ],
                                                                 output_device=device,
                                                                 **kwargs) for
                       x in modules]
        else:
            modules = torch.nn.parallel.DistributedDataParallel(modules,
                                                                device_ids=[device, ],
                                                                output_device=device,
                                                                **kwargs)

    else:
        modules = torch.nn.DataParallel(modules)
    return modules

#models/tcl/tcl.py:
import matplotlib.pyplot as plt
import torch
import argparse
import copy
import torch.nn.functional as F
import tqdm
import numpy as np
import torch.nn as nn
import torch.distributed as dist
from models.basic_template import TrainTask
from network import backbone_dict
from .tcl_wrapper import SimCLRWrapper
from utils.ops import convert_to_ddp, convert_to_cuda
from models import model_dict

@model_dict.register('tcl')
class TCL(TrainTask):
    def set_model(self):
        opt = self.opt
        encoder_type, dim_in = backbone_dict[opt.encoder_name]
        kwargs = {'encoder_type': encoder_type, 'in_dim': dim_in, 'fea_dim': opt.feat_dim, 'T': opt.temp,
                  'num_cluster': self.num_cluster, 'mixup_alpha': opt.mixup_alpha, 'num_samples': self.num_samples,
                  'scale1': opt.scale1, 'scale2': opt.scale2}
        if opt.arch == 'simclr':
            tcl = SimCLRWrapper(**kwargs).cuda()
        else:
            raise NotImplemented
        tcl.register_buffer('pseudo_labels', self.gt_labels.cpu())
        if opt.syncbn:
            tcl = torch.nn.SyncBatchNorm.convert_sync_batchnorm(tcl)
        params = list(tcl.parameters())
        optimizer = torch.optim.SGD(params=params, lr=opt.learning_rate, momentum=opt.momentum,
                                    weight_decay=opt.weight_decay)
        # tcl = convert_to_ddp(tcl)
        self.logger.modules = [tcl, optimizer]
        self.tcl = tcl
        self.optimizer = optimizer
        self.tcl.register_buffer('pseudo_labels', self.gt_labels.cpu().cuda())  # Ensure CUDA placement

    @staticmethod
    def build_options():
        parser = argparse.ArgumentParser('Private arguments for training of different methods')
        parser.add_argument('--sep_gmm', action='store_true')
        parser.add_argument('--temp', type=float, help='temp for contrastive loss')
        parser.add_argument('--scale1', type=float)
        parser.add_argument('--scale2', type=float)
        parser.add_argument('--cls_loss_weight', type=float, default=1.0, help='cls_loss_weight')
        parser.add_argument('--align_loss_weight', type=float, default=1.0, help='align_loss_weight')
        parser.add_argument('--ent_loss_weight', type=float, default=1.0, help='ent_loss_weight')
        parser.add_argument('--ne_loss_weight', type=float, default=1.0, help='ne_loss_weight')
        parser.add_argument('--mixup_alpha', type=float, default=1.0, help='cls_loss_weight')
        parser.add_argument('--arch', type=str, default='simclr', help='simclr')
        return parser

    def train(self, inputs, indices, n_iter):
        opt = self.opt
        is_warmup = not (self.cur_epoch >= opt.warmup_epochs)
        self.tcl.warmup = is_warmup
        images, _ = inputs
        self.tcl.train()
        im_w, im_q, im_k = images
        indices = indices.cpu()
        # compute loss
        contrastive_loss, cls_loss1, cls_loss2, ent_loss, ne_loss, align_loss = self.tcl(im_w, im_q, im_k, indices)
        # SGD
        self.optimizer.zero_grad()
        loss = contrastive_loss + \
               opt.cls_loss_weight * (cls_loss1 + cls_loss2) + \
               opt.ent_loss_weight * ent_loss + \
               opt.ne_loss_weight * ne_loss + \
               opt.align_loss_weight * align_loss
        loss.backward()
        self.optimizer.step()
        # self.logger.msg([contrastive_loss, cls_loss1, cls_loss2, ent_loss, ne_loss, align_loss], n_iter)

    def extract_features(self, model, loader):
        opt = self.opt
        features = torch.zeros(len(loader.dataset), opt.feat_dim).cuda()
        all_labels = torch.zeros(len(loader.dataset)).cuda()
        cluster_labels = torch.zeros(len(loader.dataset), self.num_cluster).cuda()
        model.eval()
        encoder = model.encoder_q.cuda()
        classifier = model.classifier_q.cuda()
        projector = model.projector_q.cuda()
        local_features = []
        local_labels = []
        local_cluster_labels = []
        for inputs in tqdm.tqdm(loader, disable=not self.verbose):
            images, labels = convert_to_cuda(inputs)
            local_labels.append(labels)
            x = encoder(images)
            local_cluster_labels.append(F.softmax(classifier(x), dim=1))
            local_features.append(F.normalize(projector(x), dim=1))
        local_features = torch.cat(local_features, dim=0)
        local_labels = torch.cat(local_labels, dim=0)
        local_cluster_labels = torch.cat(local_cluster_labels, dim=0)
        indices = torch.Tensor(list(iter(loader.sampler))).long().cuda()
        features.index_add_(0, indices, local_features)
        all_labels.index_add_(0, indices, local_labels.float())
        cluster_labels.index_add_(0, indices, local_cluster_labels.float())

        if dist.is_initialized():
            dist.all_reduce(features, op=dist.ReduceOp.SUM)
            dist.all_reduce(all_labels, op=dist.ReduceOp.SUM)
            dist.all_reduce(cluster_labels, op=dist.ReduceOp.SUM)
            mask = torch.norm(features, dim=1) > 1.5
            all_labels[mask] = all_labels[mask] / dist.get_world_size()
            cluster_labels[mask] = cluster_labels[mask] / dist.get_world_size()
            features = F.normalize(features, dim=1)
        labels = all_labels.long()
        return features, cluster_labels, labels

    def hist(self, assignments, is_clean, labels, n_iter, sample_type='context_assignments_hist'):
        fig, ax = plt.subplots()
        ax.hist(assignments[is_clean, labels[is_clean]].cpu().numpy(), label='clean', bins=100, alpha=0.5)
        ax.hist(assignments[~is_clean, labels[~is_clean]].cpu().numpy(), label='noisy', bins=100, alpha=0.5)
        ax.legend()
        import io
        from PIL import Image
        buf = io.BytesIO()
        fig.savefig(buf)
        buf.seek(0)
        img = Image.open(buf)
        self.logger.save_image(img, n_iter, sample_type=sample_type)
        plt.close()

    @torch.no_grad()
    def psedo_labeling(self, n_iter):
        opt = self.opt
        assert not opt.whole_dataset
        self.logger.msg_str('Generating the psedo-labels')
        labels = self.gt_labels
        confidence, context_assignments, features, cluster_labels, = self.correct_labels(self.tcl, labels)
        self.tcl.confidences.copy_(confidence.float())
        self.evaluate(self.tcl, features, confidence, cluster_labels, labels, context_assignments, n_iter)

    def evaluate(self, model, features, confidence, cluster_labels, labels, context_assignments, n_iter):
        opt = self.opt
        clean_labels = torch.Tensor(
            self.create_dataset(opt.data_folder, opt.dataset, train=True, transform=None)[0].targets).cuda().long()
        is_clean = clean_labels.cpu().numpy() == labels.cpu().numpy()
        self.hist(context_assignments, is_clean, labels, n_iter)
        train_acc = (torch.argmax(cluster_labels, dim=1) == clean_labels).float().mean()
        test_features, test_cluster_labels, test_labels = self.extract_features(model, self.test_loader)
        test_acc = (test_labels == torch.argmax(test_cluster_labels, dim=1)).float().mean()
        from utils.knn_monitor import knn_predict
        knn_labels = knn_predict(test_features, features, clean_labels,
                                 classes=self.num_cluster, knn_k=200, knn_t=0.1)[:, 0]
        self.logger.msg_str(torch.unique(torch.argmax(test_cluster_labels, dim=1), return_counts=True))
        knn_acc = (test_labels == knn_labels).float().mean()
        estimated_noise_ratio = (confidence > 0.5).float().mean().item()
        if opt.scale1 is None:
            self.tcl.scale1 = estimated_noise_ratio
        if opt.scale2 is None:
            self.tcl.scale2 = estimated_noise_ratio
        noise_accuracy = ((confidence > 0.5) == (clean_labels == labels)).float().mean()
        from sklearn.metrics import roc_auc_score
        context_noise_auc = roc_auc_score(is_clean, confidence.cpu().numpy())
        self.logger.msg([estimated_noise_ratio, noise_accuracy,
                         context_noise_auc, train_acc, test_acc, knn_acc], n_iter)

    def correct_labels(self, model, labels):
        opt = self.opt
        features, cluster_labels, _ = self.extract_features(model, self.memory_loader)
        confidence, context_assignments, centers = self.noise_detect(cluster_labels, labels, features)
        model.prototypes.copy_(centers)
        model.context_assignments.copy_(context_assignments.float())
        return confidence, context_assignments, features, cluster_labels

    def noise_detect(self, cluster_labels, labels, features):
        opt = self.opt
        centers = F.normalize(cluster_labels.T.mm(features), dim=1)
        context_assignments_logits = features.mm(centers.T) / opt.temp
        context_assignments = F.softmax(context_assignments_logits, dim=1)
        losses = - context_assignments[torch.arange(labels.size(0)), labels]
        losses = losses.cpu().numpy()[:, np.newaxis]
        losses = (losses - losses.min()) / (losses.max() - losses.min())
        labels = labels.cpu().numpy()

        from sklearn.mixture import GaussianMixture
        confidence = np.zeros((losses.shape[0],))
        if opt.sep_gmm:
            for i in range(self.num_cluster):
                mask = labels == i
                c = losses[mask, :]
                gm = GaussianMixture(n_components=2, random_state=0).fit(c)
                pdf = gm.predict_proba(c)
                confidence[mask] = (pdf / pdf.sum(1)[:, np.newaxis])[:, np.argmin(gm.means_)]
        else:
            gm = GaussianMixture(n_components=2, random_state=0).fit(losses)
            pdf = gm.predict_proba(losses)
            confidence = (pdf / pdf.sum(1)[:, np.newaxis])[:, np.argmin(gm.means_)]
        confidence = torch.from_numpy(confidence).float().cuda()
        return confidence, context_assignments, centers

    def test(self, n_iter):
        pass

    def train_transform(self, normalize):
        import torchvision.transforms as transforms
        from utils import TwoCropTransform
        opt = self.opt
        train_transform = [
            transforms.RandomResizedCrop(size=opt.img_size, scale=(opt.resized_crop_scale, 1)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomSolarize(threshold=128, p=0.5),
            transforms.ToTensor(),
            normalize
        ]

        weak_transform = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(opt.img_size,
                                  padding=int(opt.img_size * 0.125)),
            transforms.ToTensor(),
            normalize
        ]

        train_transform = transforms.Compose(train_transform)
        weak_transform = transforms.Compose(weak_transform)

        def ThreeCropTransform(img):
            return [weak_transform(img), ] + TwoCropTransform(train_transform)(img)

        return ThreeCropTransform

#models/tcl/tcl_wrapper.py:
import torch
import torch.nn as nn
import torch.nn.functional as F

def mixup(input, alpha=1.0):
    bs = input.size(0)
    randind = torch.randperm(bs).to(input.device)
    import numpy as np
    lam = np.random.beta(alpha, alpha)
    lam = torch.ones_like(randind).float() * lam
    lam = torch.max(lam, 1. - lam)
    lam_expanded = lam.view([-1] + [1] * (input.dim() - 1))
    input = lam_expanded * input + (1. - lam_expanded) * input[randind]
    return input, randind, lam.unsqueeze(1)


class Wrapper(nn.Module):

    @staticmethod
    def create_projector(in_dim, out_dim):
        return nn.Sequential(nn.Linear(in_dim, in_dim),
                             nn.BatchNorm1d(in_dim),
                             nn.ReLU(inplace=True),
                             nn.Linear(in_dim, out_dim),
                             nn.BatchNorm1d(out_dim),
                             )

    @staticmethod
    def create_classifier(in_dim, out_dim, dropout=0.25):
        return nn.Sequential(nn.Linear(in_dim, in_dim),
                             nn.BatchNorm1d(in_dim),
                             nn.ReLU(inplace=True),
                             nn.Dropout(p=dropout),
                             nn.Linear(in_dim, out_dim),
                             nn.BatchNorm1d(out_dim)
                             )

    def __init__(self,
                 encoder_type,
                 in_dim,
                 num_cluster,
                 fea_dim=128,
                 T=0.07,
                 mixup_alpha=1.0,
                 num_samples=0,
                 scale1=0.,
                 scale2=1.
                 ):
        super(Wrapper, self).__init__()

        self.T = T
        self.num_cluster = num_cluster
        self.warmup = False
        self.in_dim = in_dim
        self.fea_dim = fea_dim
        self.mixup_alpha = mixup_alpha
        self.scale1 = scale1
        self.scale2 = scale2
        self.num_samples = num_samples

        self.encoder_q = encoder_type()
        self.projector_q = self.create_projector(self.in_dim, self.fea_dim)
        self.classifier_q = self.create_classifier(self.in_dim, self.num_cluster)

        self.register_buffer('prototypes', torch.randn(self.num_cluster, fea_dim))
        self.prototypes = F.normalize(self.prototypes, dim=1)
        self.register_buffer('confidences', torch.zeros(self.num_samples))
        self.register_buffer('context_assignments', torch.zeros(self.num_samples, self.num_cluster))

    def inference(self, im_w, im_q, im_k):
        pass

    def forward_contrastive_loss(self, q, k, indices):
        pass

    def forward_reg_loss(self, pred_logits):
        pred_softmax = F.softmax(pred_logits, dim=1)
        ent_loss = - (pred_softmax * F.log_softmax(pred_logits, dim=1)).sum(dim=1).mean()
        prob_mean = pred_softmax.mean(dim=0)
        ne_loss = (prob_mean * prob_mean.log()).sum()
        return ent_loss, ne_loss

    def forward_loss(self, im_w, im_q, im_k, indices):
        pass

    def forward_cls_loss(self,
                         q_w, w_logits,
                         q_logits1, q_logits2, mix_logits,
                         q_mix, mix_randind, mix_lam, indices):

        with torch.no_grad():
            labels = self.pseudo_labels[indices]
            confidences = self.confidences[indices].unsqueeze(1)

            targets_onehot_noise = F.one_hot(labels, self.num_cluster).float().cuda()
            w_prob = F.softmax(w_logits.detach(), dim=1)
            q_prob1 = F.softmax(q_logits1.detach(), dim=1)
            q_prob2 = F.softmax(q_logits2.detach(), dim=1)

            # targets_mix_corrected = (w_prob + q_prob1 + q_prob2) / 3.

            def comb(p1, p2, lam):
                return (1 - lam) * p1 + lam * p2

            targets_corrected1 = comb(q_prob2, targets_onehot_noise, confidences * self.scale1)
            targets_corrected2 = comb(q_prob1, targets_onehot_noise, confidences * self.scale1)
            targets_mix_corrected = comb((q_prob1 + q_prob2) * 0.5, targets_onehot_noise, confidences * self.scale2)
            targets_mix_corrected = targets_mix_corrected.repeat((q_mix.size(0) // q_logits1.size(0), 1))
            targets_mix_corrected = comb(targets_mix_corrected[mix_randind], targets_mix_corrected, mix_lam)

            targets_mix_noise = targets_onehot_noise.repeat((q_mix.size(0) // q_logits1.size(0), 1))
            targets_mix_noise = comb(targets_mix_noise[mix_randind], targets_mix_noise, mix_lam)

        align_logits = q_mix.mm(self.prototypes.T) / self.T

        def CE(logits, targets):
            return - (targets * F.log_softmax(logits, dim=1)).sum(-1).mean()

        if self.warmup:
            cls_loss1 = F.cross_entropy(q_logits1, labels) + \
                        F.cross_entropy(q_logits2, labels)
            cls_loss2 = CE(mix_logits, targets_mix_noise)
            align_loss = CE(align_logits, targets_mix_noise)
        else:
            align_loss = CE(align_logits, targets_mix_corrected)
            cls_loss1 = CE(q_logits1, targets_corrected1) + \
                        CE(q_logits2, targets_corrected2)
            cls_loss2 = CE(mix_logits, targets_mix_corrected)
        return cls_loss1, cls_loss2, align_loss
    def forward(self, im_w, im_q, im_k, indices):
        outputs = self.forward_loss(im_w, im_q, im_k, indices)

        return outputs

class SimCLRWrapper(Wrapper):
    def __init__(self,
                 encoder_type,
                 in_dim,
                 num_cluster,
                 fea_dim=128,
                 T=0.07,
                 mixup_alpha=1.0,
                 num_samples=0,
                 scale1=0.,
                 scale2=1.
                 ):
        super(SimCLRWrapper, self).__init__(encoder_type=encoder_type,
                                            in_dim=in_dim,
                                            num_cluster=num_cluster,
                                            fea_dim=fea_dim,
                                            mixup_alpha=mixup_alpha,
                                            num_samples=num_samples,
                                            scale1=scale1,
                                            scale2=scale2,
                                            T=T)
        from utils.infonce import InstanceLoss
        self.loss = InstanceLoss(T)

    def inference(self, im_w, im_q, im_k):
        im_mix, mix_randind, mix_lam = mixup(torch.cat([im_w, im_q, im_k]), alpha=self.mixup_alpha)
        # compute query features
        x_q = self.encoder_q(torch.cat([im_w, im_mix, im_q, im_k]))  # queries: NxC
        q = self.projector_q(x_q)  # queries: NxC
        q_logits = self.classifier_q(x_q)  # queries: NxC

        q = nn.functional.normalize(q, dim=1)  # already normalized
        q_w, q_m, q1, q2 = q.split([im_w.size(0), im_mix.size(0), im_q.size(0), im_k.size(0)])
        w_logits, m_logits, q_logits1, q_logits2 = q_logits.split([im_w.size(0),
                                                                   im_mix.size(0), im_q.size(0), im_k.size(0)])

        return q_w, w_logits, q1, q_logits1, q2, q_logits2, q_m, m_logits, mix_randind, mix_lam

    def forward_contrastive_loss(self, q1, q2, indices):
        import torch.distributed as dist
        if dist.is_initialized():
            from utils.gather_layer import GatherLayer
            q1 = torch.cat(GatherLayer.apply(q1), dim=0)
            q2 = torch.cat(GatherLayer.apply(q2), dim=0)
        contrastive_loss = self.loss(q1, q2)
        return contrastive_loss

    def forward_loss(self, im_w, im_q, im_k, indices):
        q_w, w_logits, q1, q_logits1, q2, q_logits2, q_mix, logits_mix, mix_randind, mix_lam = \
            self.inference(im_w, im_q, im_k)

        contrastive_loss = self.forward_contrastive_loss(q1, q2, indices)

        cls_loss1, cls_loss2, align_loss = self.forward_cls_loss(q_w,
                                                                 w_logits,
                                                                 q_logits1,
                                                                 q_logits2,
                                                                 logits_mix,
                                                                 q_mix,
                                                                 mix_randind,
                                                                 mix_lam,
                                                                 indices)

        ent_loss, ne_loss = self.forward_reg_loss(torch.cat([q_logits1, q_logits2, logits_mix]))

        return contrastive_loss, cls_loss1, cls_loss2, ent_loss, ne_loss, align_loss

#models/configs/cifar100_90_prer18.yml
batch_size: 480
num_devices: 1
wandb: false
project_name: noise_label
entity: zzhuang
dataset: cifar10
resized_crop_scale: 0.2
label_file: /kaggle/input/twin-contrastive-learning/TCL-master/models/tcl/data/sym_noise_cifar10_20.npy
encoder_name: bigresnet18_preact
epochs: 200
feat_dim: 256
img_size: 32
learning_rate: 0.03
learning_eta_min: 0.01
syncbn: true
reassign: 1
save_freq: 50
save_checkpoints: true
temp: 0.25
use_gaussian_blur: false
warmup_epochs: 20
weight_decay: 0.001
dist: false
num_workers: 32
model_name: tcl
mixup_alpha: 1.0

#utils/sampler.py:
import torch
import torch.distributed as dist
from torch.utils.data.sampler import Sampler

class RandomSampler(Sampler):

    def __init__(self, dataset=None, batch_size=0, num_iter=None, restore_iter=0,
                 weights=None, replacement=True, seed=0, shuffle=True, num_replicas=None, rank=None):
        super(RandomSampler, self).__init__(dataset)
        self.dist = dist.is_initialized()
        if self.dist:
            self.num_replicas = dist.get_world_size()
            self.rank = dist.get_rank()
        else:
            self.num_replicas = 1
            self.rank = 0
        if num_replicas is not None:
            self.num_replicas = num_replicas
        if rank is not None:
            self.rank = rank
        self.dataset = dataset
        self.batch_size = batch_size * self.num_replicas
        self.num_samples = num_iter * self.batch_size
        self.restore = restore_iter * self.batch_size
        self.weights = weights
        self.replacement = replacement
        self.seed = seed
        self.shuffle = shuffle

    def __iter__(self):
        # deterministically shuffle
        g = torch.Generator()
        g.manual_seed(self.seed)
        if self.shuffle:
            if self.weights is None:
                n = len(self.dataset)
                epochs = self.num_samples // n + 1
                indices = []
                for e in range(epochs):
                    g = torch.Generator()
                    g.manual_seed(self.seed + e)
                    # drop last
                    indices.extend(torch.randperm(len(self.dataset), generator=g).tolist()[:n - n % self.batch_size])
                indices = indices[:self.num_samples]
                # indices = torch.randint(high=n, size=(self.num_samples,), dtype=torch.int64, generator=g).tolist()
            else:
                indices = torch.multinomial(self.weights, self.num_samples, self.replacement, generator=g).tolist()
        else:
            raise NotImplementedError('No shuffle has not been implemented.')

        # subsample
        indices = indices[self.restore + self.rank:self.num_samples:self.num_replicas]

        return iter(indices)

    def __len__(self):
        return (self.num_samples - self.restore) // self.num_replicas

    def set_epoch(self, epoch: int) -> None:
        self.seed = epoch

    def set_weights(self, weights: torch.Tensor) -> None:
        self.weights = weights

#utils/optimizers.py:
# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch.optim import *  # noqa: F401,F403
from torch.optim.optimizer import Optimizer

class LARS(Optimizer):
    
    def __init__(self,
                 params,
                 lr,
                 momentum=0,
                 weight_decay=0,
                 dampening=0,
                 eta=0.001,
                 nesterov=False,
                 eps=1e-8):

        if momentum < 0.0:
            raise ValueError(f'Invalid momentum value: {momentum}')
        if weight_decay < 0.0:
            raise ValueError(f'Invalid weight_decay value: {weight_decay}')
        if eta < 0.0:
            raise ValueError(f'Invalid LARS coefficient value: {eta}')
        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
            eta=eta)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError(
                'Nesterov momentum requires a momentum and zero dampening')

        self.eps = eps
        super(LARS, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(LARS, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            eta = group['eta']
            nesterov = group['nesterov']
            lr = group['lr']
            lars_exclude = group.get('lars_exclude', False)
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad
                if lars_exclude:
                    local_lr = 1.
                else:
                    weight_norm = torch.norm(p).item()
                    grad_norm = torch.norm(d_p).item()
                    if weight_norm != 0 and grad_norm != 0:
                        # Compute local learning rate for this layer
                        local_lr = eta * weight_norm / \
                            (grad_norm + weight_decay * weight_norm + self.eps)
                    else:
                        local_lr = 1.
                actual_lr = local_lr * lr
                d_p = d_p.add(p, alpha=weight_decay).mul(actual_lr)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = \
                                torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf
                p.add_(-d_p)

        return loss
    
#utils/mutlicrop_transformer.py:

import random
import cv2
from PIL import ImageFilter
import numpy as np
import torchvision.transforms as transforms

class MultiCropTransform(object):

    def __init__(self,
                 old_transform: transforms.Compose,
                 size_crops,
                 nmb_crops,
                 min_scale_crops,
                 max_scale_crops):
        assert len(size_crops) == len(nmb_crops)
        assert len(min_scale_crops) == len(nmb_crops)
        assert len(max_scale_crops) == len(nmb_crops)

        trans = []
        for i in range(len(size_crops)):
            # REPLACE
            transform = []
            for t in old_transform.transforms:
                if isinstance(t, transforms.RandomResizedCrop):
                    transform.append(transforms.RandomResizedCrop(
                        size_crops[i],
                        scale=(min_scale_crops[i], max_scale_crops[i]),
                    ))
                    continue
                transform.append(t)

            trans.extend([transforms.Compose(transform)] * nmb_crops[i])
        self.trans = trans

    def __call__(self, img):
        multi_crops = list(map(lambda trans: trans(img), self.trans))
        return multi_crops
    
#utils/model_register.py:

import torch
import numpy as np
from torch import nn
import torch
import torch.nn.functional as F
from typing import Union
import os.path as osp
import os
import time
from torchvision.utils import save_image
import torch.distributed as dist
import math
import inspect
import collections.abc as string_classes
import collections.abc as container_abcs
import warnings
from utils.ops import load_network

def get_varname(var):
    """
    Gets the name of var. Does it from the out most frame inner-wards.
    :param var: variable to get name from.
    :return: string
    """
    for fi in reversed(inspect.stack()):
        names = [var_name for var_name, var_val in fi.frame.f_locals.items() if var_val is var]
        if len(names) > 0:
            return names[0]

def reduce_tensor(rt):
    rt = rt.clone()
    if dist.is_initialized():
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        world_size = dist.get_world_size()
    else:
        world_size = 1
    rt /= world_size
    return rt

class LoggerX(object):

    def __init__(self, save_root, enable_wandb=False, **kwargs):
        # assert dist.is_initialized()
        self.models_save_dir = osp.join(save_root, 'save_models')
        self.images_save_dir = osp.join(save_root, 'save_images')
        os.makedirs(self.models_save_dir, exist_ok=True)
        os.makedirs(self.images_save_dir, exist_ok=True)
        self._modules = []
        self._module_names = []
        # self.world_size = dist.get_world_size()
        if dist.is_initialized():
            self.world_size = dist.get_world_size() 
            self.local_rank = dist.get_rank()
        else:
            self.world_size = 1  # Default value for single-GPU runs
            self.local_rank = 0
        self.enable_wandb = enable_wandb
        if enable_wandb and self.local_rank == 0:
            import wandb
            wandb.init(dir=save_root, settings=wandb.Settings(_disable_stats=True, _disable_meta=True), **kwargs)

    @property
    def modules(self):
        return self._modules

    @property
    def module_names(self):
        return self._module_names

    @modules.setter
    def modules(self, modules):
        for i in range(len(modules)):
            self._modules.append(modules[i])
            self._module_names.append(get_varname(modules[i]))

    def append(self, module, name=None):
        self._modules.append(module)
        if name is None:
            name = get_varname(module)
        self._module_names.append(name)

    def checkpoints(self, epoch):
        if self.local_rank != 0:
            return
        for i in range(len(self.modules)):
            module_name = self.module_names[i]
            module = self.modules[i]
            # if epoch>190:
            torch.save(module.state_dict(), osp.join(self.models_save_dir, '{}-{}'.format(module_name, epoch)))

    def load_checkpoints(self, epoch):
        for i in range(len(self.modules)):
            module_name = self.module_names[i]
            module = self.modules[i]
            module.load_state_dict(load_network(osp.join(self.models_save_dir, '{}-{}'.format(module_name, epoch))))

    def msg(self, stats, step):
        output_str = '[{}] {:05d}, '.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), step)
        output_dict = {}
        for i in range(len(stats)):
            if isinstance(stats, (list, tuple)):
                var = stats[i]
                var_name = get_varname(stats[i])
            elif isinstance(stats, dict):
                var_name, var = list(stats.items())[i]
            else:
                raise NotImplementedError
            if isinstance(var, torch.Tensor):
                var = var.detach().mean()
                var = reduce_tensor(var)
                var = var.item()
            output_str += '{} {:2.5f}, '.format(var_name, var)
            output_dict[var_name] = var

        if self.enable_wandb and self.local_rank == 0:
            import wandb
            wandb.log(output_dict, step)

        if self.local_rank == 0:
            print(output_str)

    def msg_str(self, output_str):
        if self.local_rank == 0:
            print(str(output_str))

    def save_image(self, img, n_iter, sample_type):
        if isinstance(img, torch.Tensor):
            from torchvision.transforms.functional import to_pil_image
            img = to_pil_image(img.cpu())
        fname = f'{str(n_iter).zfill(7)}-{self.local_rank}-{sample_type}.png'
        img.save(osp.join(self.images_save_dir, fname))
        self.msg_str(f'save {fname} to {self.images_save_dir}')

#utils/knn_monitor.py:
# -*- coding: UTF-8 -*-

import torch
import tqdm
import torch.nn.functional as F
import torch.distributed as dist
# https://colab.research.google.com/github/facebookresearch/moco/blob/colab-notebook/colab/moco_cifar10_demo.ipynb
# test using a knn monitor
@torch.no_grad()
def knn_monitor(memory_features,
                memory_labels,
                test_features,
                test_labels,
                knn_k,
                knn_t):
    classes = len(torch.unique(memory_labels))
    pred_labels = knn_predict(test_features, memory_features, memory_labels, classes, knn_k, knn_t)

    top1 = (pred_labels[:, 0] == test_labels).float().mean()

    return top1

# knn monitor as in InstDisc https://arxiv.org/abs/1805.01978
# implementation follows http://github.com/zhirongw/lemniscate.pytorch and https://github.com/leftthomas/SimCLR
def knn_predict_internal(feature, feature_bank, feature_labels, classes, knn_k, knn_t):
    # compute cos similarity between each feature vector and feature bank ---> [B, N]
    sim_matrix = feature.mm(feature_bank.T)
    # [B, K]
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
    # [B, K]
    sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices)
    sim_weight = (sim_weight / knn_t).exp()
    # counts for each class
    one_hot_label = torch.zeros(feature.size(0) * knn_k, classes, device=sim_labels.device)
    # [B*K, C]
    one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
    # weighted score ---> [B, C]
    pred_scores = torch.sum(one_hot_label.view(feature.size(0), -1, classes) * sim_weight.unsqueeze(dim=-1), dim=1)
    pred_labels = pred_scores.argsort(dim=-1, descending=True)
    return pred_labels

def knn_predict(feature, feature_bank, feature_labels, classes, knn_k, knn_t):
    split_size = 512
    pred_labels = []
    for f in feature.split(split_size, dim=0):
        pred_labels.append(knn_predict_internal(f, feature_bank, feature_labels, classes, knn_k, knn_t))
    return torch.cat(pred_labels, dim=0)

#utils/infonce.py:
import torch
import torch.nn.functional as F
import torch.nn as nn

def mask_correlated_samples(batch_size):
    N = 2 * batch_size
    mask = torch.ones((N, N))
    mask = mask.fill_diagonal_(0)
    for i in range(batch_size):
        mask[i, batch_size + i] = 0
        mask[batch_size + i, i] = 0
    mask = mask.bool()
    return mask

def compute_infonce_loss(z_i, z_j, temperature, reduction: str = 'mean'):
    batch_size = z_i.size(0)
    mask = mask_correlated_samples(batch_size)
    N = 2 * batch_size
    z = torch.cat((z_i, z_j), dim=0)
    sim = torch.matmul(z, z.T) / temperature
    sim_i_j = torch.diag(sim, batch_size)
    sim_j_i = torch.diag(sim, -batch_size)
    positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
    negative_samples = sim[mask].reshape(N, -1)
    labels = torch.zeros(N).to(positive_samples.device).long()
    logits = torch.cat((positive_samples, negative_samples), dim=1)
    return F.cross_entropy(logits, labels, reduction=reduction)

class InstanceLoss(nn.Module):
    def __init__(self, temperature):
        super(InstanceLoss, self).__init__()
        self.temperature = temperature
        self.mask = None
        # self.mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, z_i, z_j):
        batch_size = z_i.size(0)
        if self.mask is None:
            self.mask = self.mask_correlated_samples(batch_size)
        N = 2 * batch_size
        z = torch.cat((z_i, z_j), dim=0)
        sim = torch.matmul(z, z.T) / self.temperature
        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss

if __name__ == '__main__':
    x1 = torch.zeros(128, 10)
    x2 = torch.zeros(128, 10)
    loss = compute_infonce_loss(x1, x2, 1.0, reduction='none')
    print(loss.size())

#utils/grad_scaler.py:

import torch
# from torch._six import inf
inf = float('inf') 

def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]),
                                norm_type)
    return total_norm

class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"
    def __init__(self,
                 optimizer=None,
                 amp=False,
                 clip_grad=None):
        self._scaler = torch.cuda.amp.GradScaler()
        self.clip_grad = clip_grad
        self.optimizer = optimizer
        self.amp = amp
    def __call__(self, loss, optimizer=None, clip_grad=None, parameters=None, update_grad=True, backward_kwargs={}):
        if optimizer is None:
            optimizer = self.optimizer
        if clip_grad is None:
            clip_grad = self.clip_grad
        if self.amp:
            self._scaler.scale(loss).backward(**backward_kwargs)
        else:
            loss.backward(**backward_kwargs)
        norm = None
        if update_grad:
            if self.amp:
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
            if clip_grad is not None:
                assert parameters is not None
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                if parameters is not None:
                    norm = get_grad_norm_(parameters)
            if self.amp:
                self._scaler.step(optimizer)
                self._scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
        return norm
    def state_dict(self):
        return self._scaler.state_dict()
    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)

#utils/gather_layer.py:

import torch
# from torch._six import inf
inf = float('inf') 
def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]),
                                norm_type)
    return total_norm

class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"
    def __init__(self,
                 optimizer=None,
                 amp=False,
                 clip_grad=None):
        self._scaler = torch.cuda.amp.GradScaler()
        self.clip_grad = clip_grad
        self.optimizer = optimizer
        self.amp = amp
    def __call__(self, loss, optimizer=None, clip_grad=None, parameters=None, update_grad=True, backward_kwargs={}):
        if optimizer is None:
            optimizer = self.optimizer
        if clip_grad is None:
            clip_grad = self.clip_grad
        if self.amp:
            self._scaler.scale(loss).backward(**backward_kwargs)
        else:
            loss.backward(**backward_kwargs)
        norm = None
        if update_grad:
            if self.amp:
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
            if clip_grad is not None:
                assert parameters is not None
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                if parameters is not None:
                    norm = get_grad_norm_(parameters)
            if self.amp:
                self._scaler.step(optimizer)
                self._scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
        return norm
    def state_dict(self):
        return self._scaler.state_dict()
    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)

#utils__init__.py:
from torchvision import transforms
from PIL import Image
import torch
import torch.nn.functional as F
import numpy as np
from munkres import Munkres
from sklearn import metrics
import warnings
import matplotlib.pyplot as plt
import tqdm
import torch.distributed as dist
from utils.ops import convert_to_cuda, is_root_worker
from .knn_monitor import knn_monitor
@torch.no_grad()
def extract_features(extractor, loader):
    extractor.eval()
    local_features = []
    local_labels = []
    for inputs in tqdm.tqdm(loader, disable=not is_root_worker()):
        images, labels = convert_to_cuda(inputs)
        local_labels.append(labels)
        local_features.append(extractor(images))
    local_features = torch.cat(local_features, dim=0)
    local_labels = torch.cat(local_labels, dim=0)
    indices = torch.Tensor(list(iter(loader.sampler))).long().cuda()
    features = torch.zeros(len(loader.dataset), local_features.size(1)).cuda()
    all_labels = torch.zeros(len(loader.dataset)).cuda()
    counts = torch.zeros(len(loader.dataset)).cuda()
    features.index_add_(0, indices, local_features)
    all_labels.index_add_(0, indices, local_labels.float())
    counts[indices] = 1.
    if dist.is_initialized():
        dist.all_reduce(features, op=dist.ReduceOp.SUM)
        dist.all_reduce(all_labels, op=dist.ReduceOp.SUM)
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)
    # account for the few samples that are computed twice
    labels = (all_labels / counts).long()
    features /= counts[:, None]
    return features, labels

# @torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    output = torch.cat(tensors_gather, dim=0)
    return output

@torch.no_grad()
def shuffling_forward(inputs, encoder):
    # shuffle for making use of BN
    inputs, idx_unshuffle = _batch_shuffle_ddp(inputs)
    inputs = encoder(inputs)  # keys: NxC
    # undo shuffle
    inputs = _batch_unshuffle_ddp(inputs, idx_unshuffle)
    return inputs

@torch.no_grad()
def _batch_shuffle_ddp(x):
    """
    Batch shuffle, for making use of BatchNorm.
    *** Only support DistributedDataParallel (DDP) model. ***
    """
    # gather from all gpus
    batch_size_this = x.shape[0]
    x_gather = concat_all_gather(x)
    batch_size_all = x_gather.shape[0]
    num_gpus = batch_size_all // batch_size_this
    # random shuffle index
    idx_shuffle = torch.randperm(batch_size_all).cuda()
    # broadcast to all gpus
    torch.distributed.broadcast(idx_shuffle, src=0)
    # index for restoring
    idx_unshuffle = torch.argsort(idx_shuffle)
    # shuffled index for this gpu
    gpu_idx = torch.distributed.get_rank()
    idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

    return x_gather[idx_this], idx_unshuffle

@torch.no_grad()
def _batch_unshuffle_ddp(x, idx_unshuffle):
    """
    Undo batch shuffle.
    *** Only support DistributedDataParallel (DDP) model. ***
    """
    # gather from all gpus
    batch_size_this = x.shape[0]
    x_gather = concat_all_gather(x)
    batch_size_all = x_gather.shape[0]
    num_gpus = batch_size_all // batch_size_this
    # restored index for this gpu
    gpu_idx = torch.distributed.get_rank()
    idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

    return x_gather[idx_this]

@torch.no_grad()
def _momentum_update(q_params, k_params, m):
    """
    Momentum update
    """
    if not isinstance(q_params, (list, tuple)):
        q_params, k_params = [q_params, ], [k_params, ]
    for param_q, param_k in zip(q_params, k_params):
        param_k.data = param_k.data * m + param_q.data * (1. - m)
class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform1, transform2=None):
        self.transform1 = transform1
        self.transform2 = transform1 if transform2 is None else transform2
    def __call__(self, x):
        return [self.transform1(x), self.transform2(x)]
    def __str__(self):
        return f'transform1 {str(self.transform1)} transform2 {str(self.transform2)}'
