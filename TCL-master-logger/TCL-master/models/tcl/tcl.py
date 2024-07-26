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

print("I am inside models/tcl/tcl.py")
@model_dict.register('tcl')
class TCL(TrainTask):

    def set_model(self):
        opt = self.opt
        print('Setting up model...')
        encoder_type, dim_in = backbone_dict[opt.encoder_name]
        print('Encoder type:', encoder_type)
        print('Dimension in:', dim_in)
        kwargs = {'encoder_type': encoder_type, 'in_dim': dim_in, 'fea_dim': opt.feat_dim, 'T': opt.temp,
                  'num_cluster': self.num_cluster, 'mixup_alpha': opt.mixup_alpha, 'num_samples': self.num_samples,
                  'scale1': opt.scale1, 'scale2': opt.scale2}
        print('Model setup kwargs:', kwargs)
        if opt.arch == 'simclr':
            print('Initializing SimCLRWrapper model...')
            tcl = SimCLRWrapper(**kwargs).cuda()
        else:
            raise NotImplemented
        tcl.register_buffer('pseudo_labels', self.gt_labels.cpu())
        print('Registered pseudo labels buffer on CPU.')

        if opt.syncbn:
            print('Converting model to SyncBatchNorm...')
            tcl = torch.nn.SyncBatchNorm.convert_sync_batchnorm(tcl)

        params = list(tcl.parameters())
        # print('Initialized parameters for optimization:', params)
        optimizer = torch.optim.SGD(params=params, lr=opt.learning_rate, momentum=opt.momentum,
                                    weight_decay=opt.weight_decay)
        print('Initialized optimizer with learning rate:', opt.learning_rate)
        # tcl = convert_to_ddp(tcl)
        self.logger.modules = [tcl, optimizer]
        self.tcl = tcl
        self.optimizer = optimizer
        self.tcl.register_buffer('pseudo_labels', self.gt_labels.cpu().cuda())  # Ensure CUDA placement
        print('Model setup completed.')

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
        print('Training TCL model...')
        is_warmup = not (self.cur_epoch >= opt.warmup_epochs)
        print('Is warmup:', is_warmup)
        self.tcl.warmup = is_warmup

        images, _ = inputs
        self.tcl.train()

        im_w, im_q, im_k = images

        indices = indices.cpu()
        # compute loss
        contrastive_loss, cls_loss1, cls_loss2, ent_loss, ne_loss, align_loss = self.tcl(im_w, im_q, im_k, indices)
        print('Losses - contrastive_loss:', contrastive_loss.item(), 'cls_loss1:', cls_loss1.item(), 
          'cls_loss2:', cls_loss2.item(), 'ent_loss:', ent_loss.item(), 'ne_loss:', ne_loss.item(), 
          'align_loss:', align_loss.item())
        # SGD
        self.optimizer.zero_grad()
        loss = contrastive_loss + \
               opt.cls_loss_weight * (cls_loss1 + cls_loss2) + \
               opt.ent_loss_weight * ent_loss + \
               opt.ne_loss_weight * ne_loss + \
               opt.align_loss_weight * align_loss
        print('Total loss:', loss.item())
        loss.backward()
        self.optimizer.step()
        self.logger.msg([contrastive_loss, cls_loss1, cls_loss2, ent_loss, ne_loss, align_loss], n_iter)
        print('Training completed.')    

    def extract_features(self, model, loader):
        opt = self.opt
        print('Extracting features from the model...')
        features = torch.zeros(len(loader.dataset), opt.feat_dim).cuda()
        print('Initialized features tensor shape:', features.shape)
        all_labels = torch.zeros(len(loader.dataset)).cuda()
        print('Initialized all_labels tensor shape:', all_labels.shape)
        cluster_labels = torch.zeros(len(loader.dataset), self.num_cluster).cuda()
        print('Initialized cluster_labels tensor shape:', cluster_labels.shape)

        model.eval()
        # encoder = model.module.encoder_q
        # classifier = model.module.classifier_q
        # projector = model.module.projector_q
        encoder = model.encoder_q.cuda()
        classifier = model.classifier_q.cuda()
        projector = model.projector_q.cuda()

        local_features = []
        local_labels = []
        local_cluster_labels = []
        for inputs in tqdm.tqdm(loader, disable=not self.verbose):
            images, labels = convert_to_cuda(inputs)
            local_labels.append(labels)
            print('Processed images shape:', images.shape, 'labels shape:', labels.shape)
            x = encoder(images)
            print('Encoder output shape:', x.shape)
            local_cluster_labels.append(F.softmax(classifier(x), dim=1))
            print('Local cluster labels shape:', local_cluster_labels[-1].shape)
            local_features.append(F.normalize(projector(x), dim=1))
            print('Local features shape:', local_features[-1].shape)
        local_features = torch.cat(local_features, dim=0)
        local_labels = torch.cat(local_labels, dim=0)
        local_cluster_labels = torch.cat(local_cluster_labels, dim=0)

        indices = torch.Tensor(list(iter(loader.sampler))).long().cuda()

        features.index_add_(0, indices, local_features)
        print('Updated features tensor shape after indexing:', features.shape)
        all_labels.index_add_(0, indices, local_labels.float())
        print('Updated all_labels tensor shape after indexing:', all_labels.shape)
        cluster_labels.index_add_(0, indices, local_cluster_labels.float())
        print('Updated cluster_labels tensor shape after indexing:', cluster_labels.shape)

        if dist.is_initialized():
            dist.all_reduce(features, op=dist.ReduceOp.SUM)
            dist.all_reduce(all_labels, op=dist.ReduceOp.SUM)
            dist.all_reduce(cluster_labels, op=dist.ReduceOp.SUM)

            mask = torch.norm(features, dim=1) > 1.5
            all_labels[mask] = all_labels[mask] / dist.get_world_size()
            cluster_labels[mask] = cluster_labels[mask] / dist.get_world_size()
            features = F.normalize(features, dim=1)
            print('Reduced and normalized features tensor shape:', features.shape)
        labels = all_labels.long()
        print('Final labels tensor shape:', labels.shape)
        print('Feature extraction completed.')
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
        print('Obtained ground truth labels:', len(labels))
        confidence, context_assignments, features, cluster_labels, = self.correct_labels(self.tcl, labels)
        print('Obtained confidence, context assignments, features, and cluster labels from correct_labels function.')
        self.tcl.confidences.copy_(confidence.float())
        print('Copied confidence values to the tcl model.')
        self.evaluate(self.tcl, features, confidence, cluster_labels, labels, context_assignments, n_iter)
        print('Evaluated the tcl model using obtained features, confidence, cluster labels, and context assignments.')

    def evaluate(self, model, features, confidence, cluster_labels, labels, context_assignments, n_iter):
        opt = self.opt
        clean_labels = torch.Tensor(
            self.create_dataset(opt.data_folder, opt.dataset, train=True, transform=None)[0].targets).cuda().long()
        print('Clean labels from the training dataset:', len(clean_labels))
        is_clean = clean_labels.cpu().numpy() == labels.cpu().numpy()
        print('Check if each label is clean:', is_clean)
        self.hist(context_assignments, is_clean, labels, n_iter)
        print('Histogram of context assignments computed.')
        train_acc = (torch.argmax(cluster_labels, dim=1) == clean_labels).float().mean()
        print('Training accuracy:', train_acc)
        test_features, test_cluster_labels, test_labels = self.extract_features(model, self.test_loader)
        print('Extracted features, test cluster labels, and test labels.')
        test_acc = (test_labels == torch.argmax(test_cluster_labels, dim=1)).float().mean()
        print('Test accuracy:', test_acc)
        from utils.knn_monitor import knn_predict
        knn_labels = knn_predict(test_features, features, clean_labels,
                                 classes=self.num_cluster, knn_k=200, knn_t=0.1)[:, 0]
        print('KNN labels predicted for the test features:', knn_labels)
        self.logger.msg_str(torch.unique(torch.argmax(test_cluster_labels, dim=1), return_counts=True))

        knn_acc = (test_labels == knn_labels).float().mean()
        print('KNN accuracy:', knn_acc)

        estimated_noise_ratio = (confidence > 0.5).float().mean().item()
        print('Estimated noise ratio:', estimated_noise_ratio)
        if opt.scale1 is None:
            self.tcl.scale1 = estimated_noise_ratio
            print('Updated scale1 of tcl model:', self.tcl.scale1)
        if opt.scale2 is None:
            self.tcl.scale2 = estimated_noise_ratio
            print('Updated scale2 of tcl model:', self.tcl.scale2)

        noise_accuracy = ((confidence > 0.5) == (clean_labels == labels)).float().mean()
        print('Noise accuracy:', noise_accuracy)
        from sklearn.metrics import roc_auc_score
        context_noise_auc = roc_auc_score(is_clean, confidence.cpu().numpy())
        print('Context noise AUC:', context_noise_auc)
        self.logger.msg([estimated_noise_ratio, noise_accuracy,
                         context_noise_auc, train_acc, test_acc, knn_acc], n_iter)

    def correct_labels(self, model, labels):
        opt = self.opt

        features, cluster_labels, _ = self.extract_features(model, self.memory_loader)
        print('Extracted features and cluster labels from the memory loader.')
        confidence, context_assignments, centers = self.noise_detect(cluster_labels, labels, features)
        print('Computed confidence, context assignments, and centers for noise detection.')

        # model.module.prototypes.copy_(centers)
        # model.module.context_assignments.copy_(context_assignments.float())
        model.prototypes.copy_(centers)
        model.context_assignments.copy_(context_assignments.float())
        print('Copied centers and context assignments to the model.')

        return confidence, context_assignments, features, cluster_labels

    def noise_detect(self, cluster_labels, labels, features):
        opt = self.opt
        print("Parameters of the config file:", opt)
        centers = F.normalize(cluster_labels.T.mm(features), dim=1)
        print("Computed cluster centers after matrix multiplication:", centers.shape)  # Printing the shape of computed cluster centers
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
        print("Computed confidence scores:", confidence.shape)
        return confidence, context_assignments, centers

    def test(self, n_iter):
        print(" this function does nothing--------------------------")
        pass

    def train_transform(self, normalize):
        '''
        simclr transform
        :param normalize:
        :return:
        '''
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
