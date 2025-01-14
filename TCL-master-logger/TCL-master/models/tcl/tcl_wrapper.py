import torch
import torch.nn as nn
import torch.nn.functional as F

print("I am inside models/tcl/tcl_wrapper.py")
def mixup(input, alpha=1.0):
    bs = input.size(0)
    print("Batch size:", bs)
    randind = torch.randperm(bs).to(input.device)
    print("Random indices for shuffling:", randind)
    # beta = torch.distributions.beta.Beta(alpha, alpha)
    # lam = beta.sample([bs]).to(input.device)
    import numpy as np
    lam = np.random.beta(alpha, alpha)
    print("Generated mixing coefficient (lambda):", lam)
    lam = torch.ones_like(randind).float() * lam
    lam = torch.max(lam, 1. - lam)
    lam_expanded = lam.view([-1] + [1] * (input.dim() - 1))
    print("Expanded mixing coefficient (lambda):", lam_expanded.shape)
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
        print("Temperature (T):", self.T)
        self.num_cluster = num_cluster
        print("Number of clusters:", self.num_cluster)
        self.warmup = False
        self.in_dim = in_dim
        print("Input dimension (in_dim):", self.in_dim)
        self.fea_dim = fea_dim
        print("Feature dimension (fea_dim):", self.fea_dim)
        self.mixup_alpha = mixup_alpha
        print("Mixup alpha:", self.mixup_alpha)
        self.scale1 = scale1
        print("Scale 1:", self.scale1)
        self.scale2 = scale2
        print("Scale 2:", self.scale2)
        self.num_samples = num_samples
        print("Number of samples:", self.num_samples)

        self.encoder_q = encoder_type()
        print("Encoder:", self.encoder_q)
        self.projector_q = self.create_projector(self.in_dim, self.fea_dim)
        print("Projector (Q):", self.projector_q)
        self.classifier_q = self.create_classifier(self.in_dim, self.num_cluster)
        print("Classifier (Q):", self.classifier_q)

        self.register_buffer('prototypes', torch.randn(self.num_cluster, fea_dim))
        print("Prototypes shape:", self.prototypes.shape)
        self.prototypes = F.normalize(self.prototypes, dim=1)
        self.register_buffer('confidences', torch.zeros(self.num_samples))
        print("Confidences shape:", self.confidences.shape)
        self.register_buffer('context_assignments', torch.zeros(self.num_samples, self.num_cluster))

    def inference(self, im_w, im_q, im_k):
        print(" this inference function does nothing ")
        pass

    def forward_contrastive_loss(self, q, k, indices):
        print("this forward contrastive loss function does nothing")
        pass

    def forward_reg_loss(self, pred_logits):
        pred_softmax = F.softmax(pred_logits, dim=1)
        print("Softmax probabilities shape:", pred_softmax.shape)
        ent_loss = - (pred_softmax * F.log_softmax(pred_logits, dim=1)).sum(dim=1).mean()
        print("Entropy loss:", ent_loss.item())
        prob_mean = pred_softmax.mean(dim=0)
        print("Mean probability shape:", prob_mean.shape)
        ne_loss = (prob_mean * prob_mean.log()).sum()
        print("Negative entropy loss:", ne_loss.item())
        return ent_loss, ne_loss

    def forward_loss(self, im_w, im_q, im_k, indices):
        print("this forward loss function does not do anything")
        pass

    def forward_cls_loss(self,
                         q_w, w_logits,
                         q_logits1, q_logits2, mix_logits,
                         q_mix, mix_randind, mix_lam, indices):

        with torch.no_grad():
            labels = self.pseudo_labels[indices]
            print("Pseudo labels shape:", labels.shape)
            confidences = self.confidences[indices].unsqueeze(1)
            print("Confidences shape:", confidences.shape)
            targets_onehot_noise = F.one_hot(labels, self.num_cluster).float().cuda()
            print("One-hot encoded targets shape:", targets_onehot_noise.shape)
            w_prob = F.softmax(w_logits.detach(), dim=1)
            q_prob1 = F.softmax(q_logits1.detach(), dim=1)
            q_prob2 = F.softmax(q_logits2.detach(), dim=1)
            print("Softmax probabilities shapes - w_prob:", w_prob.shape, "q_prob1:", q_prob1.shape, "q_prob2:", q_prob2.shape)

            # targets_mix_corrected = (w_prob + q_prob1 + q_prob2) / 3.

            def comb(p1, p2, lam):
                return (1 - lam) * p1 + lam * p2

            targets_corrected1 = comb(q_prob2, targets_onehot_noise, confidences * self.scale1)
            targets_corrected2 = comb(q_prob1, targets_onehot_noise, confidences * self.scale1)
            print("Targets corrected shapes - targets_corrected1:", targets_corrected1.shape, "targets_corrected2:", targets_corrected2.shape)
            targets_mix_corrected = comb((q_prob1 + q_prob2) * 0.5, targets_onehot_noise, confidences * self.scale2)
            targets_mix_corrected = targets_mix_corrected.repeat((q_mix.size(0) // q_logits1.size(0), 1))
            targets_mix_corrected = comb(targets_mix_corrected[mix_randind], targets_mix_corrected, mix_lam)
            print("Targets mix corrected shape:", targets_mix_corrected.shape)

            targets_mix_noise = targets_onehot_noise.repeat((q_mix.size(0) // q_logits1.size(0), 1))
            targets_mix_noise = comb(targets_mix_noise[mix_randind], targets_mix_noise, mix_lam)
            print("Mixed noise targets shape:", targets_mix_noise.shape)

        align_logits = q_mix.mm(self.prototypes.T) / self.T
        print("Alignment logits shape:", align_logits.shape)

        def CE(logits, targets):
            return - (targets * F.log_softmax(logits, dim=1)).sum(-1).mean()

        if self.warmup:
            cls_loss1 = F.cross_entropy(q_logits1, labels) + \
                        F.cross_entropy(q_logits2, labels)
            print("Classification loss 1 during warmup:", cls_loss1.item())
            cls_loss2 = CE(mix_logits, targets_mix_noise)
            print("Classification loss 2 during warmup:", cls_loss2.item())
            align_loss = CE(align_logits, targets_mix_noise)
            print("Alignment loss during warmup:", align_loss.item())
        else:
            align_loss = CE(align_logits, targets_mix_corrected)
            print("Alignment loss after warmup:", align_loss.item())
            cls_loss1 = CE(q_logits1, targets_corrected1) + \
                        CE(q_logits2, targets_corrected2)
            print("Classification loss 1 after warmup:", cls_loss1.item())
            cls_loss2 = CE(mix_logits, targets_mix_corrected)
            print("Classification loss 2 after warmup:", cls_loss2.item())

        return cls_loss1, cls_loss2, align_loss

    def forward(self, im_w, im_q, im_k, indices):
        outputs = self.forward_loss(im_w, im_q, im_k, indices)
        print("Outputs after forward pass:", outputs)
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
        print("Contrastive Loss:", contrastive_loss)
        print("Classification Loss 1:", cls_loss1)
        print("Classification Loss 2:", cls_loss2)
        print("Entropy Loss:", ent_loss)
        print("Negative Entropy Loss:", ne_loss)
        print("Alignment Loss:", align_loss)
        return contrastive_loss, cls_loss1, cls_loss2, ent_loss, ne_loss, align_loss
