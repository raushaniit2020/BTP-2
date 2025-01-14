import torch
import torch.nn.functional as F
import torch.nn as nn

print("I am inside utils/infonce.py")
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
