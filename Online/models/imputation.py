import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim * 2, dim * 4),
            nn.BatchNorm1d(dim * 4),
            nn.ReLU(),
            nn.Linear(dim * 4, dim * 2),
            nn.BatchNorm1d(dim * 2),
            nn.ReLU(),
            nn.Linear(dim * 2, dim),
        )
        self.res_fc = nn.Linear(dim * 2, dim)

    def forward(self, f_v_partial, f_a):
        x = torch.cat([f_v_partial, f_a], dim=-1)
        out = self.net(x)
        res = self.res_fc(x)
        return out + res  # residual path helps stability


class Discriminator(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim * 2, dim * 2),
            nn.LayerNorm(dim * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(dim * 2, dim),
            nn.LeakyReLU(0.2),
            nn.Linear(dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, f_v, f_a):
        x = torch.cat([f_v, f_a], dim=-1)
        return self.net(x)



def imputation_loss(D, G, f_v_part, f_a, f_v_full, f_v_hat):
    real_score = D(f_v_full, f_a)
    fake_score = D(f_v_hat.detach(), f_a)

    loss_D = -torch.mean(
        torch.log(real_score + 1e-6)
      + torch.log(1.0 - fake_score + 1e-6)
    )

    pred_fake = D(f_v_hat, f_a)
    loss_G_adv = -torch.mean(torch.log(pred_fake + 1e-6))
    loss_G_reg = F.mse_loss(f_v_hat, f_v_full)
    loss_G = loss_G_adv + 0.5 * loss_G_reg

    return loss_D, loss_G
