import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectraEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(31, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 6)

    def forward(self, x):
        x = F.softplus(self.fc1(x))  # ✅ Softplus 作为非线性激活
        x = F.relu(self.fc2(x))  # ✅ 确保非线性足够
        x = F.softplus(self.fc3(x))
        x = self.fc4(x)  # ✅ 直接传递到 fc3
        return x


class SpectraDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(12, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 31)


    def forward(self, x):
        x = F.softplus(self.fc1(x))  # ✅ Softplus 确保非线性
        x = F.relu(self.fc2(x))  # ✅ 确保 fc2 也有激活
        x = F.softplus(self.fc3(x))
        x = self.fc4(x)  # ✅ 直接输出
        return x


class SpectraModel(nn.Module):
    def __init__(self, input_dim=31, latent_dim=6):
        super().__init__()
        self.encoder_R = SpectraEncoder()
        self.encoder_L = SpectraEncoder()
        self.decoder = SpectraDecoder()

        self.function_block = nn.Sequential(
            nn.Linear(12, 128),
            nn.Softplus(),
            nn.Linear(128, 64),
            nn.Softplus(),
            nn.Linear(64, 12),
        )

    def forward(self, R1, L1, R2, L2):
        zR1 = self.encoder_R(R1)
        zL1 = self.encoder_L(L1)
        zR2 = self.encoder_R(R2)
        zL2 = self.encoder_L(L2)
        # ZC = (zR1 * zL1) + (zR2 * zL2)

        # **合并 latent 并输入 function_block**
        new_latent1 = self.function_block(torch.cat((zR1, zL1), dim=1))  # [batch, 12]
        new_latent2 = self.function_block(torch.cat((zR2, zL2), dim=1))  # [batch, 12]

        # **最终 latent 送入 decoder**
        final_latent = new_latent1 + new_latent2
        output = self.decoder(final_latent)

        return output, zR1, zL1, zR2, zL2


# =========================================================
# 6. 损失函数
# =========================================================
def loss_function(pred, target):
    return F.mse_loss(pred, target)


def scale_linearity_loss(model, reflectance, illumination):
    device = reflectance.device
    batch_size = reflectance.shape[0]

    alphaR = torch.rand(batch_size, 1, device=device) * 0.9 + 0.1
    alphaL = torch.rand(batch_size, 1, device=device) * 9.0 + 1.0

    R_scaled = reflectance * alphaR
    L_scaled = illumination * alphaL

    zR = model.encoder_R(reflectance)
    zR_scaled = model.encoder_R(R_scaled)
    zL = model.encoder_L(illumination)
    zL_scaled = model.encoder_L(L_scaled)

    scale_diff_R = zR_scaled[:, 0] - (zR[:, 0] * alphaR.squeeze(1))
    scale_diff_L = zL_scaled[:, 0] - (zL[:, 0] * alphaL.squeeze(1))

    broad_diff_R = zR_scaled[:, 1:] - zR[:, 1:]
    broad_diff_L = zL_scaled[:, 1:] - zL[:, 1:]

    scale_loss = torch.mean(scale_diff_R ** 2 + scale_diff_L ** 2)
    shape_loss = torch.mean(broad_diff_R ** 2 + broad_diff_L ** 2)

    return scale_loss + shape_loss


def total_loss_function(model, pred, target, reflectance, illumination, scale_weight=0.1):
    mse_loss = loss_function(pred, target)
    scale_lin_loss = scale_linearity_loss(model, reflectance, illumination)
    # total_loss = mse_loss + scale_weight * scale_lin_loss
    total_loss = mse_loss
    return total_loss