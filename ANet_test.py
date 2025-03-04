import torch
import matplotlib.pyplot as plt
from src.SpectraDataset import SpectraDataset
from src.ANet import total_loss_function, SpectraModel


# =========================================================
batch_size = 2048
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# =========================================================


def test(model, test_loader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in test_loader:
            R1, L1, R2, L2, target = [b.to(device) for b in batch]

            pred, zR1, zL1, zR2, zL2 = model(R1, L1, R2, L2)

            loss = total_loss_function(model, pred, target, R1, L1)
            total_loss += loss.item()

    return total_loss / len(test_loader)  # 返回平均 test loss




if __name__ == '__main__':

    # 加载模型
    checkpoint = torch.load("./ckpts/model.ckpt")
    if 'model_state_dict' in checkpoint:
        checkpoint = checkpoint['model_state_dict']

    model = SpectraModel(input_dim=31, latent_dim=9)  # 你的模型结构必须匹配保存时的结构
    model.load_state_dict(checkpoint)  # 加载 state_dict

    model.eval()

    # 加载数据集
    test_dataset = SpectraDataset(data_split='test')
    R1, L1, R2, L2, target = test_dataset[0]

    # 初始化模型
    model = SpectraModel(input_dim=31, latent_dim=9)
    pred, _, _, _, _ = model(R1.unsqueeze(0), L1.unsqueeze(0), R2.unsqueeze(0), L2.unsqueeze(0))
    pred = pred.squeeze(0)  # 去掉 batch 维度

    # 绘制曲线
    plt.figure(figsize=(8, 5))
    plt.plot(target.cpu().numpy(), label="Target", linestyle="--", marker="o")
    plt.plot(pred.cpu().detach().numpy(), label="Predicted", linestyle="-", marker="x")
    plt.xlabel("Spectral Index")
    plt.ylabel("Intensity")
    plt.title("target vs truth")
    plt.legend()
    plt.grid()
    plt.show()
    


