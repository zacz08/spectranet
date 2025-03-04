import torch
import os
import torch.optim as optim
from torch.utils.data import DataLoader
from src.SpectraDataset import SpectraDataset
from src.ANet import total_loss_function, SpectraModel


# =========================================================
batch_size = 2048
epochs = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# =========================================================

def train(model, train_loader, optimizer, device):
    model.train()

    # log settings
    num_samples = len(train_loader)
    num_logs_per_epoch = 5  # 每个 epoch 记录 5 次 loss
    steps_per_log = num_samples // num_logs_per_epoch  # 每隔多少步打印一次 loss

    total_loss = 0

    for batch_idx, batch in enumerate(train_loader):
        R1, L1, R2, L2, target = [b.to(device) for b in batch]

        # 前向传播
        pred, zR1, zL1, zR2, zL2 = model(R1, L1, R2, L2)

        # 计算损失
        loss = total_loss_function(model, pred, target, R1, L1)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # 每 steps_per_log 步打印一次 loss
        if (batch_idx + 1) % steps_per_log == 0:
            print(f"  Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.6f}")

    # 在每个 epoch 结束后保存模型
    avg_loss = total_loss / len(train_loader)
    save_checkpoint(model, optimizer, epoch+1, avg_loss)


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


def save_checkpoint(model, optimizer, epoch, loss, save_dir="ckpts"):
    """ 保存模型到文件 """
    os.makedirs(save_dir, exist_ok=True)  # 创建目录
    save_path = os.path.join(save_dir, f"model.ckpt")
    
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }
    
    torch.save(checkpoint, save_path)


if __name__ == '__main__':

    # 加载数据集
    train_dataset = SpectraDataset(data_split='train')
    train_loader = DataLoader(train_dataset, num_workers=0, batch_size=batch_size, shuffle=True)

    test_dataset = SpectraDataset(data_split='test')
    test_loader = DataLoader(test_dataset, num_workers=0, batch_size=batch_size, shuffle=False)

    # 初始化模型和优化器
    model = SpectraModel(input_dim=31, latent_dim=9).to(device)
    optimizer = optim.Adam(model.parameters(), lr=5e-4)

    # 训练循环
    for epoch in range(epochs):
        print(f"\nEpoch [{epoch+1}/{epochs}]")
        
        # 训练
        train(model, train_loader, optimizer, device)

        # 测试
        test_loss = test(model, test_loader, device)
        print(f"  Test Loss: {test_loss:.6f}")
    


