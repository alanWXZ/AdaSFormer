import torch
import torch.nn as nn
import torch.nn.functional as F

class GumbelSoftmaxParameter(nn.Module):
    def __init__(self, num_bins=10, tau_init=5.0, tau_min=0.5, anneal_rate=0.01):
        super().__init__()
        # 可学习参数，相当于每个bin的logit
        self.logits = nn.Parameter(torch.zeros(num_bins))
        self.num_bins = num_bins
        # bin的取值，可以根据任务调整范围，比如 [0, 1]
        self.register_buffer("bin_values", torch.linspace(0, 1, num_bins))

        self.tau_init = tau_init
        self.tau_min = tau_min
        self.anneal_rate = anneal_rate
        self.tau = tau_init

    def update_tau(self, epoch):
        self.tau = max(self.tau_init * torch.exp(torch.tensor(-self.anneal_rate * epoch)), self.tau_min)

    def forward(self, hard=False):
        # Gumbel-Softmax 采样
        y_soft = F.gumbel_softmax(self.logits, tau=self.tau, hard=False)  # [num_bins]

        # 预测值 = soft one-hot * bin 值
        y_hard = F.gumbel_softmax(self.logits, tau=self.tau, hard=True)

        y_st = y_hard.detach() - y_soft.detach() + y_soft
        return y_st


# ===== 测试 =====
if __name__ == "__main__":
    torch.manual_seed(0)
    model = GumbelSoftmaxParameter(num_bins=8, tau=0.5)

    # 假设目标是 0.7
    target = torch.tensor(0.7)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    for step in range(100):
        optimizer.zero_grad()
        pred, probs = model(hard=False)
        loss = F.mse_loss(pred, target)
        loss.backward()
        optimizer.step()

        if step % 20 == 0:
            print(f"Step {step}: pred={pred.item():.3f}, loss={loss.item():.4f}")

    print("最终预测:", model(hard=True)[0].item())
    print("概率分布:", model(hard=False)[1].detach())
