import numpy as np
import torch, math
import torch.nn as nn
from entmax import entmax15, sparsemax

class DropBlock(nn.Module):
    def __init__(self, p, bs=1):
        super(DropBlock, self).__init__()
        self.p = p
        self.bs = bs

    def forward(self, x):
        gamma = self.p / (self.bs ** 2)
        mask = (torch.rand(x.shape[0], *x.shape[2:]) < gamma)
        bm = self._compute_block_mask(mask.float().to(x.device))
        return x * bm[:, None, :] * bm.numel() / bm.sum()

    def _compute_block_mask(self, mask):
        block_mask = nn.functional.max_pool1d(
            input=mask[:, None, :], kernel_size=self.bs, stride=1, padding=self.bs // 2
        )
        if self.bs % 2 == 0:
            block_mask = block_mask[:, :, :-1]
        return (1 - block_mask.squeeze(1))

dblock = DropBlock(p=0.75, bs=3)

def torch_cdf(x):
    neg_ones = x < 0
    x = x.abs()
    a1 = 0.319381530
    a2 = -0.356563782
    a3 = 1.781477937
    a4 = -1.821255978
    a5 = 1.330274429
    k = 1.0 / (1.0 + 0.2316419 * x)
    k2 = k * k
    k3 = k2 * k
    k4 = k3 * k
    k5 = k4 * k
    c = (a1 * k + a2 * k2 + a3 * k3 + a4 * k4 + a5 * k5)
    phi = 1.0 - c * (-x * x / 2.0).exp() * 0.3989422804014327
    phi[neg_ones] = 1.0 - phi[neg_ones]
    return phi

def calculate_psr(rewards):
    mean, std = rewards.mean(dim=0), rewards.std(dim=0)
    rdiff = rewards - mean
    zscore = rdiff / std
    skew = (zscore**3).mean(dim=0)
    kurto = ((zscore**4).mean(dim=0) - 4) / 4
    sharpe = mean / std
    psr_in = (1 - skew * sharpe + kurto * sharpe**2) / (len(rewards) - 1)
    psr_out = torch_cdf(sharpe / psr_in.sqrt())
    psr_out[psr_out.isnan()] = 0.0
    return mean, std, psr_out

def covv(X):
    D = X.shape[-1]
    mean = torch.mean(X, dim=-1).unsqueeze(-1)
    X = X - mean
    return 1/(D-1) * X @ X.transpose(-1, -2)

def corr(X, eps=1e-08):
    D = X.shape[-1]
    std = torch.std(X, dim=-1).unsqueeze(-1)
    mean = torch.mean(X, dim=-1).unsqueeze(-1)
    X = (X - mean) / (std + eps)
    return 1/(D-1) * X @ X.transpose(-1, -2)

def get_entropy(eigen_vectors, portfolio_weights):
    eigen_vector_weights = eigen_vectors @ portfolio_weights.T
    rba = eigen_vector_weights ** 2
    rba = rba / rba.sum(dim=0, keepdim=True)
    # calculate entropy
    entropy = -torch.nansum(rba * torch.log(rba), dim=0) / math.log(rba.shape[0])
    return entropy

class RobustQDPortfolioEnsemble:
    def __init__(self, data, index, batch_size):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data = torch.tensor(data, dtype=torch.float32).to(self.device)
        #self.data = self.data.pin_memory().cuda(non_blocking=True)
        self.index = torch.tensor(index, dtype=torch.float32).to(self.device)
        self.eigen_vectors = torch.linalg.eigh(covv(self.data.T))[1]
        self.weights = nn.Parameter(torch.zeros((batch_size, self.data.shape[1]), dtype=torch.float32).to(self.device))

    def calculate_reward(self, weights, valid_data, index, eigen_vectors, train = False):
        #weights = weights.softmax(dim=1) if train else entmax15(weights, dim=1)
        weights = entmax15(weights, dim=1) if train else sparsemax(weights, dim=1)
        rets = weights.matmul(valid_data.T - index[:len(valid_data)].to(self.device))
        
        rets = dblock(rets.unsqueeze(1)).squeeze(1) if train else rets
        omg = rets.clamp(min=0.0).mean(dim=1) / rets.abs().mean(dim=1)

        if train:
            matrix = corr(rets).fill_diagonal_(0.0)
            corr_max = matrix.max(dim=1)[0]

            ww = torch.arange(1, len(valid_data)+1).pow(0.5).to(self.device)
            rets = (rets * (ww/ww.sum())).sum(dim=0)

            entropy = get_entropy(eigen_vectors, weights)
            return -(calculate_psr(rets.T)[-1]*omg) * entropy / corr_max

        return -(calculate_psr(rets.T)[-1]*omg)


    def optimize(self, cutoff_index, epochs):
        self.weights.data.zero_()
        optimizer = torch.optim.Rprop([self.weights], lr=1e-3)

        best_loss = float("inf")
        best_epoch = 0
        test_losses = []
        training_entropies = []

        for epoch in range(epochs):
            optimizer.zero_grad()
            train_loss = self.calculate_reward(self.weights, self.data[:cutoff_index], self.index, self.eigen_vectors, train=True).mean()
            train_loss.backward()
            optimizer.step()

            with torch.no_grad():
                test_loss = self.calculate_reward(self.weights.mean(dim=0).unsqueeze(0), self.data[cutoff_index:], self.index, self.eigen_vectors, train=False).mean().item()
                entropy = get_entropy(self.eigen_vectors, sparsemax(self.weights.mean(dim=0))).item()

                if test_loss < best_loss:
                    best_loss = test_loss
                    best_epoch = epoch

                test_losses.append(test_loss)
                training_entropies.append(entropy)

        return self.weights.clone().detach(), best_epoch, test_losses, training_entropies

    def final_portfolio(self, epochs):
        self.weights.data.zero_()
        optimizer = torch.optim.Rprop([self.weights], lr=1e-3)

        best_weights = None
        best_loss = float("inf")

        for epoch in range(epochs):
            optimizer.zero_grad()
            train_loss = self.calculate_reward(self.weights, self.data, self.index, self.eigen_vectors, train=True).mean()
            train_loss.backward()
            optimizer.step()

            if train_loss.item() < best_loss:
                best_loss = train_loss.item()
                best_weights = self.weights.clone().detach()

        return sparsemax(best_weights.mean(dim=0), dim=0).detach().cpu().numpy()