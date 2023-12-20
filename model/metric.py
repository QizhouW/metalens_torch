import torch
import torch.nn.functional as F

def mae(output, target):
    return F.l1_loss(output, target)

def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)

def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)

def r2_score(output, target):
    ss_res = torch.sum((target - output) ** 2)
    ss_tot = torch.sum((target - torch.mean(target)) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2

