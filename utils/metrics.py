import numpy as np
import torch
from sklearn.metrics import f1_score, confusion_matrix

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss, labels, predictions, top5 = 0.0, [], [], 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += torch.nn.functional.cross_entropy(logits, y, reduction='sum').item()
        top5 += logits.topk(5, dim=1).indices.eq(y[:, None]).any(dim=1).sum().item()
        labels.extend(y.cpu().tolist())
        predictions.extend(logits.argmax(1).cpu().tolist())
    return dict(loss=total_loss/len(labels), top1=float(np.mean(np.array(labels)==predictions)*100),
                top5=100*top5/len(labels), macro_f1=float(f1_score(labels, predictions,
                labels=list(range(37)), average='macro', zero_division=0))), labels, predictions

def plot_confusion(labels, predictions, classes, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cm = confusion_matrix(labels, predictions, labels=list(range(37)))
    fig, ax = plt.subplots(figsize=(14,12))
    im=ax.imshow(cm / np.maximum(cm.sum(1, keepdims=True), 1), cmap='Blues', vmin=0,vmax=1)
    ax.set_xticks(range(37), classes, rotation=90, fontsize=7)
    ax.set_yticks(range(37), classes, fontsize=7)
    ax.set(xlabel='Predicted breed', ylabel='True breed', title='Test confusion matrix (row normalized)')
    fig.colorbar(im, ax=ax, fraction=.04)
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)
    pairs=sorted([(int(cm[i,j]+cm[j,i]), classes[i], classes[j],int(cm[i,j]),int(cm[j,i]))
                  for i in range(37) for j in range(i+1,37)], reverse=True)
    return cm, pairs[:3]
