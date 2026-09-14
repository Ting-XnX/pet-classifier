"""对预测类别计算 Grad-CAM；前向捕获特征并直接求梯度，避免原地 ReLU 的 backward hook 冲突。"""
import torch
import numpy as np
from data.dataset import MEAN, STD

def gradcam(model, x, target=None):
    model.eval()
    features=[]
    handle=model.layer4.register_forward_hook(lambda module, inputs, output: features.append(output))
    try:
        logits=model(x)
        target=int(logits.argmax(1).item()) if target is None else target
        grads=torch.autograd.grad(logits[0,target],features[0])[0]
        weights=grads.mean(dim=(2,3),keepdim=True)
        cam=(weights*features[0]).sum(1,keepdim=True).relu()
        cam=torch.nn.functional.interpolate(cam,size=x.shape[-2:],mode='bilinear',align_corners=False)[0,0]
        cam=cam.detach().cpu().numpy()
        cam=(cam-cam.min())/(cam.max()-cam.min()+1e-8)
        rgb=x[0].detach().cpu().permute(1,2,0).numpy()*np.array(STD)+np.array(MEAN)
        return np.clip(rgb,0,1),cam,target
    finally:
        handle.remove()

def save_cam(model, x, truth, classes, path):
    import matplotlib.pyplot as plt
    rgb,cam,pred=gradcam(model,x)
    fig,axes=plt.subplots(1,2,figsize=(7,3.5))
    axes[0].imshow(rgb); axes[0].set_title('Input (center crop)')
    axes[1].imshow(rgb); axes[1].imshow(cam,cmap='jet',alpha=.42,vmin=0,vmax=1)
    axes[1].set_title('Grad-CAM: predicted class')
    for ax in axes: ax.axis('off')
    fig.suptitle(f'True: {classes[truth]} | Pred: {classes[pred]}',fontsize=10)
    fig.tight_layout(); fig.savefig(path,dpi=180); plt.close(fig)

