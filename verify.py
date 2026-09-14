"""关键科研约束的轻量检查；不把随机输入当作实验数据。"""
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms as T
from data.dataset import build_datasets
from models.model import build_model
from utils.gradcam import gradcam
from utils.metrics import evaluate

def main():
    torch.set_num_threads(4)
    data,split=build_datasets('datasets','runs/split.json')
    for key in ('val','test'):
        assert not any(isinstance(t,(T.RandomCrop,T.RandomHorizontalFlip)) for t in data[key].transform.transforms)
        assert torch.equal(data[key][0][0],data[key][0][0])
    assert data['train'][0][0].shape==(3,224,224)
    model=build_model(False)
    shapes=[]
    hook=model.layer4.register_forward_hook(lambda m,i,o:shapes.append(list(o.shape)))
    model.eval()
    with torch.no_grad():
        output=model(torch.randn(2,3,224,224))
    hook.remove()
    assert list(output.shape)==[2,37] and shapes==[[2,512,7,7]]
    rgb,cam,target=gradcam(model,torch.randn(1,3,224,224))
    assert cam.shape==(224,224) and bool(torch.isfinite(torch.from_numpy(cam)).all())
    class Identity(torch.nn.Module):
        def forward(self,x):return x
    logits=torch.eye(37)*10
    m,_,_=evaluate(Identity(),DataLoader(TensorDataset(logits,torch.arange(37)),batch_size=8),'cpu')
    assert m['top1']==100 and m['top5']==100 and m['macro_f1']==1
    result=dict(status='passed',split_sizes={k:len(v) for k,v in data.items()},
                checks=['disjoint stratified split','deterministic evaluation transforms',
                        'ResNet feature and logits dimensions','finite Grad-CAM','known perfect predictions'])
    Path('verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(result)

if __name__=='__main__':main()
