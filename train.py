"""全参数微调；只由验证集选择 checkpoint，训练期间不访问测试集。"""
import argparse, csv, json, os, random, time, sys, platform
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from data.dataset import build_datasets
from models.model import build_model
from utils.metrics import evaluate
from utils.runtime import select_device

def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed); random.seed(seed)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root', default='datasets')
    p.add_argument('--output', default='runs/baseline')
    p.add_argument('--split', default='runs/split.json')
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--label_smoothing', type=float, default=0.0)
    args=p.parse_args()
    out=Path(args.output); out.mkdir(parents=True, exist_ok=True)
    if (out/'history.csv').exists():
        raise FileExistsError('输出目录已有实验，避免混写日志，请更换 --output。')
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark=False
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(4)
    device=select_device()
    datasets, split=build_datasets(args.root,args.split,args.seed)
    generator=torch.Generator().manual_seed(args.seed)
    train=DataLoader(datasets['train'],batch_size=args.batch_size,shuffle=True,
        num_workers=args.workers,worker_init_fn=seed_worker,generator=generator,pin_memory=device.type=='cuda',
        persistent_workers=args.workers>0)
    val=DataLoader(datasets['val'],batch_size=args.batch_size,num_workers=args.workers,
                   persistent_workers=args.workers>0)
    model=build_model().to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=0.01)
    criterion=torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    config=vars(args)|dict(device=str(device),gpu=torch.cuda.get_device_name() if device.type=='cuda' else None,
        torch=str(torch.__version__), python=sys.version, platform=platform.platform(), lr=1e-4, weight_decay=.01,
        split_sizes={k:len(v) for k,v in datasets.items()})
    (out/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    writer=SummaryWriter(str(out/'tensorboard'))
    best=-1; start=time.perf_counter()
    with (out/'history.csv').open('w',newline='',encoding='utf-8') as f:
        fields=['epoch','train_loss','train_ce','train_accuracy','val_loss','val_accuracy','seconds']
        csvwriter=csv.DictWriter(f,fieldnames=fields); csvwriter.writeheader()
        for epoch in range(1,args.epochs+1):
            model.train(); loss_sum=ce_sum=correct=n=0
            for step,(x,y) in enumerate(train):
                if epoch==1 and step==0: print('Batch shape:',list(x.shape),flush=True)
                x,y=x.to(device),y.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits=model(x); loss=criterion(logits,y)
                loss.backward(); optimizer.step()
                n+=len(y); loss_sum+=loss.item()*len(y)
                ce_sum+=torch.nn.functional.cross_entropy(logits.detach(),y,reduction='sum').item()
                correct+=logits.detach().argmax(1).eq(y).sum().item()
            metrics,_,_=evaluate(model,val,device)
            row=dict(epoch=epoch,train_loss=loss_sum/n,train_ce=ce_sum/n,train_accuracy=100*correct/n,
                     val_loss=metrics['loss'],val_accuracy=metrics['top1'],seconds=time.perf_counter()-start)
            csvwriter.writerow(row); f.flush()
            for k,v in row.items():
                if k not in ('epoch','seconds'): writer.add_scalar(k,v,epoch)
            writer.flush(); print(json.dumps(row),flush=True)
            if metrics['top1']>best:
                best=metrics['top1']
                torch.save(dict(model=model.state_dict(),epoch=epoch,val_metrics=metrics,
                                config=config,classes=split['classes']),out/'best_model.pth')
    writer.close()

if __name__=='__main__': main()
