import argparse, json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import numpy as np
import torch
from torch.utils.data import DataLoader
from data.dataset import build_datasets
from models.model import build_model
from utils.metrics import evaluate, plot_confusion
from utils.gradcam import save_cam
from utils.runtime import select_device

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--root',default='datasets')
    p.add_argument('--split',default='runs/split.json')
    p.add_argument('--workers',type=int,default=0)
    args=p.parse_args()
    torch.set_num_threads(4)
    device=select_device()
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    data,split=build_datasets(args.root,args.split,checkpoint['config']['seed'])
    model=build_model(False).to(device); model.load_state_dict(checkpoint['model'])
    metrics,truth,pred=evaluate(model,DataLoader(data['test'],batch_size=32,num_workers=args.workers),device)
    out=Path(args.checkpoint).parent/'evaluation'; out.mkdir(exist_ok=True)
    classes=split['classes']
    cm,pairs=plot_confusion(truth,pred,classes,out/'confusion_matrix.png')
    np.savetxt(out/'confusion_matrix.csv',cm,fmt='%d',delimiter=',')
    cases={}
    for kind,want_correct in [('correct',True),('wrong',False)]:
        index=next((i for i,(a,b) in enumerate(zip(truth,pred)) if (a==b)==want_correct),None)
        if index is not None:
            x,y=data['test'][index]
            save_cam(model,x.unsqueeze(0).to(device),y,classes,out/f'gradcam_{kind}.png')
            cases[kind]=dict(test_index=index,global_index=split['test'][index],true=classes[y],pred=classes[pred[index]])
    result=dict(metrics=metrics,checkpoint_epoch=checkpoint['epoch'],val_metrics=checkpoint['val_metrics'],
                confusion_pairs=pairs,cases=cases,n_test=len(truth))
    (out/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    (out/'predictions.json').write_text(json.dumps(dict(truth=truth,pred=pred)),encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__': main()
