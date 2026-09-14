"""一键执行两个预先确定的实验，再进行测试集评估。"""
import os, subprocess, sys
from pathlib import Path

if __name__=='__main__':
    os.chdir(Path(__file__).resolve().parent)
    os.environ.setdefault('TORCH_HOME',str(Path('cache').resolve()))
    for name,smoothing in [('baseline','0'),('label_smoothing','0.1')]:
        subprocess.run([sys.executable,'train.py','--output',f'runs/{name}',
                        '--label_smoothing',smoothing],check=True)
    for name in ('baseline','label_smoothing'):
        subprocess.run([sys.executable,'evaluate.py','--checkpoint',f'runs/{name}/best_model.pth'],check=True)
    subprocess.run([sys.executable,'make_report.py'],check=True)
