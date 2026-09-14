"""固定分层划分；为每个子集独立设置变换。"""
import json
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision.datasets import OxfordIIITPet
from torchvision import transforms as T

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

class PetSubset(Dataset):
    def __init__(self, datasets, indices, training=False):
        self.datasets, self.indices = datasets, indices
        self.transform = T.Compose([T.Resize(256),
            T.RandomCrop(224) if training else T.CenterCrop(224),
            *([T.RandomHorizontalFlip()] if training else []),
            T.ToTensor(), T.Normalize(MEAN, STD)])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        i = self.indices[index]
        a, b = self.datasets
        image, target = a[i] if i < len(a) else b[i-len(a)]
        return self.transform(image), target

def build_datasets(root: str, split_path: str, seed: int = 42):
    datasets = [OxfordIIITPet(root, split=s, target_types='category', download=True)
                for s in ('trainval', 'test')]
    labels = np.array(datasets[0]._labels + datasets[1]._labels)
    ids = np.arange(len(labels))
    path = Path(split_path)
    train, remainder = train_test_split(ids, test_size=0.30, stratify=labels, random_state=seed)
    val, test = train_test_split(remainder, test_size=0.50,
                               stratify=labels[remainder], random_state=seed)
    expected = dict(seed=seed, total=len(ids), classes=datasets[0].classes,
                    train=train.tolist(), val=val.tolist(), test=test.tolist())
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8')) != expected:
            raise ValueError('划分文件与当前种子/数据不一致，请使用独立输出目录。')
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(expected, indent=2), encoding='utf-8')
    assert not (set(train) & set(val) or set(train) & set(test) or set(val) & set(test))
    assert len(train)+len(val)+len(test) == len(ids)
    for split in (train, val, test):
        assert len(set(labels[split])) == 37
    return {k: PetSubset(datasets, expected[k], k == 'train') for k in ('train','val','test')}, expected
