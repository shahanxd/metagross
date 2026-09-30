from PIL import Image; import numpy as np, collections
from pathlib import Path
for s in ['val','test','train']:
    ims=sorted(Path(s,'image_res').glob('*'))
    ms=sorted(Path(s,'labelids_res').glob('*'))
    for p in ims[:2]+ms[:2]:
        im=Image.open(p); a=np.array(im); print(p, im.size, im.mode, a.shape, np.unique(a) if a.ndim==2 else '')
    cnt=np.zeros(256,np.int64); sizes=collections.Counter()
    for p in ms[::15]:
        a=np.array(Image.open(p)); sizes[a.shape]+=1; cnt+=np.bincount(a.ravel(),minlength=256)
    tot=cnt.sum(); print(s, {i:round(c/tot,4) for i,c in enumerate(cnt) if c}, sizes)
    print(s, collections.Counter(p.stem.rsplit('_',1)[0] for p in ims).most_common(30))
