import sys, random, collections, io
sys.argv=[sys.argv[0]]+sys.argv[1:]
exec(open('ziplist.py').read().split("url=sys.argv[1]")[0])
url=sys.argv[1]
z=zipfile.ZipFile(HttpFile(url))
names=[n for n in z.namelist() if not n.endswith('/')]
random.seed(0)
print(random.sample(names,12))
pref=collections.Counter(n.split('/')[-1].split('_')[0][:12] for n in names)
print(pref.most_common(25))
# read one image and one mask for resolution
try:
    from PIL import Image
except Exception as e:
    print('no PIL'); sys.exit()
imgs=[n for n in names if '/images/' in n][:1]; msk=[n for n in names if '/masks' in n][:1]
for n in imgs+msk:
    im=Image.open(io.BytesIO(z.read(n))); import numpy as np; a=np.array(im); print(n, im.size, im.mode, (np.unique(a) if a.ndim==2 else ''))
