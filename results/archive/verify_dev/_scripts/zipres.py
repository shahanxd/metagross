import sys, io
exec(open('ziplist.py').read().split("url=sys.argv[1]")[0])
from PIL import Image
z=zipfile.ZipFile(HttpFile(sys.argv[1]))
names=[n for n in z.namelist() if not n.endswith('/')]
picks=[]
for key in sys.argv[2:]:
    picks += [n for n in names if key in n][:1]
for n in picks:
    im=Image.open(io.BytesIO(z.read(n)))
    ex = sorted(set(im.getdata())) if im.mode in ('L','P') else ''
    print(n, im.size, im.mode, ex if len(str(ex))<200 else str(ex)[:200])
