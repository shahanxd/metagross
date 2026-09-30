import urllib.request, io, zipfile, sys, collections
class HttpFile(io.RawIOBase):
    def __init__(s,url):
        s.url=url; s.pos=0
        r=urllib.request.urlopen(urllib.request.Request(url,method='HEAD'))
        s.size=int(r.headers['Content-Length']); s.url=r.geturl()
    def seekable(s): return True
    def readable(s): return True
    def tell(s): return s.pos
    def seek(s,o,w=0):
        s.pos = o if w==0 else (s.pos+o if w==1 else s.size+o); return s.pos
    def read(s,n=-1):
        if n<0: n=s.size-s.pos
        if n==0: return b''
        req=urllib.request.Request(s.url,headers={'Range':'bytes=%d-%d'%(s.pos,s.pos+n-1)})
        d=urllib.request.urlopen(req).read(); s.pos+=len(d); return d
    def readinto(s,b):
        d=s.read(len(b)); b[:len(d)]=d; return len(d)
url=sys.argv[1]
z=zipfile.ZipFile(HttpFile(url))
names=z.namelist()
print('entries',len(names))
cnt=collections.Counter('/'.join(n.split('/')[:-1]) for n in names)
for k,v in sorted(cnt.items()): print(v,k)
print([n for n in names if n.endswith('.txt')][:20])
