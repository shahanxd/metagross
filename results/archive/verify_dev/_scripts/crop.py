import sys
from PIL import Image
src, out, x0, y0, x1, y1 = sys.argv[1], sys.argv[2], *map(int, sys.argv[3:7])
Image.open(src).crop((x0, y0, x1, y1)).save(out)
