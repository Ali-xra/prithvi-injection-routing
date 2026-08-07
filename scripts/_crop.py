# -*- coding: utf-8 -*-
import io, os, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
from PIL import Image

S = r"C:\Users\aliso\Desktop\proje\_shots"
D = os.path.join(S, "crop")
os.makedirs(D, exist_ok=True)

for name, top, h, sc in [("docsindex_top.png", 0, 820, .58),
                         ("playbook_top.png", 0, 700, .58)]:
    im = Image.open(os.path.join(S, name))
    w, H = im.size
    c = im.crop((0, top, w, min(top + h, H)))
    c = c.resize((int(c.size[0] * sc), int(c.size[1] * sc)))
    out = os.path.join(D, name)
    c.save(out)
    print(name, im.size, "->", c.size)
