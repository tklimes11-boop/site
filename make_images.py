# Generates the 3 illustrative raster maps (WebP, 4 widths each) – synthetic terrain, not real data.
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
W,H=1600,1000
def noise(scale,seed):
    r=np.random.default_rng(seed).standard_normal((H,W))
    k=np.hypot(np.fft.fftfreq(H)[:,None],np.fft.fftfreq(W)[None,:])
    z=np.fft.ifft2(np.fft.fft2(r)*np.exp(-(k*scale)**2)).real
    return (z-z.min())/np.ptp(z)
yy,xx=np.mgrid[0:H,0:W]
z=.85*noise(95,3)+.15*noise(40,5)+.45*xx/W; z=(z-z.min())/np.ptp(z)
gy,gx=np.gradient(z); slope=np.hypot(gx,gy)
rng=np.random.default_rng(11)
PAPER=(232,236,227); INK=(30,44,58); MARK=(179,57,47); WATER=(95,143,174)
sx=lambda y:int(W*.58+110*np.sin(y/150)+40*np.sin(y/57))
def relief():
    base=np.array((196,211,188))+(np.array((226,222,196))-np.array((196,211,188)))*z[...,None]
    return base*np.clip(1+55*(gx+gy),.82,1.12)[...,None]
def contours(a):
    q=np.floor(z*11); m=np.zeros((H,W),bool)
    m[1:]|=q[1:]!=q[:-1]; m[:,1:]|=q[:,1:]!=q[:,:-1]
    a[m]=a[m]*.3+np.array((138,106,68))*.7
def parcels(a):
    sw,sh=400,250
    pts=np.array([(x,y) for x in np.linspace(0,sw,12) for y in np.linspace(0,sh,8)])+rng.uniform(-14,14,(96,2))
    gy_,gx_=np.mgrid[0:sh,0:sw]
    ids=np.argmin((gx_[...,None]-pts[:,0])**2+(gy_[...,None]-pts[:,1])**2,axis=2)
    ids=np.repeat(np.repeat(ids,4,0),4,1)
    tints=np.array([(222,214,168),(190,205,160),(214,222,190),(205,196,150)])[rng.integers(0,4,96)]
    a[:]=a*.6+tints[ids]*.4
    b=np.zeros((H,W),bool); b[1:]|=ids[1:]!=ids[:-1]; b[:,1:]|=ids[:,1:]!=ids[:,:-1]
    a[b]=(120,128,104)
def erosion(a):
    m=Image.fromarray(((slope>np.percentile(slope,82))*255).astype('uint8')).filter(ImageFilter.GaussianBlur(7))
    al=(np.asarray(m)/255*.2)[...,None]; a[:]=a*(1-al)+np.array(MARK)*al
def mark(d,pts):
    d.line(pts,fill=PAPER,width=13,joint="curve"); d.line(pts,fill=MARK,width=7,joint="curve")
    for p in (pts[0],pts[-1]):
        d.ellipse([p[0]-3.5,p[1]-3.5,p[0]+3.5,p[1]+3.5],fill=MARK)
def seg(cx,cy,ang,l=55): return [(cx-l*np.cos(ang),cy-l*np.sin(ang)),(cx+l*np.cos(ang),cy+l*np.sin(ang))]
def pick(n,gap,avoid_stream=False):
    ys,xs=np.mgrid[70:H-70:10,70:W-70:10]; out=[]
    for i in np.argsort(-slope[ys,xs].ravel()):
        y,x=ys.ravel()[i],xs.ravel()[i]
        if avoid_stream and abs(x-sx(y))<95: continue
        if all(np.hypot(x-a,y-b)>gap for a,b,_ in out): out.append((x,y,np.arctan2(gx[y,x],-gy[y,x])))
        if len(out)==n: break
    return out
def stream(d): d.line([(sx(y),y) for y in range(0,H,8)],fill=WATER,width=8,joint="curve")
def finish(img,name):
    d=ImageDraw.Draw(img); d.rectangle([0,0,W-1,H-1],outline=INK,width=3)
    for w in (480,800,1200,1600):
        img.resize((w,int(w*H/W)),Image.LANCZOS).save(f"img/{name}-{w}.webp",quality=74,method=6)
# level 1 – no context: blank graticule, arbitrary placement
a=np.zeros((H,W,3))+PAPER; a[::100]=(214,220,206); a[:,::100]=(214,220,206)
img=Image.fromarray(a.astype('uint8')); d=ImageDraw.Draw(img)
for _ in range(9): mark(d,seg(rng.uniform(120,W-120),rng.uniform(120,H-120),rng.uniform(0,np.pi)))
finish(img,"uroven-1")
# level 2 – terrain
a=relief(); contours(a); img=Image.fromarray(a.clip(0,255).astype('uint8')); d=ImageDraw.Draw(img); stream(d)
for x,y,t in pick(8,190): mark(d,seg(x,y,t))
finish(img,"uroven-2")
# level 3 – terrain + parcels + erosion + stream + riparian buffer
a=relief(); parcels(a); contours(a); erosion(a); img=Image.fromarray(a.clip(0,255).astype('uint8')); d=ImageDraw.Draw(img); stream(d)
for x,y,t in pick(7,210,True): mark(d,seg(x,y,t))
for y0,y1 in ((90,330),(520,780)): mark(d,[(sx(y)+36,y) for y in range(y0,y1,10)])
finish(img,"uroven-3")
