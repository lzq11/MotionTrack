import configparser
import os
import shutil
import numpy as np

root = "/home/lzq/Doc/Research/Dataset/MOT/jmt/images/"
dst = "/home/lzq/Doc/Research/JMT"

splits = ["train","test1"]
# splits = ["train","val","test1"]

os.makedirs(dst, exist_ok=True)
os.makedirs(os.path.join(dst,'images'), exist_ok=True)
os.makedirs(os.path.join(dst,'labels'), exist_ok=True)

for split in splits:
    data_dir = root + split
    seq_names = [s for s in sorted(os.listdir(data_dir))]
    os.makedirs(os.path.join(dst,'images',split), exist_ok=True)
    os.makedirs(os.path.join(dst,'labels',split), exist_ok=True)
    for seq_name in seq_names:
        
        seq_dir = os.path.join(data_dir,seq_name,seq_name)
        seqinfoini=configparser.ConfigParser()
        seqinfoini.read(os.path.join(data_dir,seq_name,"seqinfo.ini")) 
        seq_width = int(seqinfoini.get("Sequence","imWidth"))
        seq_height = int(seqinfoini.get("Sequence","imHeight"))

        img_names = [s for s in sorted(os.listdir(seq_dir))]

        for img_name in img_names:
            img_dir = os.path.join(seq_dir,img_name)
            dst_dir = os.path.join(dst,'images',split)
            shutil.copy(img_dir,dst_dir,follow_symlinks=True)
        
        if split in ["train","val"]:
            det_dir = os.path.join(data_dir, seq_name,'det','det.txt')
            det = np.loadtxt(det_dir, dtype=np.float64, delimiter=',')
            for fid, tid, x, y, w, h, score, cls, occ in det:
                fid = int(fid)
                xc = (x + w / 2)/seq_width
                yc = (y + h / 2)/seq_height
                w = w/seq_width
                h = h/seq_height
                label_fpath = os.path.join(dst,"labels",split,'out{}_{:04d}.txt'.format(int(seq_name),fid))
                label_str = '{:d} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(int(cls)-1,xc,yc,w,h)
                with open(label_fpath, 'a') as f:
                    f.write(label_str)





