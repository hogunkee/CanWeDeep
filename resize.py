import os
from PIL import Image

#DIR = 'after'
OUT = 'korea'
SIZE = 128

def resize(DIR1, DIR2, OUT='korea', SIZE=64):
    os.system('mkdir %s'%OUT)
    afiles = sorted(os.listdir(DIR2))
    bfiles = sorted(os.listdir(DIR1))

    assert len(afiles)==len(bfiles)
    for i in range(len(afiles)):
        afile = afiles[i]
        bfile = bfiles[i]
        if 'png' in bfile or 'jpg' in bfile or 'PNG' in bfile:
            aim = Image.open(os.path.join(DIR2, afile))
            bim = Image.open(os.path.join(DIR1, bfile))
            x, y = bim.size
            if x > y:
                a_resized_img = aim.resize((SIZE, int(SIZE*y/x)), Image.ANTIALIAS)
                b_resized_img = bim.resize((SIZE, int(SIZE*y/x)), Image.ANTIALIAS)
                a_pos = (SIZE, int(SIZE/2*(1-y/x)))
                b_pos = (0, int(SIZE/2*(1-y/x)))
            else:
                a_resized_img = aim.resize((int(SIZE*x/y), SIZE), Image.ANTIALIAS)
                b_resized_img = bim.resize((int(SIZE*x/y), SIZE), Image.ANTIALIAS)
                a_pos = (SIZE + int(SIZE/2*(1-x/y)), 0)
                b_pos = (int(SIZE/2*(1-x/y)), 0)

            new_im = Image.new('RGB', (2*SIZE, SIZE))
            new_im.paste(a_resized_img, a_pos)
            new_im.paste(b_resized_img, b_pos)
            save_name = os.path.join(OUT, afile)
            print(save_name)
            new_im.save(save_name)

resize('crop_before', 'crop_after', 'korea128', 128)
#resize('crop_before', 'crop_after')
