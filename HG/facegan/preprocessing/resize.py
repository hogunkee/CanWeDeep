import os
from PIL import Image

#DIR = 'after'
OUT = 'resize2'
SIZE = 128

def resize(DIR):
    os.system('mkdir %s'%OUT)
    os.system('mkdir %s'%os.path.join(OUT,DIR))
    files = os.listdir(DIR)
    for file in files:
        if 'png' in file or 'jpg' in file or 'PNG' in file:
            im = Image.open(os.path.join(DIR, file))
            x, y = im.size
            if x > y:
                resized_img = im.resize((SIZE, int(SIZE*y/x)), Image.ANTIALIAS)
                pos = (0, int(SIZE/2*(1-y/x)))
            else:
                resized_img = im.resize((int(SIZE*x/y), SIZE), Image.ANTIALIAS)
                pos = (int(SIZE/2*(1-x/y)), 0)

            new_im = Image.new('RGB', (SIZE, SIZE))
            new_im.paste(resized_img, pos)
            save_name = os.path.join(os.path.join(OUT, DIR), file)
            print(save_name)
            new_im.save(save_name)

resize('before')
resize('after')
