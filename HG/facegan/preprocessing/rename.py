import os

count = 1

files = sorted(os.listdir())
for f in files:
    if '.PNG' in f:
        '''
        new_f = '%.3d.PNG'%count
        count += 1
        '''
        num = int(f.split('.')[0])
        new_f = str(num-1) + '.PNG'
        os.rename(f, new_f)
