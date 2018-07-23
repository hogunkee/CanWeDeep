import os

count = 1

files = os.listdir()
for f in files:
    if '.png' in f and '_' in f:
        '''
        new_f = '%.3d.PNG'%count
        count += 1
        '''
        num = int(f.split('_')[1].split('.')[0])
        new_f = str(num) + '.PNG'
        #new_f = str(num+1) + '.PNG'
        os.rename(f, new_f)
