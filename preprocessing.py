 .gimport glob
from PIL import Image
import numpy as np 
import os

files = [f for f in glob.glob('label/*/*/*/*')]
'''
files = []
for _,_,c in os.walk('label/'):
    files.append(c)
'''   
print(len(files))

for file in files:
    img = np.asarray(Image.open(files[0]))
    img[img>0] = 255