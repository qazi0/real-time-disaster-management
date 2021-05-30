import glob
import os
import sys

split = sys.argv[1]

images = [os.path.basename(x) for x in sorted(glob.glob(f'{split}/images/*.jpg'))] 

with open(f'{split}-min.txt','w') as f:
	for image in images:
		print(f'Processing image: {image}')
		f.write(f'data/custom/{split}/images/{image}\n')




