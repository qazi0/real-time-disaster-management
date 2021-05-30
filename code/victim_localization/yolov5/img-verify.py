from os import listdir
from PIL import Image
import sys

path = sys.argv[1]
for filename in listdir(path):
  if filename.endswith('.jpg'):
    try:
      img = Image.open(path+filename) # open the image file
      img.verify() # verify that it is, in fact an image
      print('Image ',filename,' verified.')
    except (IOError, SyntaxError) as e:
      print('Bad file:', filename) # print out the names of corrupt files