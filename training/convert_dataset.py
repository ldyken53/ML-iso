#!/usr/bin/env python3

import os

from config import *
from util import *
from dataset import *
from image import *

def crop(img, cropx, cropy):

    y,x,_ = img.shape
    cropx = x - int(x*cropx/100)
    cropy = y - int(y*cropy/100)
    startx = x//2 - cropx//2
    starty = y//2 - cropy//2

    return img[starty:starty+cropy, startx:startx+cropx]

def main():
  # Parse the command line arguments
  cfg = parse_args(description='Converts a set of feature image to a different image format.')

  
  for file in os.listdir(cfg.input):
    if os.path.isdir(file) or file[-len(cfg.format):] != cfg.format:
        continue

    infile = os.path.join(cfg.input, file)
    outfile = os.path.join(cfg.output, file)
    outfile = os.path.splitext(outfile)[0]
    if 'color' in outfile:
        outfile = outfile.replace('-color', f'_{cfg.suffix}.ldr.exr')
    elif 'gradient' in outfile:
        outfile = outfile.replace('-gradient', f'_{cfg.suffix}.nrm.exr')
    else:
        continue
        #outfile += f'_{cfg.suffix}.ldr.exr'

    print(f'Writing file {outfile}')

    # Load the input image
    image = load_image(infile, num_channels=3)
    image = crop(image, cfg.cropx, cfg.cropy)


    # Load metadata for the image if it exists
    tonemap_exposure = cfg.exposure
    metadata = load_image_metadata(infile)
    if metadata:
        tonemap_exposure = metadata['exposure']

    # Convert the image to tensor
    image = image_to_tensor(image, batch=True)

    # Transform the image
    input_feature  = get_image_feature(infile)
    output_feature = get_image_feature(outfile)
    image = transform_feature(image, input_feature, output_feature, tonemap_exposure)

    # Save the image
    save_image(outfile, tensor_to_image(image))

if __name__ == '__main__':
  main()