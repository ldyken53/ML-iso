#!/usr/bin/env python3

import os

from config import *
from util import *
from dataset import *
from image import *

def crop(img, cropx, cropy):

    y, x, _ = img.shape
    cropx = x - int(x*cropx/100)
    cropy = y - int(y*cropy/100)
    startx = x//2 - cropx//2
    starty = y//2 - cropy//2

    return img[starty:starty+cropy, startx:startx+cropx]


def main():
    # Parse the command line arguments
    cfg = parse_args(
        description='Creates a temporal image from mutliple single frames by stacking them along the y axis.')

    images = list(os.listdir(cfg.input))
    #groups = set([image.split('_')[-1].split('.')[0] for image in images])
    groups = set(['001spp', 'reference'])
    #features = set([image.split('.')[-2] for image in images])
    features = set([image.split('.')[0].split('-')[-1] for image in images])

    stride = 8
    img_per_stride = 2


#   for group in groups:
#     #group_images = [image for image in images if image.split('_')[-1].split('.')[0] == group]
#     group_images = images

    for feature in features:
        #feature_images = [image for image in group_images if image.split('.')[-2] == feature]
        feature_images = [image for image in images if image.split(
            '.')[0].split('-')[-1] == feature]
        # feature_images.sort(key = lambda x: int(x.split('-')[-1].split('_')[0])) #TODO: This ordering is data naming specific
        # TODO: This ordering is data naming specific
        feature_images.sort(key=lambda x: int(x.split('t')[2].split('-')[0]))

        i = 0
        while i*cfg.num_frames < len(feature_images):

            for j in range(img_per_stride):
                feat = 'ldr' if feature == 'rgba' else 'nrm'
                outfile_ref = os.path.join(
                    cfg.output, f'seq_{i*stride+j}_ref.{feat}.exr')
                outfile_001spp = os.path.join(
                    cfg.output, f'seq_{i*stride+j}_001spp.{feat}.exr')
                print('Preparing ' + outfile_ref + ' and ' + outfile_001spp)
                outimages = []

                for image in feature_images[i*cfg.num_frames*stride: (i+1)*cfg.num_frames*stride: stride]:
                    # ingroup = image.split('_')[-1].split('.')[0]

                    # if ingroup != group:
                    #     continue

                    infile = os.path.join(cfg.input, image)
                    img = load_image(infile, num_channels=3)
                    #print('Adding ' + infile)
                    outimages.append(img)

                
                if len(outimages) < cfg.num_frames:
                    continue

                try:
                    outimage = np.concatenate(outimages, axis=0)
                    save_image(outfile_ref, outimage)
                    save_image(outfile_001spp, outimage)
                except:
                    print('Skipping ' + outfile_ref + ' due to errors.')

            i += 1


if __name__ == '__main__':
    main()
