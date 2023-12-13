import numpy as np
import os

from image import *

masks = None

def GenerateMask(noise_type, size):
    if noise_type == 'uniform':
        return np.random.uniform(0.0, 1.0, (size,size))

def GenerateMasks(noise_type, num_masks=64, sizes=[16,32,64,128,256]):
    print('Generating Masks')
    if noise_type == 'blue' or noise_type == 'stbn':
        LoadMasks(noise_type)
        return

    global masks
    masks = {}
    for size in sizes:
        masks[str(size)] = []
        for _ in range(num_masks):
            masks[str(size)].append(GenerateMask(noise_type, size))


def LoadMasks(noise_type):
    print('Loading Masks')
    global masks
    masks = {}
    #texture_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../data/noise_textures/')
    texture_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), f'../data/noise_textures/{noise_type}/')

    for file_name in os.listdir(texture_dir):
        file_name_path = os.path.join(texture_dir, file_name)
        if not os.path.isdir(file_name_path):
            continue

        print('Loading dir', file_name_path)
        masks[file_name] = []

        imgfile_names = os.listdir(file_name_path)
        imgfile_names = sorted(imgfile_names, key=lambda d: int(d.split('_')[-1].split('.')[0]))

        for imgfile_name in imgfile_names:
            imgfile_name = os.path.join(file_name_path, imgfile_name)
            try:
                img = load_image(imgfile_name, num_channels=1)
                img_array = np.reshape(img, (int(file_name), int(file_name)))
                masks[file_name].append(img_array)
            except:
                print('Skipping file', imgfile_name)


def GetGauss(size, sigma=1.85, muu=0.0):
    x, y = np.meshgrid(np.linspace(-3*sigma,3*sigma,size), np.linspace(-3*sigma,3*sigma,size)) 
    dst = np.sqrt(x*x+y*y) 
    
    # Calculating Gaussian array 
    gauss = np.exp(-( (dst-muu)**2 / ( 2.0 * sigma**2 ) ) ) 

    return gauss

def GetFoveatedMask(shape, focus, r):
    gauss = GetGauss(r*2)

    # print("Shape", shape)
    # print("RShape", (shape[0]+2*r, shape[1]+2*r))

    fovmask = np.zeros((shape[0]+2*r, shape[1]+2*r))

    try:
        fovmask[focus[0]:focus[0]+2*r, focus[1]:focus[1]+2*r] = gauss
    except:
        print('Error', gauss.shape, fovmask.shape, focus, r)

    return fovmask[r:shape[0]+r, r:shape[1]+r]


def GetSampleMask(shape, focus, r, size=32, threshold=.995, maskIdx=0):
    #noise = bn.GetVoidAndClusterBlueNoise((size,size), 1.5)
    noise = masks[str(size)][maskIdx%len(masks[str(size)])]

    mask = np.tile(noise, [(shape[0] // size) + 1, (shape[1] // size) + 1])
    mask = mask[:shape[0], :shape[1]]
    mask = mask / np.max(mask)

    fovmask = GetFoveatedMask(shape, focus, r)
    fovmask = fovmask / np.max(fovmask)

    mask[mask+fovmask >= threshold] = 1
    mask[mask+fovmask < threshold] = 0

    return mask

def GetSampleMaskSequence(shape, r, num=12, threshold=.995, mask_size=64, noise_type='blue'):
    if masks == None:
        GenerateMasks(noise_type)

    # Define random start and end points
    # Add foveated area radius r as offset to avoid invalid locations
    start = np.array([np.random.randint(0, shape[0]), np.random.randint(0, shape[1])], dtype=np.int)
    end = np.array([np.random.randint(0, shape[0]), np.random.randint(0, shape[1])], dtype=np.int)

    # Calculate their distance
    dist = np.abs(start - end)

    # Generate sequence of 30 frames
    frames = np.empty((num,shape[0],shape[1]))
    for i, point in enumerate(np.linspace(start, end, num=num, dtype=np.int)):
        #point = (shape[0],shape[1])
        # if i == 14:
        #     point = (800, 175)
        # print('POINT', point)
        frames[i, :, :] = GetSampleMask(shape=shape, focus=point, r=r, size=mask_size, threshold=threshold, maskIdx=i)

        # if i == 14:
        #     img = np.expand_dims(frames[i, :, :], 2)
        #     img = np.repeat(img, 3, 2)
        #     save_image('mask.png', img)

    return frames
