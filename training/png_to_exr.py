import OpenEXR
import os
from PIL import Image
import array
import sys
import matplotlib.pyplot as plt
import numpy as np

def png_to_openexr(input_png_path, output_exr_path):
    # Open the PNG image using PIL
    png_image = Image.open(input_png_path)

    # Extract image data
    width, height = png_image.size
    pixels = list(png_image.getdata())

    # Convert the pixel values to a flat float array for each channel
    # r_channel = array.array('f', [pixel[0] / 255.0 for pixel in pixels])
    # g_channel = array.array('f', [pixel[1] / 255.0 for pixel in pixels])
    # b_channel = array.array('f', [pixel[2] / 255.0 for pixel in pixels])
    gamma = 2.2  # Adjust the gamma value if needed
    r_channel = array.array('f', [(pixel[0] / 255.0)**gamma for pixel in pixels])
    g_channel = array.array('f', [(pixel[1] / 255.0)**gamma for pixel in pixels])
    b_channel = array.array('f', [(pixel[2] / 255.0)**gamma for pixel in pixels])

    # Create an OpenEXR output file
    exr_file = OpenEXR.OutputFile(output_exr_path, OpenEXR.Header(width, height))

    # Write the pixel data to the OpenEXR file for each channel
    exr_file.writePixels({'R': r_channel, 'G': g_channel, 'B': b_channel})

    # Close the OpenEXR file
    exr_file.close()

def traverse_directory(directory_path):
    # Ensure the provided path is a directory
    if not os.path.isdir(directory_path):
        print(f"Error: {directory_path} is not a valid directory.")
        return

    # Traverse through the directory
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            # Check if the file has a PNG extension
            if file.lower().endswith(".png"):
                # Print or process the PNG file
                new_file_name = file[:-4] + ".ldr.exr"
                png_to_openexr(os.path.join(root, file), os.path.join(root, new_file_name))

if __name__ == "__main__":
    # Replace 'your_directory_path' with the path of the directory you want to traverse
    directory_path = './spec-iso-data'
    traverse_directory(directory_path)
    # png_to_openexr(sys.argv[1], "output.exr")