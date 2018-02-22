import numpy as np
import os
import scipy as sp
import time
from PIL import Image
from pyevtk.hl import gridToVTK
from scipy import ndimage
import sys

def read_tiff(path, n_images):
    # Source: https://stackoverflow.com/questions/18602525/python-pil-for-loop-to-work-with-multi-image-tiff
    """
    Reads in the .tiff stack of the segmentation
    path - Path to the multipage-tiff file
    n_images - Number of pages in the tiff file
    """
    img = Image.open(path)
    images = []
    for i in range(n_images):
        try:
            img.seek(i)
            slice_ = np.zeros(img.size)
            for j in range(slice_.shape[0]):
                for k in range(slice_.shape[1]):
                    slice_[j, k] = img.getpixel((j, k))

            images.append(slice_)

        except EOFError:
            # Not enough frames in img
            break
    # Transpose because something happens to array somewhere between importing it, and saving it back again.
    # and this reverses it.
    return np.array(images, dtype=np.uint8).transpose()


def save_data(image, filename):
    """
    Short fx that saves the image/np array as a .npy
    :param image: the array
    :param filename: string before '.npy'
    """
    np.save(filename, image)


def load_data(filename):
    """
    Short fx that opens a .npy file and returns it
    :param filename: filename
    :return: array
    """
    return np.load(filename + '.npy')


def save_images(images, numImages, fileprefix):
    """
    Saves data as image files + imgvol file (for visIt - when VTK doesn't work)
    :param images:  Data to be saved
    :param numImages: number of images in images
    :param fileprefix: name of file
    """
    directory = "{}/".format(fileprefix)
    if not os.path.exists(directory):
        os.makedirs(directory)

    filehandle = open("{}/{}.imgvol".format(fileprefix, fileprefix), "w")

    for i in range(0, numImages):
        im = Image.fromarray(images[:, :, i])
        filename = "{}/{}{}.png".format(fileprefix, fileprefix, i)
        filehandle.write("{}{}.png\n".format(fileprefix, i))
        im.save(filename)

    filehandle.close()


def connected_structure(no):
    """
    Structural element for the erosion/dilation operators
    :param no:
    :return: structural elememnt
    scipy.ndimage.morphology.generate_binary_structure(rank, connectivity)
    ^^^ could be bettter. or
    skimage.morphology._____
    """
    if no == 6:
        structure = np.zeros((3,) * 3)
        structure[1, :, 1] = 1
        structure[:, 1, 1] = 1
        structure[1, 1, :] = 1
    elif no == 26:
        structure = np.ones((3,) * 3)
    else:
        print 'error: connected structure'
    return structure


def dilate(data, structure, num=1):
    """
    performs dilation on the data set n number of times
    :param data:
    :param structure: the structuring element
    :param num:
    :return: dilated image
    """
    for i in range(num):
        data = sp.ndimage.morphology.binary_dilation(data, structure).astype(np.uint8)
    return data


def erode(data, structure, num=1):
    """
    performs erosion on the data set n number of times
    :param data:
    :param structure: the structuring element
    :param num:
    :return: dilated image
    """
    for i in range(num):
        data = sp.ndimage.morphology.binary_erosion(data, structure).astype(np.uint8)
    return data


def save_VTR(data, file):
    """
    save as a .vtr file for VisIt
    obtained/adapted from https://pyscience.wordpress.com/2014/09/06/numpy-to-vtk-converting-your-numpy-arrays-to-vtk-arrays-and-files/
    :param data: array
    :param file: filename
    """
    dim = data.shape
    x = np.arange(0, dim[0] + 1)
    y = np.arange(0, dim[1] + 1)
    z = np.arange(0, dim[2] + 1)
    filename = "./" + file

    gridToVTK(filename, x, y, z, cellData={file: data})


def grow_region(imgs, seed, n):
    """
    Region growing algorithm.
    :param imgs: data set
    :param seed: initial starting point.
    :param n: number of erosion/dilation operators
    :return: binary image of seeded region
    """
    # erode the image
    img_eroded = erode(imgs, connected_structure(6), n)
    # binary value controlling the loop
    isGrowing = True

    # Eliminate all the areas that arn't possibly region from search space
    img_skeleton = (img_eroded == 0) * -1
    img_skeleton[seed] = 1

    # Img_skeleton holds information about each voxel.
    # -1 is not part of the region.
    # 0 could be - unknown
    # 1 is part of region.

    # inital growth
    growth = [seed]
    iter_num = 0

    # directions of potential growth
    directions = [[0, 0, 1], [0, 0, -1],
                  [0, 1, 0], [0, -1, 0],
                  [1, 0, 0], [-1, 0, 0]]

    while isGrowing:
        isGrowing = False
        new_growth = []

        iter_num = iter_num + 1
        print iter_num,

        # Goes through every recent growth point and searches +1/-1 around it and sees if it is 'connected', if it is it
        # adds it to the search space for the next loop, else it removes it from being checked again.
        for point in growth:
            for direction in directions:
                coord = np.array(point) + direction
                if img_skeleton[tuple(coord)] == 0:
                    if np.sum(imgs[coord[0] - 1:coord[0] + 2,
                              coord[1] - 1:coord[1] + 2,
                              coord[2] - 1:coord[2] + 2]) == 27:
                        isGrowing = True
                        new_growth.append(coord)
                        img_skeleton[tuple(coord)] = 1
                        # break

                    else:
                        img_skeleton[tuple(coord)] = -1

        growth = new_growth

    print 'growth done'

    # turning it back into a binary image
    img_skeleton = (img_skeleton == 1) * 1
    # dilating it back
    img_skeleton = dilate(img_skeleton, connected_structure(26), n)
    return img_skeleton


def run_grow_region(loadfilename, savefilename, seed, n):
    """
    Loads data, Calls the grow_region function and saves the data
    2. to clean the tb growing.
    :param loadfilename: string - file prefix
    :param savefilename: string - file prefix
    :param seed: seed pooint
    :param n: number of erosion/dilations
    """
    loadfilename = loadfilename + '.npy'
    print savefilename
    data = load_data(loadfilename)

    # Flips zeros and ones.
    data = (data == 0)*1

    growth = grow_region(data.astype(np.uint8),seed,n)

    # second region grow to smooth out the growth carried out through n erosions/dilations. now ignoring the not grown into region from above.
    growth = grow_region((growth * data).astype(np.uint8),seed,0)
    growth = dilate(growth, connected_structure(26), 1)

    # save data
    save_data(growth, savefilename)
    save_images(growth*255, data.shape[2], savefilename)
    save_VTR(growth, savefilename)


def remove_region(datafilename, regionfilename, savefilename):
    """
    Removes a region from another region
    :param datafilename: string, The data that the region is to be removed from
    :param regionfilename: string, the region that is getting removed
    :param savefilename: string, filename
    :return:
    """
    data = load_data(datafilename)
    region = load_data(regionfilename)
    # makes sure both are 0 and 1s
    region = (region > 0) * 1
    data = (data > 0) * 1

    # remove region from data,
    data[region == 1] = 1

    save_data(data.astype(np.uint8), savefilename)
    # save_images(data.astype(np.uint8) * 255, data.shape[2], savefilename)