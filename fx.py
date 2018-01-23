import numpy as np
import os
import scipy as sp
import time
from PIL import Image
from pyevtk.hl import gridToVTK
from scipy import ndimage


def read_tiff(path, n_images):
    # Source: https://stackoverflow.com/questions/18602525/python-pil-for-loop-to-work-with-multi-image-tiff
    """
    Reads in the .tiff.
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

    return np.array(images, dtype=np.uint8)


def save_data(image, filename):
    np.save(filename, image)


def load_data(filename):
    return np.load(filename)


def save_images(images, numImages, fileprefix):
    """
    Saves data as image files + imgvol file (visIt)
    :param images:  Data
    :param numImages:
    :param fileprefix: name of file
    :return: /
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


def search_region(im_size):
    """
    # Creates a list of all the coordinates/indexes of the image.
    :param im_size:
    :return:
    """
    x_p = np.arange(1, im_size[1], 1, dtype=int)
    y_p = np.arange(1, im_size[0], 1, dtype=int)
    z_p = np.arange(1, im_size[2], 1, dtype=int)
    region = np.vstack(np.meshgrid(x_p, y_p, z_p)).reshape(3, -1).T
    region = region.tolist()
    return region


def connected_structure(no):
    """
    Structural element for the erosion/dilation operators
    :param no:
    :return:
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
    for i in range(num):
        data = sp.ndimage.morphology.binary_dilation(data, structure).astype(np.uint8)
    return data


def erode(data, structure, num=1):
    for i in range(num):
        data = sp.ndimage.morphology.binary_erosion(data, structure).astype(np.uint8)
    return data


def save_VTR(data, file):
    """
    save as a .vtr file for VisIt
    :param data:
    :param file:
    :return:
    """
    dim = data.shape
    x = np.arange(0, dim[0] + 1)
    y = np.arange(0, dim[1] + 1)
    z = np.arange(0, dim[2] + 1)
    filename = "./" + file

    gridToVTK(filename, x, y, z, cellData={file: data})

def grow_region(images, seed):
    """
    Region growing algorithm
    :param images:
    :param seed:
    :return:
    """
    im_size = images.shape
    searchable = search_region(im_size)
    region = [list(seed)]
    wholeregion = region
    searchable.remove(list(seed))
    count = 0
    isGrowing = True
    while isGrowing:
        # Limit the volume where expansion searches
        count = count + 1
        print count
        print len(searchable)
        print len(region)
        print ''
        temp_grow = []
        searched = []
        isGrowing = False
        x_Max = max([item[1] for item in region]) + 1
        y_Max = max([item[0] for item in region]) + 1
        z_Max = max([item[2] for item in region]) + 1
        x_Min = min([item[1] for item in region]) - 1
        y_Min = min([item[0] for item in region]) - 1
        z_Min = min([item[2] for item in region]) - 1
        # print x_Max, x_Min, y_Max, y_Min, z_Min,z_Max
        # print searchable
        for coord in searchable:
            # Check whether in vicinity of region.
            # np.where???
            if (x_Min <= coord[1] <= x_Max) and (y_Min <= coord[0] <= y_Max) and (z_Min <= coord[2] <= z_Max):
                # Check whether connected to region
                # print ''
                # print coord

                for point in region:

                    # loop
                    # print point
                    # -  actually its not its like 8
                    # print 'x'
                    # print (point[0] - 1 <= coord[0] <= point[0] + 1) and (point[1] - 1 <= coord[1] <= point[1] + 1) and (point[2] - 1 <= coord[2] <= point[2] + 1)
                    # if (point[0] - 1 <= coord[0] <= point[0] + 1) and (point[1] - 1 <= coord[1] <= point[1] + 1) and (
                    #             point[2] - 1 <= coord[2] <= point[2] + 1):

                    # 4 connectivity
                    if ((point[0] - 1 <= coord[0] <= point[0] + 1) and (coord[1] == point[1]) and (
                        coord[2] == point[2])) or (
                            (point[1] - 1 <= coord[1] <= point[1] + 1) and (coord[0] == point[0]) and (
                        coord[2] == point[2])) or (
                            (point[2] - 1 <= coord[2] <= point[2] + 1) and (coord[1] == point[1]) and (
                        coord[0] == point[0])):

                        # if connected, check whether to grow region.
                        searched.append(coord)
                        # print coord
                        if np.sum(images[coord[0] - 1:coord[0] + 2, coord[1] - 1:coord[1] + 2,
                                  coord[2] - 1:coord[2] + 2]) == 27:
                            # print'y'
                            isGrowing = True
                            temp_grow.append(coord)
                            break

        region = temp_grow
        wholeregion = wholeregion + temp_grow
        # print len(searchable)
        searchable = [x for x in searchable if x not in searched]
        # print len(searchable)
        #
        # if count == 1:
        #     isGrowing = False

    R = np.zeros(im_size).astype(np.uint8)

    for i in wholeregion:
        # print tuple(i)
        # print synLung[tuple(i)]
        R[tuple(i)] = 1
        # print synLung[tuple(i)]

    R[seed] = 1
    print R
    return R

def grow_region2(images, seed):
    im_size = images.shape
    growths = [tuple(seed)]
    searched = [tuple(seed)]
    seed = np.asarray(seed)

    growth_dir = [(0, 0, 0, 0, 0, 1),
                  (0, 0, 0, 0, -1, 0),
                  (0, 0, 0, 1, 0, 0),
                  (0, 0, -1, 0, 0, 0),
                  (0, 1, 0, 0, 0, 0),
                  (-1, 0, 0, 0, 0, 0)]

    g_dir = [[0, 0, 1], [0, 0, -1],
             [0, 1, 0], [0, -1, 0],
             [1, 0, 0], [-1, 0, 0]]
    # print g_dir
    count = 0
    # TODO Check if on boundary

    # else
    # x y z
    search_dim = np.array([-1, 1, -1, 1, -1, 1])
    isGrowing = True
    while isGrowing:
        count += 1
        print count
        bucket = np.zeros(6)

        for x in range(np.max([search_dim[0] + seed[1], 0]), np.min([search_dim[1] + seed[1] + 1, im_size[1]])):
            for y in range(np.max([search_dim[2] + seed[0], 0]), np.min([search_dim[3] + seed[0] + 1, im_size[0]])):
                for z in range(np.max([search_dim[4] + seed[2], 0]), np.min([search_dim[5] + seed[2] + 1, im_size[2]])):
                    # print x, y, z
                    point = np.array([y, x, z])
                    # print point
                    # if found before
                    if tuple(point) in searched:
                        # print 'L'
                        continue
                    else:
                        # if connected
                        for growth in growths:
                            direction = point - growth

                            if list(direction) in g_dir:
                                d = np.where((direction == g_dir).all(axis=1))
                                bucket[d[0]] += 1
                                # print 'k'
                                searched.append(tuple(point))
                                # check region.
                                if np.sum(images[point[1] - 1:point[1] + 2, point[0] - 1:point[0] + 2,
                                          point[2] - 1:point[2] + 2]) == 27:
                                    growths.append(tuple(point))
                                    break
        if np.sum(bucket) > 0:
            for i in range(6):
                if (bucket[int(i)] != 0):
                    search_dim += growth_dir[int(i)]
                    # print search_dim
        else:
            isGrowing = False
        # print np.asarray(growths)
        R = np.zeros(im_size).astype(np.uint8)
        for i in growths:
            R[tuple(i)] = 1
            # print R

    return R


def grow_region3(images, seed):
    im_size = images.shape
    growths = [tuple(seed)]
    # searched = [tuple(seed)]
    seed = np.asarray(seed)
    R = np.zeros(im_size).astype(np.uint8)
    searched = np.zeros(im_size).astype(np.uint8)

    growth_dir = [(0, 0, 0, 0, 0, 1),
                  (0, 0, 0, 0, -1, 0),
                  (0, 0, 0, 1, 0, 0),
                  (0, 0, -1, 0, 0, 0),
                  (0, 1, 0, 0, 0, 0),
                  (-1, 0, 0, 0, 0, 0)]

    g_dir = [[0, 0, 1], [0, 0, -1],
             [0, 1, 0], [0, -1, 0],
             [1, 0, 0], [-1, 0, 0]]
    # print g_dir
    count = 0
    # TODO Check if on boundary

    # else
    # x y z
    search_dim = np.array([-1, 1, -1, 1, -1, 1])
    isGrowing = True
    while isGrowing:
        count += 1
        print count
        bucket = np.zeros(6)

        x_range = [np.max([search_dim[0] + seed[1], 0]), np.min([search_dim[1] + seed[1] + 1, im_size[1]])]
        y_range = [np.max([search_dim[2] + seed[0], 0]), np.min([search_dim[3] + seed[0] + 1, im_size[0]])]
        z_range = [np.max([search_dim[4] + seed[2], 0]), np.min([search_dim[5] + seed[2] + 1, im_size[2]])]

        for x in range(x_range[0], x_range[1]):
            for y in range(y_range[0], y_range[1]):
                for z in range(z_range[0], z_range[1]):
                    # print x, y, z
                    # point = np.array([y, x, z])
                    point = (x, y, z)
                    # print point
                    # if found before
                    if searched[point] == 1:
                        # print 'L'
                        continue
                    else:
                        # if connected
                        ha = np.array(point)
                        for growth in growths:
                            direction = ha - growth

                            if list(direction) in g_dir:
                                d = np.where((direction == g_dir).all(axis=1))
                                bucket[d[0]] += 1
                                # print 'k'
                                searched[point] = 1
                                # check region.
                                # OR go through each point and if zero, stop.
                                if np.sum(images[point[1] - 1:point[1] + 2, point[0] - 1:point[0] + 2,
                                          point[2] - 1:point[2] + 2]) == 27:
                                    growths.append(point)
                                    break
        if np.sum(bucket) > 0:
            for i in range(6):
                if bucket[int(i)] != 0:
                    search_dim += growth_dir[int(i)]
                    # print search_dim
        else:
            isGrowing = False
            # print np.asarray(growths)

    for i in growths:
        R[tuple(i)] = 1
    # print R
    print R
    return R


def grow_region4(imgs, seed, n):
    """
    Region growing algortihm
    :param imgs: data set
    :param seed: point.
    :param n: number of erosion/dilation operators
    :return:
    """
    t0 = time.time()
    fh = open("SS.txt", "w")
    img_size = imgs.shape
    # print img_size
    img2 = np.zeros(img_size).astype(np.uint8)
    img2 = erode(imgs, connected_structure(6), n)
    # print img2
    iter_num = 0
    growth_num = 0
    isGrowing = True

    # img_skeleton = np.zeros(img_size).astype(np.uint8)
    img_skeleton = (img2 == 0) * -1
    img_skeleton[seed] = 3
    growth = [list(seed)]
    neighbourhood = []
    # print imgs[seed]
    while isGrowing:
        iter_num = iter_num + 1
        print iter_num
        print len(neighbourhood)
        print len(growth)
        print growth_num


        isGrowing = False
        new_growth = []
        t1 = time.time()
        x_Max = max([item[2] for item in growth]) + 1
        y_Max = max([item[1] for item in growth]) + 1
        z_Max = max([item[0] for item in growth]) + 1
        x_Min = min([item[2] for item in growth]) - 1
        y_Min = min([item[1] for item in growth]) - 1
        z_Min = min([item[0] for item in growth]) - 1
        print z_Min, z_Max, y_Min, y_Max, x_Min, x_Max
        # print searchable
        print ''
        # Limit the volume where expansion searches
        s = img_skeleton[z_Min:z_Max+1, y_Min:y_Max+1, x_Min:x_Max+1]
        # print s
        s[s == 0] = 1
        # s[(0,1,0)] = 4
        # s[0,0,1] = 5
        # s[1,0,0] = 6
        # print s
        s = (np.vstack(s.reshape(z_Max-z_Min+1, -1,order='F'))).reshape(-1,y_Max-y_Min+1,order='C').T
        # print s
        s = s.reshape(1,-1).T
        # print s
        # print img_skeleton
        x_p = np.arange(x_Min, x_Max + 1, 1, dtype=int)
        y_p = np.arange(y_Min, y_Max + 1, 1, dtype=int)
        z_p = np.arange(z_Min, z_Max + 1, 1, dtype=int)
        # print z_p, y_p, x_p,
        e =  (np.vstack(np.meshgrid(z_p,  y_p, x_p)).reshape(3, -1,).T)
        # print e
        neighbourhood = e[(np.where(s == 1)[0]).T, :]
        neighbourhood = neighbourhood.tolist()
        t2 = time.time()
        # for i in neighbourhood:
        #     print i
        #     print img_skeleton[tuple(i)]
        # print neighbourhood
        # print len(neighbourhood)
        # print growth
        # For every point in search region
        for coord in neighbourhood:
            # Take a point in the object region
            for point in growth:
                # Check whether they are 6 connected
                # print coord, point
                if ((point[0] - 1 <= coord[0] <= point[0] + 1) and (coord[1] == point[1]) and (coord[2] == point[2])) \
                        or ((point[1] - 1 <= coord[1] <= point[1] + 1) and (coord[0] == point[0]) and (coord[2] == point[2])) \
                        or ((point[2] - 1 <= coord[2] <= point[2] + 1) and (coord[1] == point[1]) and (coord[0] == point[0])):
                    # print coord, point
                    # print 'hi'
                    img_skeleton[tuple(coord)] = -1 # Checked - remove from list
                    # if connected, check whether to grow point.
                    # print np.sum(imgs[coord[0] - 1:coord[0] + 2, coord[1] - 1:coord[1] + 2,
                    #                coord[2] - 1:coord[2] + 2])
                    # print np.sum(imgs[coord[0] - 1:coord[0] + 2,
                    #           coord[1] - 1:coord[1] + 2,
                    #           coord[2] - 1:coord[2] + 2])
                    if np.sum(imgs[coord[0] - 1:coord[0] + 2,
                              coord[1] - 1:coord[1] + 2,
                              coord[2] - 1:coord[2] + 2]) == 27:

                        isGrowing = True
                        new_growth.append(coord)
                        img_skeleton[tuple(coord)] = 3
                        growth_num += 1
                        break
                else:
                    img_skeleton[tuple(coord)] = 1
        # print img_skeleton
        t3 = time.time()
        fh.write("{:f} \t {:f}\n".format((t2-t1), (t3-t2)))

        growth = new_growth
        # if iter_num == 20:
            # isGrowing = False
        # print img_skeleton
    img_skeleton = (img_skeleton == 3)*1
    img_skeleton = dilate(img_skeleton, connected_structure(26), n)
    fh.write("{:f}\n".format(time.time()-t0))
    fh.close()
    return img_skeleton

def grow_region5(imgs, seed, n):
    """
    Region growing algorithm. - same as 4 but without the print statements and commented out lines.
    :param imgs: data set
    :param seed: point.
    :param n: number of erosion/dilation operators
    :return:
    """

    img2 = erode(imgs, connected_structure(6), n)
    isGrowing = True

    # Img_skeleton holds infomation about each voxel.
    # -1 is not part of the region.
    # 0 could be - unknown
    # 1 could be - unknown but is what the search region contains
    # 3 is part of region.

    # Elminate all the areas that arnt possibly region from search space
    img_skeleton = (img2 == 0) * -1
    img_skeleton[seed] = 3

    growth = [list(seed)]
    neighbourhood = []
    iter_num = 0

    while isGrowing:
        isGrowing = False
        new_growth = []

        iter_num = iter_num + 1
        print iter_num

        x_Max = max([item[2] for item in growth]) + 1
        y_Max = max([item[1] for item in growth]) + 1
        z_Max = max([item[0] for item in growth]) + 1
        x_Min = min([item[2] for item in growth]) - 1
        y_Min = min([item[1] for item in growth]) - 1
        z_Min = min([item[0] for item in growth]) - 1

        # Limit the volume where expansion searches
        s = img_skeleton[z_Min:z_Max+1, y_Min:y_Max+1, x_Min:x_Max+1]

        # examine whether it has been searched before or is new.
        s[s == 0] = 1

        # Reorder it to make it the same shape as the reordered meshgrid bellow
        s = (np.vstack(s.reshape(z_Max-z_Min+1, -1,order='F'))).reshape(-1,y_Max-y_Min+1,order='C').T
        s = s.reshape(1,-1).T

        x_p = np.arange(x_Min, x_Max + 1, 1, dtype=int)
        y_p = np.arange(y_Min, y_Max + 1, 1, dtype=int)
        z_p = np.arange(z_Min, z_Max + 1, 1, dtype=int)
        # correspooinding coordinates of each voxel in search space
        e =  (np.vstack(np.meshgrid(z_p,  y_p, x_p)).reshape(3, -1,).T)

        # creates the search space depending on img_skeleton value
        neighbourhood = e[(np.where(s == 1)[0]).T, :]
        neighbourhood = neighbourhood.tolist()

        # For every point in search space
        for coord in neighbourhood:
            # Take a point that grew in the previous iteration
            for point in growth:
                # Check whether they are connected / 6
                if ((point[0] - 1 <= coord[0] <= point[0] + 1) and (coord[1] == point[1]) and (coord[2] == point[2])) \
                        or ((point[1] - 1 <= coord[1] <= point[1] + 1) and (coord[0] == point[0]) and (coord[2] == point[2])) \
                        or ((point[2] - 1 <= coord[2] <= point[2] + 1) and (coord[1] == point[1]) and (coord[0] == point[0])):
                    # Remove from list for further checking
                    img_skeleton[tuple(coord)] = -1
                    # if connected, check whether to grow point.
                    if np.sum(imgs[coord[0] - 1:coord[0] + 2,
                              coord[1] - 1:coord[1] + 2,
                              coord[2] - 1:coord[2] + 2]) == 27:

                        isGrowing = True
                        new_growth.append(coord)
                        img_skeleton[tuple(coord)] = 3
                        break
                # else:
                #     img_skeleton[tuple(coord)] = 1
        growth = new_growth

    img_skeleton = (img_skeleton == 3)*1
    img_skeleton = dilate(img_skeleton, connected_structure(26), n)
    return img_skeleton
