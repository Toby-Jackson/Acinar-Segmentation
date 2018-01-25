import numpy as np
import scipy as sp
import fx
from scipy import ndimage
import cProfile

# tiff = fx.read_tiff('Segmentation.tif',500)
# print tiff.shape
#
# mmm = tiff[:,:,:]
# fx.save_data(mmm,'mmm')

# # t = fx.load_data('A250.npy')
# # n = np.copy(m)
# # s = t[:,:,:]
# # t[0:100,0:50,0:25] = 100
# # seed = (249,249,249)
# # # print s.shape
# # s = (s == 0)*1
# n = m[100:300,50:400,300::]
# # s = fx.erode(s, fx.connected_structure(6),2)
# # print s
# # s[tuple(seed)] = 122
# print n.shape
# fx.save_images((n).astype(np.uint8),200,'t2')
# fx.save_data(n,'n')
# SS = fx.grow_region4(s.astype(np.uint8),seed,2)
# fx.save_VTR(SS,'SS')
# fx.save_data(SS,'SS')


# xHalf = fx.load_data('A250.npy')
# #
# # xHalf = ~xHalf
# #
# xHalf = (xHalf == 0)*1
# print xHalf
# # xHalf = fx.erode(xHalf, fx.connected_structure(6),4)
#
# seed = (230,30,55)
# R = fx.grow_region4(xHalf.astype(np.uint8),seed,4)
# # R = fx.dilate(R,fx.connected_structure(26),5)
#
# # fx.save_images(255 * xHalf, 250, "half")
# fx.save_images(255*R, 250, "lll")
# fx.save_VTR(R, "xhalf2")
# # fx.save_VTR(R, "R")
#
# fx.save_data(R,"RRRRRR")

# TB = fx.load_data('RRRRRR.npy')
# xHalfed = xHalf
# #
# xHalfed[TB == 1] = 120
# # # xHalf = xHalf.astype(np.uint8)
# fx.save_images(xHalfed, 250, "RRRRRRRR")
#
# # xHalfed[TB == 1] = 1
#
# xHalfed = ~xHalfed.astype(np.uint8)
#
# fx.save_images(xHalfed, 250, "lel")

# x = 15
#
# # square = np.ones([5,6,7]).astype(np.uint8)
# square = np.ones([x,x,x]).astype(np.uint8)
# square[0:3,0:3,0:3] = 0
# print square
#
# # print square
# x = int(round(x / 2))
# # x = 4
# # R = fx.grow_region7(square, (2, 3, 4),1)
# # print R
#
#
#
#
# cProfile.run('fx.grow_region7(square, (x,x,x),1)')

# m = fx.load_data('mmm.npy')

# #
# data = fx.load_data('mmm.npy')
# data = (data == 0)*1
# data = fx.erode(data,fx.connected_structure(6),3)
#
# fx.save_images(255*data.astype(np.uint8),500,'eroded')

# #
#
# fx.save_data(data,'mmm')
# filename = 'q1'
# # data = fx.load_data('n.npy')
# # print data
#

# seed(row,col,image)
seed = (220,220,300)
# #
# # # print data.shape
# # # growth = fx.grow_region4(data.astype(np.uint8),seed,2)
# # # fx.save_VTR(growth, filename)
# # # fx.save_data(growth, filename)
# #
# # g = fx.grow_region6(data,seed,2)
# # fx.save_images(g,500,'q')
# # fx.save_VTR(g, filename)
# # fx.save_data(g, filename)
# # print g
# cProfile.run("fx.run_grow_region('mmm','num7',seed,2)")
# # fx.run_grow_region('mmm','num7',seed,12)
# TB = fx.load_data('num7.npy')
# All = fx.load_data('mmm.npy')
# print TB.shape
# print All.shape
# # print All
#
# removed = ((TB == 1) * 255 + All).astype(np.uint8)
# print removed
# fx.save_data(removed,'removed')
# fx.save_images(removed,500,'removed')
# # print removed.shape
seed = (80,460,347)

fx.run_grow_region('removed','acini',seed,5)