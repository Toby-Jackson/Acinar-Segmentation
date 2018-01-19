import numpy as np
import scipy as sp
import fx
from scipy import ndimage
import cProfile

xHalf = fx.load_data('A250.npy')
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

TB = fx.load_data('RRRRRR.npy')
xHalfed = xHalf
#
xHalfed[TB == 1] = 120
# # xHalf = xHalf.astype(np.uint8)
fx.save_images(xHalfed, 250, "RRRRRRRR")
#
# # xHalfed[TB == 1] = 1
#
# xHalfed = ~xHalfed.astype(np.uint8)
#
# fx.save_images(xHalfed, 250, "lel")

# x = 6
#
# square = np.ones([5,6,7]).astype(np.uint8)
# square[0:3,0:3,0:3] = 0
# print square
#
# # print square
# # x = int(round(x / 2))
# x = 4
# R = fx.grow_region4(square, (2, 3, 4),1)
# print R
#



# cProfile.run('fx.grow_region4(square, (x,x,x),1)')

