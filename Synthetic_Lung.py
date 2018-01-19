import numpy as np
import scipy as sp
import fx
from pyevtk.hl import gridToVTK
from scipy import ndimage
# Parameters
isCreated = False

Br_dia = 30
Br_len = 50
Br_th = 20

br_dia = 15
br_len = 30
br_th = 15

Tb_dia = 10
Tb_len = 40
Tb_th = 5
Ac_dim = 30

L_dim = [int(Br_len + br_dia/2 + Tb_len + Ac_dim), int(Ac_dim/2 + Tb_dia/2 + 2*br_len + Tb_dia), int(Ac_dim)]

print L_dim

if isCreated:
    # Load data
    synLung = fx.load_data('synLung.npy')
else:
    # Make data
    synLung = np.zeros(L_dim).astype(np.uint8)

    # Bronchi
    synLung[0:Br_len,Tb_dia+br_len-Br_dia/2:Tb_dia+br_len+Br_dia/2, L_dim[2]/2-Br_th/2:L_dim[2]/2+Br_th/2] = 1

    # Bronchioles
    synLung[Br_len:Br_len+br_dia, Tb_dia:Tb_dia+br_len*2, L_dim[2]/2-br_th/2:L_dim[2]/2+br_th/2] = 1

    # Terminal Bronchioles
    synLung[Br_len+br_dia/2 - Tb_len:Br_len+br_dia/2 + Tb_len, 0:Tb_dia, L_dim[2]/2-Tb_th/2:L_dim[2]/2+Tb_th/2] = 1
    synLung[Br_len+br_dia/2 - Tb_len:Br_len+br_dia/2 + Tb_len, Tb_dia+br_len*2:2*Tb_dia+br_len*2, L_dim[2]/2-Tb_th/2:L_dim[2]/2+Tb_th/2] = 1
    # Acini
    synLung[Br_len+br_dia/2 + Tb_len:L_dim[0], -Ac_dim:, :] = 1

    fx.save_data(synLung, 'synLung')
print synLung.shape
# print synLung.shape
# print synLung
# Save as a stack of pngs.
fx.save_images(synLung*255, L_dim[2], "synLung")


# structure = fx.connected_structure(6)

# print structure
# synLung = fx.erode(synLung,structure,2)
fx.save_VTR(synLung,"5test")
seed = (60,45,15)
# print seed
# synLung[seed] = 1

R = fx.grow_region4(synLung,seed,4)

# structure = fx.connected_structure(26)

# R = fx.dilate(R,structure,2)
# synLung = fx.dilate(synLung,structure,2)

fx.save_images(R*255, L_dim[2], "R")

# print L_dim
# print L_dim[0]
# x = np.arange(0, L_dim[0]+1)
# y = np.arange(0, L_dim[1]+1)
# z = np.arange(0, L_dim[2]+1)
#
# aaa = synLung.shape
# print aaa[0]
# print type(synLung)
# print synLung
# # save as a .vtr
# gridToVTK("./synLung", x, y, z, cellData={'synLung': synLung})
# gridToVTK("./synTb", x, y, z, cellData={'synTb': R})

fx.save_VTR(R,"4test")

