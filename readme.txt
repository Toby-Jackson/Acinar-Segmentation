To obtain an 'Acinus'

* Call read_tiff or similar to load in your data and save using save_data
* Call run_grow_region with a seed in the terminal bronchiole + an integer N.
* Call remove_region to remove the previous segment from the main.
* Call run_grow_region again with a different seed in the branch of the tb and an integer M to obtain an 'acinus'

N is large enough to remove all the smaller branches
M is determined by the formula =  alveolar mouth opening / voxelsize / 2
seed = (row,col,z)


Example:
x = fx.read_tiff('D2.tif',322)
fx.save_data(x,'D2')
fx.run_grow_region('D2','D2_1', (570,40,213), 15)
fx.remove_region('D2','D2_1','D2_2')
fx.run_grow_region('D2_2','D2_3',(50,360,289),6)