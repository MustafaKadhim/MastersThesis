import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage.measure import profile_line
from skimage import io
import seaborn as sns
import skimage.segmentation as seg
import skimage.filters as filters
import skimage.draw as draw
import skimage.color as color
import pandas as pd


def show_slices(slices):
    """ Function to display row of image slices """
    fig, axes = plt.subplots(1, len(slices))
    for i, slice in enumerate(slices):
        
        axes[i].imshow(slice.T, cmap="gray", origin="lower" , interpolation='sinc')
#        axes[2].plot(np.arange(epi_img_data_real.shape[1]), np.zeros(epi_img_data_real.shape[1]) + h_prof, 'r--')
        title = [ 'WM_B1_plus', 'MP2RAGE_WM', 'MP2RAGE Seg * MP2RAGE [0-1]']
        axes[i].set_title('{}'.format(title[i]))
        divider = make_axes_locatable(axes[i])
        cax1 = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(axes[i].imshow(slice.T, cmap="gray", origin="lower" , interpolation='sinc'), cax=cax1)
        fig.tight_layout(pad=0.0)
        




WM_B1_plus = nib.load(r'C:\Users\musti\OneDrive\Skrivbord\Endast for MATLAB\WM_B1_plus_Coreg.nii.gz')
WM_B1_plus = WM_B1_plus.get_fdata()

MP2R_WM_test = nib.load(r'C:\Users\musti\OneDrive\Skrivbord\Endast for MATLAB\MP2RAGE_WM_seg_05.nii.gz')
MP2R_WM_test = MP2R_WM_test.get_fdata()


s_list = np.arange(10,250)
summ_y = []
summ_x = []
for s in s_list:
    WM_B1_plus_reshaped = WM_B1_plus[s,:,:]
    WM_B1_plus_reshaped = WM_B1_plus_reshaped[WM_B1_plus_reshaped > 0 ]
    #print(len(WM_B1_plus_reshaped))
    
    
    MP2R_WM_test_reshaped = MP2R_WM_test[s,:,:]
    MP2R_WM_test_reshaped = MP2R_WM_test_reshaped[MP2R_WM_test_reshaped > 0]
    

    summ_y.append(MP2R_WM_test_reshaped)
    summ_x.append(WM_B1_plus_reshaped)
    

df_x = pd.DataFrame(summ_x)
df_y = pd.DataFrame(summ_y)

 
coef = np.polyfit(df_x.stack(),df_y.stack()[:df_x.stack().shape[0]],1)
poly1d_fn = np.poly1d(coef) 
plt.plot(df_x.stack(),df_y.stack()[:df_x.stack().shape[0]], 'bo',  df_x.stack(), poly1d_fn(df_x.stack()), '-r',markersize=0.1)
plt.xlim((40,145))
plt.ylim((0.23,0.45))
#ax = sns.regplot(x=df_x.stack(), y=df_y.stack()[:df_x.stack().shape[0]], color="b", marker='o', scatter_kws={'s':0.1} )

#%%
fig2, ax = plt.subplots(1, 1)
ax.hist([df_x.stack()], bins=256, color='red', label = 'Local Flip Angles (WM)')
plt.legend()
fig3, ax = plt.subplots(1, 1)
ax.hist(df_y.stack()[:df_x.stack().shape[0]], bins=256, color='skyblue', label='MP2RAGE signal (WM)')  
plt.legend()

