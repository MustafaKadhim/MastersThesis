import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage.measure import profile_line
from skimage import io

import skimage.segmentation as seg
import skimage.filters as filters
import skimage.draw as draw
import skimage.color as color



B1_shadow_reg = nib.load(r'C:\Users\musti\Desktop\MP2RAGE Hampus Data\wetransfer-b5343e\New folder\Serie_1001_DREAM_60dg_REGIS_shadowreg_DREAM_combined.nii.gz')
B1_shadow_reg = B1_shadow_reg.get_fdata()
# B1_shadow_reg.astype(np.int32)
#print(epi_img_data_real.shape)

Mask_WM = nib.load(r'C:\Users\musti\Desktop\From Home\MP2RAGE_Mask_Zero_one.nii.gz')
Mask_WM = Mask_WM.get_fdata()
# epi_img_data_imag.astype(np.int32)
#print(epi_img_data_imag.shape)

MP2RAGE_WM = nib.load(r'C:\Users\musti\Desktop\From Home\MP2RAGE_Stacked0.5SkalleFree_Python.nii.gz')
MP2RAGE_WM = MP2RAGE_WM.get_fdata()

S_Mask = nib.load(r'C:\Users\musti\Desktop\From Home\ST2_Mask_Bin_mask.nii.gz')
S_Mask_data = S_Mask.get_fdata()
# S_Mask_data.astype(np.int32)

epi_img_imag = nib.load(r'C:\Users\musti\Desktop\MP2RAGE Hampus Data\Serie_604_Imaginary.nii.gz')
epi_img_data_imag = epi_img_imag.get_fdata()
# epi_img_data_imag.astype(np.int32)

# =============================================================================
# Showing The Images 
# =============================================================================

h_prof = 200

def show_slices(slices):
    """ Function to display row of image slices """
    fig, axes = plt.subplots(1, len(slices))
    for i, slice in enumerate(slices):
        
        axes[i].imshow(slice.T, cmap="gray", origin="lower" , interpolation='sinc')
#        axes[2].plot(np.arange(epi_img_data_real.shape[1]), np.zeros(epi_img_data_real.shape[1]) + h_prof, 'r--')
        title = [ 'MP2RAGE [0-1]', 'MP2RAGE Seg', 'MP2RAGE Seg * MP2RAGE [0-1]']
        axes[i].set_title('{}'.format(title[i]))
        divider = make_axes_locatable(axes[i])
        cax1 = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(axes[i].imshow(slice.T, cmap="gray", origin="lower" , interpolation='sinc'), cax=cax1)
        fig.tight_layout(pad=0.0)
        
        

# =============================================================================
# Go through the slices and Multiply by mask (MP2RAGE as Ref for Threshold)  and combine them into one Nii-file 
# =============================================================================      
images_segmented=[]
MP2RAGE_lista=[]
MP2RAGE_Mask = []
for s in range(256): 
    
    B1_shadow_reg_1 = B1_shadow_reg[s, :, :]
    #mask_csf  = seg_data
    
    
    Mask_WM_1 = Mask_WM[s, :, :]
    
    #S_Mask_1 = S_Mask[s,:,:] 
    
    
    # Simg_T1 = epi_img_data_imag[s, :, :]
    
    
    
    # Simg_T2 = epi_img_data_imag[s, :, :]
    
    
    # B1_WM = B1_shadow_reg_1 * 1 * Mask_WM_1# ta bort skalle
    MP2R_WM = MP2RAGE_WM[s,:,:] * 1 * Mask_WM_1

#Gm
# MP2RAGE_2[MP2RAGE < 0.4] = 0
# MP2RAGE_2[MP2RAGE > 0.64] = 0



    # MP2RAGE_Mask.append(B1_WM)
    MP2RAGE_Mask.append(MP2R_WM)
# MP2RAGE_Segmented =  MP2RAGE_2 * MP2RAGE
# images_segmented.append(MP2RAGE_Segmented)
    
vol = np.stack((MP2RAGE_Mask))
print(vol.shape)
    
    
new_image = nib.Nifti1Image(vol, epi_img_imag.affine)
nib.save(new_image, r'C:\Users\musti\Desktop\test\MP2RAGE_WM_seg_05.nii.gz')

#%%
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
        




WM_B1_plus = nib.load(r'C:\Users\musti\Desktop\test\WM_B1_plus_Coreg.nii.gz')
WM_B1_plus = WM_B1_plus.get_fdata()

MP2R_WM_test = nib.load(r'C:\Users\musti\Desktop\test\MP2RAGE_WM_seg_05.nii.gz')
MP2R_WM_test = MP2R_WM_test.get_fdata()


s_list = np.arange(90,190)
summ_y = []
summ_x = []
for s in s_list:
    WM_B1_plus_reshaped = WM_B1_plus[s,:,:]
    WM_B1_plus_reshaped = WM_B1_plus_reshaped[WM_B1_plus_reshaped != 0 ]
    #print(len(WM_B1_plus_reshaped))
    
    
    MP2R_WM_test_reshaped = MP2R_WM_test[s,:,:]
    MP2R_WM_test_reshaped = MP2R_WM_test_reshaped[MP2R_WM_test_reshaped != 0]
    

    summ_y.append(MP2R_WM_test_reshaped)
    summ_x.append(WM_B1_plus_reshaped)
    

df_x = pd.DataFrame(summ_x)
df_y = pd.DataFrame(summ_y)

 
# coef = np.polyfit(df_x.stack(),df_y.stack()[:df_x.stack().shape[0]],1)
# poly1d_fn = np.poly1d(coef) 
# plt.plot(df_x.stack(),df_y.stack()[:df_x.stack().shape[0]], 'bo',  df_x.stack(), poly1d_fn(df_x.stack()), '-r',markersize=0.1)
# plt.xlim((40,145))
# plt.ylim((0.23,0.45))
ax = sns.regplot(x=df_x.stack(), y=df_y.stack()[:df_x.stack().shape[0]], color="b", marker='o', scatter_kws={'s':0.1} )

#Coef = [0.000926=k  , 0.24931686=m]


#print(df_x.head())
#print(df_y.head())

    #plt.plot(WM_B1_plus_reshaped[start:end],MP2R_WM_test_reshaped[start:end], 'ok')
    
    # plt.plot(WM_B1_plus_reshaped[start:end],MP2R_WM_test_reshaped[start:end], 'ok')
    # fig1 = plt.figure(1)
    # coef = np.polyfit(WM_B1_plus_reshaped[start:end],MP2R_WM_test_reshaped[start:end],1)
    # poly1d_fn = np.poly1d(coef) 
    # plt.plot(WM_B1_plus_reshaped[start:end],MP2R_WM_test_reshaped[start:end], 'bo', WM_B1_plus_reshaped[start:end], poly1d_fn(WM_B1_plus_reshaped[start:end]), markersize=0.1)
    # plt.title('MP2RAGE vs B1+ (Global scale)') 
    # plt.xlabel('B1+')
    # plt.ylabel('MP2RAGE Intensity')
    # plt.ylim((0.25,0.48))
    # plt.xlim((40,np.max(WM_B1_plus_reshaped[start:end])))
    
    # fig2 = plt.figure(2)
    # show_slices([WM_B1_plus[s,:,:], MP2R_WM_test[s,:,:]])


# plt.plot(summ , 'ko')
# plt.title('MP2RAGE Mix (Global scale)') 
# plt.xlabel('B1+')
# plt.ylabel('MP2RAGE Intensity')
# plt.ylim((0.25,0.48))
# plt.xlim((20,np.max(x)))
   
#%%

# from os import listdir



# epi_img_imag = nib.load(r'C:\Users\musti\OneDrive\Skrivbord\MP2RAGE Data Hampus FSL\Serie_604_Imaginary.nii.gz')
# epi_img_data_imag = epi_img_imag.get_fdata()
# epi_img_data_imag.astype(np.int32)


 
# def list_files1(directory, extension):
#     return [f for f in listdir(directory) if f.endswith('_' + extension)]



# #print(epi_img_St1and2.shape)
# aa = []

# for i in range(0,256):
#     a = list_files1(r'C:\Users\musti\OneDrive\Skrivbord\OverLay', '{}.nii.gz'.format(i))
#     aa.append(a)

# MP2RAGE_List = []
# for i in range(len(aa)):
#     MP = nib.load(r'C:\Users\musti\OneDrive\Skrivbord\OverLay\{}'.format(aa[i][0]))
#     MP = MP.get_fdata()
#     MP2RAGE_List.append(MP)



# vol = np.stack((MP2RAGE_List))
# print(vol.shape)


# new_image = nib.Nifti1Image(vol, epi_img_imag.affine)
# nib.save(new_image, 'MP2RAGEcombined_Overlay.nii.gz')

    
#%%    
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage.measure import profile_line
from skimage import io

# =============================================================================
# Overlayers (Manuel Segment)
# =============================================================================

import cv2
from scipy import ndimage

epi_img_imag = nib.load(r'C:\Users\musti\OneDrive\Skrivbord\MP2RAGE Data Hampus FSL\Serie_604_Imaginary.nii.gz')
epi_img_data_imag = epi_img_imag.get_fdata()
epi_img_data_imag.astype(np.int32)


epi_img_real = nib.load(r'C:\Users\musti\OneDrive\Skrivbord\MP2RAGE Data Hampus FSL\Serie_603_Real.nii.gz')
epi_img_data_real = epi_img_real.get_fdata()
epi_img_data_real.astype(np.int32)
    
Seg_img_Mp2 = nib.load(r'C:\Users\musti\OneDrive\Skrivbord\TESTsliceandSegment\MP2RAGE_segmented.nii.gz')
Seg_img_Mp2 = Seg_img_Mp2.get_fdata()
Seg_img_Mp2.astype(np.int32)

BrainFree_Mp2 = nib.load(r'C:\Users\musti\OneDrive\Skrivbord\LinuxNew\wetransfer-9fe4ac\MP2RAGE_Stacked0.5SkalleFree_Python.nii.gz')
BrainFree_Mp2 = BrainFree_Mp2.get_fdata()
BrainFree_Mp2.astype(np.int32)


S_Mask = nib.load(r'C:\Users\musti\OneDrive\Skrivbord\LinuxNew\wetransfer-9fe4ac\ST2_Mask_Bin_mask.nii.gz')
S_Mask_data = S_Mask.get_fdata()
S_Mask_data.astype(np.int32)

# background = cv2.imread('field.jpg')
# overlay = cv2.imread('dice.png')
s = 218


Sreal_T1 = epi_img_data_real[:, :, s,0]
#mask_csf  = seg_data


Sreal_T2 = epi_img_data_real[:, :, s,1]



Simg_T1 = epi_img_data_imag[:, :, s,0]



Simg_T2 = epi_img_data_imag[:, :, s,1]

MP2RAGE = ((((Sreal_T1*Sreal_T2) + (Simg_T1*Simg_T2))/((Sreal_T1**2) + (Sreal_T2**2) + Simg_T2**2 + Simg_T1**2)) + 0.5 ) * S_Mask_data[:,:,s]
MP2RAGE[MP2RAGE < 0.005] = 0.0
    
# fig, ax = plt.subplots(1, 3)
    
rotated_img_1 = ndimage.rotate(BrainFree_Mp2[:,:,s], 90)
rotated_img_2 = ndimage.rotate(Seg_img_Mp2[:,:,s], 90)
rotated_img_3 = ndimage.rotate(MP2RAGE, 90)


# ax[0].imshow(rotated_img_3, cmap='Greys_r',  interpolation='sinc', vmin=0, vmax=1)
# ax[0].imshow(rotated_img_2, cmap='jet', alpha=0.5, interpolation='gaussian', vmin=0, vmax=1)
# ax[0].set_title('WM Overlay MP2RAGE')


# ax[1].imshow(MP2RAGE, cmap='Greys_r',  interpolation='sinc')
# ax[1].set_title('MP2RAGE [0-1]')
# divider = make_axes_locatable(ax[1])
# cax1 = divider.append_axes("right", size="5%", pad=0.05)
# fig.colorbar(ax[1].imshow(rotated_img_3, cmap='Greys_r',  interpolation='sinc', vmin=0, vmax=1), cax=cax1)
# fig.tight_layout(pad=0.1)


# ax[2].imshow(rotated_img_1, cmap='Greys_r',  interpolation='sinc')
# ax[2].imshow(rotated_img_2, cmap='jet', alpha=0.5, interpolation='sinc', vmin=0, vmax=1)
# ax[2].set_title('WM Overlay MP2RAGE [-0.5 - 0.5]')


# =============================================================================
# K-Means Cluster 
# =============================================================================



img = cv2.imread(r'C:\Users\musti\OneDrive\Skrivbord\TESTsliceandSegment\KmeansBrainMP2RAGEexemple.png')
Z = img.reshape((-1,3))
#Z[Z < 0.005] = 0

# # # convert to np.float32
Z = np.float32(Z)

# # define criteria, number of clusters(K) and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
K = 5
ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_PP_CENTERS)

# # Now convert back into uint8, and make original image
center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape((img.shape))

#cv2.imshow('K-means Segmented',res2)

#cv2.imwrite('K-meansSegmented.png', res2 )


plt.hist(MP2RAGE.ravel(),256,[0,1])
plt.axvline(x=0.4, linewidth=0.7, color='g', linestyle='--')
plt.text(0.48, 10**3, 'GM', dict(size=20), color='green')
plt.axvline(x=0.6, linewidth=0.7, color='g', linestyle='--')

plt.axvline(x=0.72, linewidth=0.7, color='r', linestyle='--')
plt.text(0.85, 10**3, 'WM', dict(size=20), color='red')
plt.axvline(x=1, linewidth=0.7, color='r', linestyle='--')

plt.title('Histogram MP2RAGE')
plt.show()

# res2[res2 < 237]= 0
# # res2[res2 < center[0][0]]


# figure_size = 15
# plt.figure(figsize=(figure_size,figure_size))
# plt.subplot(1,3,1),plt.imshow(img)
# plt.title('Original Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(1,3,2),plt.imshow(res2)
# plt.title('Segmented Image when K = %i' % K), plt.xticks([]), plt.yticks([])
# plt.subplot(1,3,3),plt.imshow(ndimage.rotate(Seg_img_Mp2[:,:,160], 90))
# plt.title('Segmented Mask'), plt.xticks([]), plt.yticks([])
# plt.show()