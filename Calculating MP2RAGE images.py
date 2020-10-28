# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 08:13:44 2020

@author: musti
"""

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

path = r"C:\Users\musti\Desktop\Hampus_New_Data"

file_1_re = "\Serie_603_Real.nii.gz"
file_1_im = "\Serie_604_Imaginary.nii.gz"

file_2_re = "\Serie_703_Real.nii.gz"
file_2_im = "\Serie_704_Imaginary.nii.gz"

file_3_re = "\Serie_804_Imaginary.nii.gz"
file_3_im = "\Serie_804_Imaginary.nii.gz"

file_4_re = "\Serie_1303_Real.nii.gz"
file_4_im = "\Serie_1304_Imaginary.nii.gz"



epi_img_real_17 = nib.load(path  + file_1_re)
epi_img_real_17_r = epi_img_real_17.get_fdata()
print("Input Shape of data:", epi_img_real_17_r.shape)



epi_img_imag_17 = nib.load(path  + file_1_im)
epi_img_imag_17_i = epi_img_imag_17.get_fdata()
#epi_img_imag_17.astype(np.int32)
#print(epi_img_data_imag.shape)


epi_img_real_12 = nib.load(path  + file_2_re)
epi_img_real_12_r = epi_img_real_12.get_fdata()

#print(epi_img_data_real.shape)

epi_img_imag_12 = nib.load(path  + file_2_im)
epi_img_imag_12_i = epi_img_imag_12.get_fdata()

#print(epi_img_data_imag.shape)



epi_img_real_09 = nib.load(path  + file_3_re)
epi_img_real_09_r = epi_img_real_09.get_fdata()

#print(epi_img_data_real.shape)

epi_img_imag_09 = nib.load(path  + file_3_im)
epi_img_imag_09_i = epi_img_imag_09.get_fdata()

#print(epi_img_data_imag.shape)


epi_img_real_06 = nib.load(path  + file_4_re)
epi_img_real_06_r = epi_img_real_06.get_fdata()

#print(epi_img_data_real.shape)

epi_img_imag_06 = nib.load(path  + file_4_im)
epi_img_imag_06_i = epi_img_imag_06.get_fdata()






S_Mask = nib.load(r'C:\Users\musti\Desktop\Hampus_NEw_Data_MP2RAGE\ST2Modified_brain_mask_Hampus.nii.gz')
S_Mask_data = S_Mask.get_fdata()
#S_Mask_data.astype(np.int32)

#%%


# =============================================================================
# Go through the slices and Multiply by mask (MP2RAGE as Ref for Threshold)  and combine them into one Nii-file 
# =============================================================================      
images_segmented=[]
MP2RAGE_lista=[]

Scalefactor = 1 #4095
plus_factor  = 0 #0.5
MP2RAGE_Masked_17 = []
MP2RAGE_Masked_12 = []
MP2RAGE_Masked_09 = []
MP2RAGE_Masked_06 = []
for s in range(epi_img_real_17_r.shape[0]):
    
    Sreal_T1_17 = epi_img_real_17_r[s, :, :,0]    
    
    Sreal_T2_17 = epi_img_real_17_r[s, :, :,1]
    
    Simg_T1_17 = epi_img_imag_17_i[s, :, :,0]
    
    Simg_T2_17 = epi_img_imag_17_i[s, :, :,1]
    
    
    
    Sreal_T1_12 = epi_img_real_12_r[s, :, :,0]    
    
    Sreal_T2_12 = epi_img_real_12_r[s, :, :,1]
    
    Simg_T1_12 = epi_img_imag_12_i[s, :, :,0]
    
    Simg_T2_12 = epi_img_imag_12_i[s, :, :,1]
    
    
    
    Sreal_T1_09 = epi_img_real_09_r[s, :, :,0]    
    
    Sreal_T2_09 = epi_img_real_09_r[s, :, :,1]
    
    Simg_T1_09 = epi_img_imag_09_i[s, :, :,0]
    
    Simg_T2_09 = epi_img_imag_09_i[s, :, :,1]    
    
    
    Sreal_T1_06 = epi_img_real_06_r[s, :, :,0]    
    
    Sreal_T2_06 = epi_img_real_06_r[s, :, :,1]
    
    Simg_T1_06 = epi_img_imag_06_i[s, :, :,0]
    
    Simg_T2_06 = epi_img_imag_06_i[s, :, :,1]    
    
    
    
    
    
    #MP2RAGE_17 = ((((Sreal_T1_17*Sreal_T2_17) + (Simg_T1_17*Simg_T2_17))/((Sreal_T1_17**2) + (Sreal_T2_17**2) + Simg_T1_17**2 + Simg_T2_17**2)) + plus_factor) * Scalefactor * S_Mask_data[s,:,:] # ta bort skalle
    #MP2RAGE_12 = ((((Sreal_T1_12*Sreal_T2_12) + (Simg_T1_12*Simg_T2_12))/((Sreal_T1_12**2) + (Sreal_T2_12**2) + Simg_T2_12**2 + Simg_T1_12**2)) + plus_factor) * Scalefactor * S_Mask_data[s,:,:]
    #MP2RAGE_09 = ((((Sreal_T1_09*Sreal_T2_09) + (Simg_T1_09*Simg_T2_09))/((Sreal_T1_09**2) + (Sreal_T2_09**2) + Simg_T2_09**2 + Simg_T1_09**2)) + plus_factor) * Scalefactor * S_Mask_data[s,:,:]
    MP2RAGE_06 = ((((Sreal_T1_06*Sreal_T2_06) + (Simg_T1_06*Simg_T2_06))/((Sreal_T1_06**2) + (Sreal_T2_06**2) + Simg_T2_06**2 + Simg_T1_06**2)) + plus_factor) * Scalefactor * S_Mask_data[s,:,:]

    # MP2RAGE_Masked_17.append(MP2RAGE_17)        
    # MP2RAGE_Masked_12.append(MP2RAGE_12)     
    # MP2RAGE_Masked_09.append(MP2RAGE_09)     
    MP2RAGE_Masked_06.append(MP2RAGE_06)     
    
# vol_17 = np.stack((MP2RAGE_Masked_17))
# vol_12 = np.stack((MP2RAGE_Masked_12))
# vol_09 = np.stack((MP2RAGE_Masked_09))
vol_06 = np.stack((MP2RAGE_Masked_06))
#print(vol_09.shape)
        
        
# new_image_17 = nib.Nifti1Image(vol_17, epi_img_real_17.affine)
# new_image_12 = nib.Nifti1Image(vol_12, epi_img_real_17.affine)
# new_image_09 = nib.Nifti1Image(vol_09, epi_img_real_17.affine)
new_image_06 = nib.Nifti1Image(vol_06, epi_img_real_17.affine)




# nib.save(new_image_17, path + "\MP2RAGE_17_05.nii.gz")
# nib.save(new_image_12, path + "\MP2RAGE_12_05.nii.gz")
# nib.save(new_image_09, path + "\MP2RAGE_09_05.nii.gz")
nib.save(new_image_06, r'C:\Users\musti\Desktop\Hampus_NEw_Data_MP2RAGE\MP2RAGE_06_05.nii.gz')

#%%

# =============================================================================
# Få ST2 för sig själv för att sedan använda Bet (Skallefritt)
# =============================================================================


# epi_img_imag_06 = nib.load(r'C:\Users\musti\OneDrive\Skrivbord\Hampus_new_Data\Serie_1304_Imaginary.nii.gz')
# epi_img_imag_06_i = epi_img_imag_06.get_fdata()

# #print(epi_img_data_imag.shape)




# S_Mask = nib.load(r'C:\Users\musti\OneDrive\Skrivbord\LinuxNew\wetransfer-9fe4ac\ST2_Mask_Bin_mask.nii.gz')
# S_Mask_data = S_Mask.get_fdata()
#S_Mask_data.astype(np.int32)

#%%

path = r"C:\Users\musti\Desktop\Gunther_NEw_Data_MP2RAGE"
file_1_ST2 = "\MP2RAGE_adiabat_05.nii.gz"
file_2_ST2 = "\MP2RAGE_FOCI53_05.nii.gz"
file_3_ST2 = "\MP2RAGE_FOCI45_05.nii.gz"

S_Mask = nib.load(r'C:\Users\musti\Desktop\Gunther_NEw_Data_MP2RAGE\MP2RAGE_Gunther_ST2_FO_53_brain_mask.nii.gz')
S_Mask_data = S_Mask.get_fdata()



epi_img_imag_ST2 = nib.load(path + file_1_ST2)
epi_img_imag_ST2_st2_ad = epi_img_imag_ST2.get_fdata() 


epi_img_imag_ST2 = nib.load(path + file_2_ST2)
epi_img_imag_ST2_st2_fo_53 = epi_img_imag_ST2.get_fdata() 



epi_img_imag_ST2 = nib.load(path + file_3_ST2)
epi_img_imag_ST2_st2_fo_45 = epi_img_imag_ST2.get_fdata() 




# epi_img_imag_ST2 = nib.load(path + file_4_ST2)
# epi_img_imag_ST2_st2_fo_45 = epi_img_imag_ST2.get_fdata() 




# =============================================================================
# Go through the slices and Multiply by mask (MP2RAGE as Ref for Threshold)  and combine them into one Nii-file 
# =============================================================================      
images_segmented=[]
MP2RAGE_lista=[]

# Scalefactor = 4095

MP2RAGE_Masked_ST2_ad_1 = []
MP2RAGE_Masked_ST2_FO_53_1 = []
MP2RAGE_Masked_ST2_FO_45_1 = []

for s in range(epi_img_imag_ST2_st2_ad.shape[0]):
    
    MP2RAGE_Masked_ST2_ad = epi_img_imag_ST2_st2_ad[s, :, :] * S_Mask_data[s,:,:]
    #MP2RAGE_Masked_ST2_ad = ((MP2RAGE_Masked_ST2_ad - MP2RAGE_Masked_ST2_ad.min()) * (1/(MP2RAGE_Masked_ST2_ad.max() - MP2RAGE_Masked_ST2_ad.min()) * 255)/255)-0.5
        
    MP2RAGE_Masked_ST2_ad_1.append(MP2RAGE_Masked_ST2_ad)   

    MP2RAGE_Masked_ST2_FO_53 = epi_img_imag_ST2_st2_fo_53[s, :, :] * S_Mask_data[s,:,:]
    #MP2RAGE_Masked_ST2_FO_53 = ((MP2RAGE_Masked_ST2_FO_53 - MP2RAGE_Masked_ST2_FO_53.min()) * (1/(MP2RAGE_Masked_ST2_FO_53.max() - MP2RAGE_Masked_ST2_FO_53.min()) * 255)/255)-0.5    
        
    MP2RAGE_Masked_ST2_FO_53_1.append(MP2RAGE_Masked_ST2_FO_53)  

    MP2RAGE_Masked_ST2_FO_45 = epi_img_imag_ST2_st2_fo_45[s, :, :] * S_Mask_data[s,:,:]
    #MP2RAGE_Masked_ST2_FO_45 = ((MP2RAGE_Masked_ST2_FO_45 - MP2RAGE_Masked_ST2_FO_45.min()) * (1/(MP2RAGE_Masked_ST2_FO_45.max() - MP2RAGE_Masked_ST2_FO_45.min()) * 255)/255)-0.5    
    
    MP2RAGE_Masked_ST2_FO_45_1.append(MP2RAGE_Masked_ST2_FO_45)  
    
     
 
    
vol_ST2_ad = np.stack((MP2RAGE_Masked_ST2_ad_1))
vol_ST2_FO_53 = np.stack((MP2RAGE_Masked_ST2_FO_53_1))
vol_ST2_FO_45 = np.stack((MP2RAGE_Masked_ST2_FO_45_1))
print(vol_ST2_ad.shape)
        
        
new_image_17_ST2_ad = nib.Nifti1Image(vol_ST2_ad, epi_img_imag_ST2.affine)
new_image_17_ST2_fo_53 = nib.Nifti1Image(vol_ST2_FO_53, epi_img_imag_ST2.affine)
new_image_17_ST2_fo_45 = nib.Nifti1Image(vol_ST2_FO_45, epi_img_imag_ST2.affine)
nib.save(new_image_17_ST2_ad, path + "\MP2RAGE_Gunther_ST2_adi_05_masked.nii.gz")
nib.save(new_image_17_ST2_fo_53, path + "\MP2RAGE_Gunther_ST2_FO_53_05_masked.nii.gz")
nib.save(new_image_17_ST2_fo_45, path + "\MP2RAGE_Gunther_ST2_FO_45_05_masked.nii.gz")


#%%

import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage.measure import profile_line
from skimage import io


# =============================================================================
# Seperating B1+ maps to later be used in DREAM 
# =============================================================================


path = r"C:\Users\musti\Desktop\Gunther_NEw_Data_MP2RAGE"
file_1_ST2 = "\Serie_901_DREAM_25dg.nii.gz"
file_2_ST2 = "\Serie_1001_DREAM_40dg.nii.gz"
file_3_ST2 = "\Serie_1201_DREAM_90dg.nii.gz"
#file_4_ST2 = "\Serie_1201_DREAM_90dg.nii.gz"




epi_img_imag_ST2 = nib.load(path + file_1_ST2)
epi_img_imag_ST2_st2_ad = epi_img_imag_ST2.get_fdata() 


epi_img_imag_ST2 = nib.load(path + file_2_ST2)
epi_img_imag_ST2_st2_fo_53 = epi_img_imag_ST2.get_fdata() 



epi_img_imag_ST2 = nib.load(path + file_3_ST2)
epi_img_imag_ST2_st2_fo_45 = epi_img_imag_ST2.get_fdata() 




# epi_img_imag_ST2 = nib.load(path + file_4_ST2)
# epi_img_imag_ST2_st2_fo_45 = epi_img_imag_ST2.get_fdata() 




# =============================================================================
# Go through the slices and Multiply by mask (MP2RAGE as Ref for Threshold)  and combine them into one Nii-file 
# =============================================================================      
images_segmented=[]
MP2RAGE_lista=[]

# Scalefactor = 4095

MP2RAGE_Masked_ST2_ad_1 = []
MP2RAGE_Masked_ST2_FO_53_1 = []
MP2RAGE_Masked_ST2_FO_45_1 = []

for s in range(epi_img_imag_ST2_st2_fo_45.shape[0]):
    
    MP2RAGE_Masked_ST2_ad = epi_img_imag_ST2_st2_ad[s, :, :,2] 
    #MP2RAGE_Masked_ST2_ad = ((MP2RAGE_Masked_ST2_ad - MP2RAGE_Masked_ST2_ad.min()) * (1/(MP2RAGE_Masked_ST2_ad.max() - MP2RAGE_Masked_ST2_ad.min()) * 255)/255)-0.5
        
    MP2RAGE_Masked_ST2_ad_1.append(MP2RAGE_Masked_ST2_ad)   

    MP2RAGE_Masked_ST2_FO_53 = epi_img_imag_ST2_st2_fo_53[s, :, :,0]
    #MP2RAGE_Masked_ST2_FO_53 = ((MP2RAGE_Masked_ST2_FO_53 - MP2RAGE_Masked_ST2_FO_53.min()) * (1/(MP2RAGE_Masked_ST2_FO_53.max() - MP2RAGE_Masked_ST2_FO_53.min()) * 255)/255)-0.5    
        
    MP2RAGE_Masked_ST2_FO_53_1.append(MP2RAGE_Masked_ST2_FO_53)  

    MP2RAGE_Masked_ST2_FO_45 = epi_img_imag_ST2_st2_fo_45[s, :, :,2]    
    #MP2RAGE_Masked_ST2_FO_45 = ((MP2RAGE_Masked_ST2_FO_45 - MP2RAGE_Masked_ST2_FO_45.min()) * (1/(MP2RAGE_Masked_ST2_FO_45.max() - MP2RAGE_Masked_ST2_FO_45.min()) * 255)/255)-0.5    
    
    MP2RAGE_Masked_ST2_FO_45_1.append(MP2RAGE_Masked_ST2_FO_45)  
    
     
 
    
vol_ST2_ad = np.stack((MP2RAGE_Masked_ST2_ad_1))
vol_ST2_FO_53 = np.stack((MP2RAGE_Masked_ST2_FO_53_1))
vol_ST2_FO_45 = np.stack((MP2RAGE_Masked_ST2_FO_45_1))
print(vol_ST2_FO_45.shape)
        
        
new_image_17_ST2_ad = nib.Nifti1Image(vol_ST2_ad, epi_img_imag_ST2.affine)
new_image_17_ST2_fo_53 = nib.Nifti1Image(vol_ST2_FO_53, epi_img_imag_ST2.affine)
new_image_17_ST2_fo_45 = nib.Nifti1Image(vol_ST2_FO_45, epi_img_imag_ST2.affine)
# nib.save(new_image_17_ST2_ad, path + "\B1_plus_NewData_25_seperated_Gunther.nii.gz")
nib.save(new_image_17_ST2_fo_53, path + "\B1_plus_NewData_40ffs_seperated_Gunther.nii.gz")
# nib.save(new_image_17_ST2_fo_45, path + "\B1_plus_NewData_90_seperated_Gunther.nii.gz")


#%%



import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage.measure import profile_line
from skimage import io


B_plus = [0.4, 0.6, 0.8, 1.0, 1.2 , 1.4]

F_inv_B_max_6 = [0.28, 0.35, 0.365, 0.42, 0.437, 0.454]
F_inv_B_max_9 = [0.25, 0.34, 0.36, 0.4, 0.440, 0.449]
F_inv_B_max_12 = [0.23, 0.34, 0.35, 0.4, 0.434, 0.440]
F_inv_B_max_17 = [0.23, 0.33, 0.347, 0.399, 0.432, 0.448]

F_inv_B_max_6_error = [0.02, 0.027, 0.06, 0.043, 0.026, 0.02 ]
F_inv_B_max_9_error = [0.07, 0.06, 0.06, 0.022, 0.039, 0.058 ]
F_inv_B_max_12_error = [0.04, 0.029, 0.035, 0.02 ,0.026, 0.017 ]
F_inv_B_max_17_error = [0.05, 0.036, 0.05, 0.02, 0.024, 0.018  ]

# plt.plot(B_plus, F_inv_B_max_6, 'r-^', linewidth=0.6, label='B1-max = 6')
# plt.plot(B_plus, F_inv_B_max_9, 'g-^', linewidth=0.6, label='B1-max = 9')
# plt.plot(B_plus, F_inv_B_max_12, 'b-^', linewidth=0.6, label='B1-max = 12')
# plt.plot(B_plus, F_inv_B_max_17, 'k-^', linewidth=0.6, label='B1-max = 17')

plt.errorbar(B_plus, F_inv_B_max_6, yerr=F_inv_B_max_6_error, marker='^', color='red', label='B1max = 6')
plt.errorbar(B_plus, F_inv_B_max_9, yerr=F_inv_B_max_6_error, marker='^', color='blue', label='B1max = 9')
plt.errorbar(B_plus, F_inv_B_max_12, yerr=F_inv_B_max_6_error, marker='^', color='green', label='B1max = 12')
plt.errorbar(B_plus, F_inv_B_max_17, yerr=F_inv_B_max_6_error, marker='^', color='black', label='B1max = 17')
plt.title("B1max effect on MP2EAGE vs B1+")
plt.ylabel("MP2RAGE")
plt.xlabel("B1+")
plt.grid()
plt.legend()
plt.show()


#%%

# =============================================================================
# Creating B1-plus full of ones for Marques' code
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage.measure import profile_line
from skimage import io



path = r"C:\Users\musti\Desktop\T1mappped Data\oldData"
file_1_ST2 = "\MP2RAGE_Stacked0.5SkalleFree_Python.nii.gz"


epi_img_imag_ST2 = nib.load(path + file_1_ST2)
epi_img_imag_ST2_st2_ad = epi_img_imag_ST2.get_fdata() 


B1_ones  = np.ones((epi_img_imag_ST2_st2_ad.shape), dtype=np.int32)
print(B1_ones.shape)
        
        
new_image_17_ST2_ad = nib.Nifti1Image(B1_ones, epi_img_imag_ST2.affine)
nib.save(new_image_17_ST2_ad, path + "\OldData_B1_plus_Map_ones.nii.gz")


#%%
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage.measure import profile_line
from skimage import io


T1_corrected_06 = 1050.6
T1_corrected_09 = 1070
T1_corrected_12 = 1095
T1_corrected_17 = 1111.6


T1_corrected_06_err = 54
T1_corrected_09_err = 59
T1_corrected_12_err = 65
T1_corrected_17_err = 65


T1_corrected_06_B1ones = 872
T1_corrected_09_B1ones = 885
T1_corrected_12_B1ones = 899.6
T1_corrected_17_B1ones = 910

T1_corrected_06_B1ones_err = 34.2
T1_corrected_09_B1ones_err = 38
T1_corrected_12_B1ones_err = 41
T1_corrected_17_B1ones_err = 43.4


plt.errorbar([6,9,12,15],[T1_corrected_06, T1_corrected_09, T1_corrected_12,T1_corrected_17 ],
             yerr= [T1_corrected_06_err, T1_corrected_09_err, T1_corrected_12_err,T1_corrected_17_err]
             ,marker= 's',mfc='red', label='Local Flip Angle Map, finv = 0.96')

plt.errorbar([6,9,12,15],[ T1_corrected_06_B1ones, T1_corrected_09_B1ones, T1_corrected_12_B1ones, T1_corrected_17_B1ones ],
             yerr= [T1_corrected_06_B1ones_err, T1_corrected_09_B1ones_err, T1_corrected_12_B1ones_err, T1_corrected_17_B1ones_err]
             ,marker= 's', mec='k', label='Without Local Flip Angle Map, finv = 0.96')


plt.errorbar([6],[1190.5],
             yerr= [63.654
]
             ,marker= 's', label=' finv= 0.80')

plt.errorbar([6],[1154.605],
             yerr= [59.715
]
             ,marker= 's', label=' finv= 0.84')

plt.errorbar([6],[1121.765
],
             yerr= [56.259
]
             ,marker= 's', label=' finv= 0.88')

plt.errorbar([6],[1091.633
],
             yerr= [53.193
]
             ,marker= 's', label=' finv= 0.92')

plt.errorbar([6],[1050.6
],
             yerr= [50.429
]
             ,marker= 's', mfc='red', mec='blue')

plt.errorbar([6],[1038.201
],
             yerr= [47.943
]
             ,marker= 's', label=' finv= 1.0')






plt.title('T1-values vs B1-max influenced by local flip angles')
plt.xlabel('B1-max [uT]',fontsize=9, fontweight='bold')
plt.ylabel('T1-values [ms]', fontsize=9, fontweight='bold')
plt.grid(color = 'black' , linestyle = ':', linewidth = 0.7)
plt.legend()






#%%

# =============================================================================
# Needs to be corrected for FOCI 4-5
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage.measure import profile_line
from skimage import io


T1_corrected_adi_53 = 1138.146

T1_corrected_FOCI_53 = 1129.536

T1_corrected_FOCI_45 = 1314.509


T1_corrected_adi_53_err = 49.814

T1_corrected_FOCI_53_err = 53.535

T1_corrected_FOCI_45_err = 38.059




T1_corrected_adi_53_B1ones = 1029.815

T1_corrected_FOCI_53_B1ones = 1027.433

T1_corrected_FOCI_45_B1ones = 1175.882


T1_corrected_adi_53_B1ones_err = 39.005

T1_corrected_FOCI_53_B1ones_err = 40.722

T1_corrected_FOCI_45_B1ones_err = 28.252




plt.errorbar(['Full Adiabatic','FOCI 5-3','FOCI 4-5'],[T1_corrected_adi_53, T1_corrected_FOCI_53, T1_corrected_FOCI_45 ],
             yerr= [T1_corrected_adi_53_err, T1_corrected_FOCI_53_err, T1_corrected_FOCI_45_err]
             ,marker= 's',mfc='red', label='Local Flip Angle Map, finv = 0.96')


plt.errorbar(['Full Adiabatic','FOCI 5-3','FOCI 4-5'],[T1_corrected_adi_53_B1ones, T1_corrected_FOCI_53_B1ones, T1_corrected_FOCI_45_B1ones ],
              yerr= [T1_corrected_adi_53_B1ones_err, T1_corrected_FOCI_53_B1ones_err, T1_corrected_FOCI_45_B1ones_err]
              ,marker= 's', mec='k', label='Without Local Flip Angle Map,, finv = 0.96')


plt.errorbar(['FOCI 4-5'],[1525.177

],
             yerr= [50.913

]
             ,marker= 's', label=' finv= 0.8')

plt.errorbar(['FOCI 4-5'],[1463.732

],
             yerr= [47.943
]
             ,marker= 's', label=' finv= 0.84')

plt.errorbar(['FOCI 4-5'],[1408.789

],
             yerr= [43.943
]
             ,marker= 's', label=' finv= 0.88')

plt.errorbar(['FOCI 4-5'],[1359.321

],
             yerr= [40
]
             ,marker= 's', label=' finv= 0.92')

plt.errorbar(['FOCI 4-5'],[1273.697

],
             yerr= [35.796

]
             ,marker= 's', label=' finv= 1.0')




plt.title('T1-values vs Puls and Sequance types influenced by local flip angles')
plt.xlabel('Puls/Sequance',fontsize=9, fontweight='bold')
plt.ylabel('T1-values [ms]', fontsize=9, fontweight='bold')
plt.grid(color = 'black' , linestyle = ':', linewidth = 0.7)
plt.legend()
 











#%%

import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage.measure import profile_line
from skimage import io

f_inv = [1, 0.96, 0.92, 0.88, 0.84, 0.80 ]
T1_corrected_MP2RAGEold_1 = [1129.3]
T1_corrected_MP2RAGEold_096 = [1160.7]
T1_corrected_MP2RAGEold_092 = [1194.95]
T1_corrected_MP2RAGEold_088 = [1232.3]
T1_corrected_MP2RAGEold_084 = [1273.4]
T1_corrected_MP2RAGEold_080 = [1318.7]

T1_corrected_MP2RAGEold_1_err = [52.5]
T1_corrected_MP2RAGEold_096_err = [55.5]
T1_corrected_MP2RAGEold_092_err = [58.9]
T1_corrected_MP2RAGEold_088_err = [62.7]
T1_corrected_MP2RAGEold_084_err = [67.0]
T1_corrected_MP2RAGEold_080_err = [71.9]


T1_corrected_MP2RAGEold_1_B1ones = 963.5
T1_corrected_MP2RAGEold_096_B1ones  =987.4
T1_corrected_MP2RAGEold_092_B1ones  = 1013
T1_corrected_MP2RAGEold_088_B1ones  = 1041
T1_corrected_MP2RAGEold_084_B1ones  = 1071.3
T1_corrected_MP2RAGEold_080_B1ones  = 1104.5

T1_corrected_MP2RAGEold_1_B1ones_err = 37.2
T1_corrected_MP2RAGEold_096_B1ones_err  = 39.1
T1_corrected_MP2RAGEold_092_B1ones_err  = 41.2
T1_corrected_MP2RAGEold_088_B1ones_err  = 43.5
T1_corrected_MP2RAGEold_084_B1ones_err  = 46.1
T1_corrected_MP2RAGEold_080_B1ones_err  = 49.2

plt.errorbar(f_inv,[T1_corrected_MP2RAGEold_1[0], T1_corrected_MP2RAGEold_096[0], T1_corrected_MP2RAGEold_092[0],T1_corrected_MP2RAGEold_088[0], T1_corrected_MP2RAGEold_084[0], T1_corrected_MP2RAGEold_080[0]  ],
             yerr= [T1_corrected_MP2RAGEold_1_err[0], T1_corrected_MP2RAGEold_096_err[0], T1_corrected_MP2RAGEold_092_err[0],T1_corrected_MP2RAGEold_088_err[0], T1_corrected_MP2RAGEold_084_err[0],T1_corrected_MP2RAGEold_080_err[0]]
             ,marker= 's',mfc='red', label='Local Flip Angle Map')


plt.errorbar(f_inv,[ T1_corrected_MP2RAGEold_1_B1ones, T1_corrected_MP2RAGEold_096_B1ones, T1_corrected_MP2RAGEold_092_B1ones, T1_corrected_MP2RAGEold_088_B1ones, T1_corrected_MP2RAGEold_084_B1ones, T1_corrected_MP2RAGEold_080_B1ones   ],
             yerr= [T1_corrected_MP2RAGEold_1_B1ones_err, T1_corrected_MP2RAGEold_096_B1ones_err, T1_corrected_MP2RAGEold_092_B1ones_err, T1_corrected_MP2RAGEold_088_B1ones_err, T1_corrected_MP2RAGEold_084_B1ones_err, T1_corrected_MP2RAGEold_080_B1ones_err  ]

             ,marker= 's', mec='k', label='Without Local Flip Angle Map')


plt.title('T1-values vs inverstion efficiency influenced by local flip angles')
plt.xlabel('Assigned Inversion efficiency',fontsize=9, fontweight='bold')
plt.ylabel('Estimated T1-values [ms] ', fontsize=9, fontweight='bold')
plt.grid(color = 'black' , linestyle = ':', linewidth = 0.7)
plt.legend()


#%%

# =============================================================================
# different Volunteers/Test subjects
# =============================================================================


import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage.measure import profile_line
from skimage import io

f_inv = [1, 0.96, 0.92, 0.88, 0.84, 0.80 ]
T1_corrected_MP2RAGEold_1 = [1129.3]
T1_corrected_MP2RAGEold_096 = [1160.7]
T1_corrected_MP2RAGEold_092 = [1194.95]
T1_corrected_MP2RAGEold_088 = [1232.3]
T1_corrected_MP2RAGEold_084 = [1273.4]
T1_corrected_MP2RAGEold_080 = [1318.7]

T1_corrected_MP2RAGEold_1_err = [52.5]
T1_corrected_MP2RAGEold_096_err = [55.5]
T1_corrected_MP2RAGEold_092_err = [58.9]
T1_corrected_MP2RAGEold_088_err = [62.7]
T1_corrected_MP2RAGEold_084_err = [67.0]
T1_corrected_MP2RAGEold_080_err = [71.9]


T1_corrected_MP2RAGEold_1_B1ones = 963.5
T1_corrected_MP2RAGEold_096_B1ones  =987.4
T1_corrected_MP2RAGEold_092_B1ones  = 1013
T1_corrected_MP2RAGEold_088_B1ones  = 1041
T1_corrected_MP2RAGEold_084_B1ones  = 1071.3
T1_corrected_MP2RAGEold_080_B1ones  = 1104.5

T1_corrected_MP2RAGEold_1_B1ones_err = 37.2
T1_corrected_MP2RAGEold_096_B1ones_err  = 39.1
T1_corrected_MP2RAGEold_092_B1ones_err  = 41.2
T1_corrected_MP2RAGEold_088_B1ones_err  = 43.5
T1_corrected_MP2RAGEold_084_B1ones_err  = 46.1
T1_corrected_MP2RAGEold_080_B1ones_err  = 49.2



plt.errorbar(f_inv,[T1_corrected_MP2RAGEold_1[0], T1_corrected_MP2RAGEold_096[0], T1_corrected_MP2RAGEold_092[0],T1_corrected_MP2RAGEold_088[0], T1_corrected_MP2RAGEold_084[0], T1_corrected_MP2RAGEold_080[0]  ],
             yerr= [T1_corrected_MP2RAGEold_1_err[0], T1_corrected_MP2RAGEold_096_err[0], T1_corrected_MP2RAGEold_092_err[0],T1_corrected_MP2RAGEold_088_err[0], T1_corrected_MP2RAGEold_084_err[0],T1_corrected_MP2RAGEold_080_err[0]]
             ,marker= 'd',mfc='red', label='Local Flip Angle Map,Full Adiabatic Vol.1', markersize= 10)


plt.errorbar(f_inv,[ T1_corrected_MP2RAGEold_1_B1ones, T1_corrected_MP2RAGEold_096_B1ones, T1_corrected_MP2RAGEold_092_B1ones, T1_corrected_MP2RAGEold_088_B1ones, T1_corrected_MP2RAGEold_084_B1ones, T1_corrected_MP2RAGEold_080_B1ones   ],
             yerr= [T1_corrected_MP2RAGEold_1_B1ones_err, T1_corrected_MP2RAGEold_096_B1ones_err, T1_corrected_MP2RAGEold_092_B1ones_err, T1_corrected_MP2RAGEold_088_B1ones_err, T1_corrected_MP2RAGEold_084_B1ones_err, T1_corrected_MP2RAGEold_080_B1ones_err  ]

             ,marker= 'd', mec='k', label='Without Local Flip Angle Map, Full Adiabatic  Vol.1', markersize= 10)


T1_corrected_MP2RAGEold_080_Hampus = [1254.334]
T1_corrected_MP2RAGEold_084_Hampus = [1214.154]
T1_corrected_MP2RAGEold_088_Hampus = [1177.598]
T1_corrected_MP2RAGEold_092_Hampus = [1144.216]
T1_corrected_MP2RAGEold_096_Hampus = [1113.498]
T1_corrected_MP2RAGEold_1_Hampus = [1085.236]

T1_corrected_MP2RAGEold_080_err_Hampus = [81.588]
T1_corrected_MP2RAGEold_084_err_Hampus = [76.273]
T1_corrected_MP2RAGEold_088_err_Hampus = [71.622]
T1_corrected_MP2RAGEold_092_err_Hampus = [67.507]
T1_corrected_MP2RAGEold_096_err_Hampus = [63.858]
T1_corrected_MP2RAGEold_1_err_Hampus = [60.56]



T1_corrected_MP2RAGEold_080_B1ones_Hampus = 1002.305
T1_corrected_MP2RAGEold_084_B1ones_Hampus  =974.983
T1_corrected_MP2RAGEold_088_B1ones_Hampus  = 949.992
T1_corrected_MP2RAGEold_092_B1ones_Hampus  = 926.761
T1_corrected_MP2RAGEold_096_B1ones_Hampus  = 905.108
T1_corrected_MP2RAGEold_1_B1ones_Hampus  = 885.562



T1_corrected_MP2RAGEold_080_B1ones_err_Hampus = 47.469
T1_corrected_MP2RAGEold_084_B1ones_err_Hampus  = 44.988
T1_corrected_MP2RAGEold_088_B1ones_err_Hampus  = 42.585
T1_corrected_MP2RAGEold_092_B1ones_err_Hampus  = 40.441
T1_corrected_MP2RAGEold_096_B1ones_err_Hampus  = 38.73
T1_corrected_MP2RAGEold_1_B1ones_err_Hampus  = 36.676

plt.errorbar(f_inv,[T1_corrected_MP2RAGEold_1_Hampus[0], T1_corrected_MP2RAGEold_096_Hampus[0], T1_corrected_MP2RAGEold_092_Hampus[0],T1_corrected_MP2RAGEold_088_Hampus[0], T1_corrected_MP2RAGEold_084_Hampus[0], T1_corrected_MP2RAGEold_080_Hampus[0]  ],
             yerr= [T1_corrected_MP2RAGEold_1_err_Hampus[0], T1_corrected_MP2RAGEold_096_err_Hampus[0], T1_corrected_MP2RAGEold_092_err_Hampus[0],T1_corrected_MP2RAGEold_088_err_Hampus[0], T1_corrected_MP2RAGEold_084_err_Hampus[0],T1_corrected_MP2RAGEold_080_err_Hampus[0]]
             ,marker= 's',mfc='red', label='Local Flip Angle Map, Full Adiabatic  Vol.2', markersize= 10)


plt.errorbar(f_inv,[ T1_corrected_MP2RAGEold_1_B1ones_Hampus, T1_corrected_MP2RAGEold_096_B1ones_Hampus, T1_corrected_MP2RAGEold_092_B1ones_Hampus, T1_corrected_MP2RAGEold_088_B1ones_Hampus, T1_corrected_MP2RAGEold_084_B1ones_Hampus, T1_corrected_MP2RAGEold_080_B1ones_Hampus   ],
             yerr= [T1_corrected_MP2RAGEold_1_B1ones_err_Hampus, T1_corrected_MP2RAGEold_096_B1ones_err_Hampus, T1_corrected_MP2RAGEold_092_B1ones_err_Hampus, T1_corrected_MP2RAGEold_088_B1ones_err_Hampus, T1_corrected_MP2RAGEold_084_B1ones_err_Hampus, T1_corrected_MP2RAGEold_080_B1ones_err_Hampus  ]

             ,marker= 's', mec='k', label='Without Local Flip Angle Map, Full Adiabatic Vol.2', markersize= 10)



T1_corrected_MP2RAGEold_080_Gunther_FO53 = [1281.932]
T1_corrected_MP2RAGEold_084_Gunther_FO53 = [1238.24]
T1_corrected_MP2RAGEold_088_Gunther_FO53 = [1198.667]
T1_corrected_MP2RAGEold_092_Gunther_FO53 = [1162.563]
T1_corrected_MP2RAGEold_096_Gunther_FO53 = [1129.536]
T1_corrected_MP2RAGEold_1_Gunther_FO53 = [1099.208]



T1_corrected_MP2RAGEold_080_err_Gunther_FO53 = [69.05]
T1_corrected_MP2RAGEold_084_err_Gunther_FO53 = [64.398]
T1_corrected_MP2RAGEold_088_err_Gunther_FO53 = [60.283]
T1_corrected_MP2RAGEold_092_err_Gunther_FO53 = [56.725]
T1_corrected_MP2RAGEold_096_err_Gunther_FO53 = [53.535]
T1_corrected_MP2RAGEold_1_err_Gunther_FO53 = [50.689]


T1_corrected_MP2RAGEold_080_B1ones_Gunther_FO53 = 1155.17
T1_corrected_MP2RAGEold_084_B1ones_Gunther_FO53  =1118.752
T1_corrected_MP2RAGEold_088_B1ones_Gunther_FO53  = 1085.747
T1_corrected_MP2RAGEold_092_B1ones_Gunther_FO53 = 1055.321
T1_corrected_MP2RAGEold_096_B1ones_Gunther_FO53 = 1027.433
T1_corrected_MP2RAGEold_1_B1ones_Gunther_FO53  = 1001.627



T1_corrected_MP2RAGEold_080_B1ones_err_Gunther_FO53 = 51.702
T1_corrected_MP2RAGEold_084_B1ones_err_Gunther_FO53  = 48.463
T1_corrected_MP2RAGEold_088_B1ones_err_Gunther_FO53  = 45.468
T1_corrected_MP2RAGEold_092_B1ones_err_Gunther_FO53 = 43.064
T1_corrected_MP2RAGEold_096_B1ones_err_Gunther_FO53  = 40.722
T1_corrected_MP2RAGEold_1_B1ones_err_Gunther_FO53  = 38.748




plt.errorbar(f_inv,[T1_corrected_MP2RAGEold_1_Gunther_FO53[0], T1_corrected_MP2RAGEold_096_Gunther_FO53[0], T1_corrected_MP2RAGEold_092_Gunther_FO53[0],T1_corrected_MP2RAGEold_088_Gunther_FO53[0], T1_corrected_MP2RAGEold_084_Gunther_FO53[0], T1_corrected_MP2RAGEold_080_Gunther_FO53[0]  ],
             yerr= [T1_corrected_MP2RAGEold_1_err_Gunther_FO53[0], T1_corrected_MP2RAGEold_096_err_Gunther_FO53[0], T1_corrected_MP2RAGEold_092_err_Gunther_FO53[0],T1_corrected_MP2RAGEold_088_err_Gunther_FO53[0], T1_corrected_MP2RAGEold_084_err_Gunther_FO53[0],T1_corrected_MP2RAGEold_080_err_Gunther_FO53[0]]
             ,marker= 'h',mfc='green', label='Local Flip Angle Map, FOCI 53 Vol.3', markersize= 10)


plt.errorbar(f_inv,[ T1_corrected_MP2RAGEold_1_B1ones_Gunther_FO53, T1_corrected_MP2RAGEold_096_B1ones_Gunther_FO53, T1_corrected_MP2RAGEold_092_B1ones_Gunther_FO53, T1_corrected_MP2RAGEold_088_B1ones_Gunther_FO53, T1_corrected_MP2RAGEold_084_B1ones_Gunther_FO53, T1_corrected_MP2RAGEold_080_B1ones_Gunther_FO53  ],
             yerr= [T1_corrected_MP2RAGEold_1_B1ones_err_Gunther_FO53, T1_corrected_MP2RAGEold_096_B1ones_err_Gunther_FO53, T1_corrected_MP2RAGEold_092_B1ones_err_Gunther_FO53, T1_corrected_MP2RAGEold_088_B1ones_err_Gunther_FO53, T1_corrected_MP2RAGEold_084_B1ones_err_Gunther_FO53, T1_corrected_MP2RAGEold_080_B1ones_err_Gunther_FO53  ]

             ,marker= 'h', mec='k', label='Without Local Flip Angle Map, FOCI 53 Vol.3', markersize= 10)



T1_corrected_MP2RAGEold_080_Gunther_ADI53 = [1292.85]
T1_corrected_MP2RAGEold_084_Gunther_ADI53 = [1248.489]
T1_corrected_MP2RAGEold_088_Gunther_ADI53 = [1208.281]
T1_corrected_MP2RAGEold_092_Gunther_ADI53 = [1171.622]
T1_corrected_MP2RAGEold_096_Gunther_ADI53 = [1138.146]
T1_corrected_MP2RAGEold_1_Gunther_ADI53= [1107.359]


T1_corrected_MP2RAGEold_080_err_Gunther_ADI53 = [64.62]
T1_corrected_MP2RAGEold_084_err_Gunther_ADI53 = [60.148]
T1_corrected_MP2RAGEold_088_err_Gunther_ADI53 = [56.263]
T1_corrected_MP2RAGEold_092_err_Gunther_ADI53 = [52.885]
T1_corrected_MP2RAGEold_096_err_Gunther_ADI53 = [49.814]
T1_corrected_MP2RAGEold_1_err_Gunther_ADI53= [47.166]


T1_corrected_MP2RAGEold_080_B1ones_Gunther_ADI53 = 1158.175
T1_corrected_MP2RAGEold_084_B1ones_Gunther_ADI53  =1121.566
T1_corrected_MP2RAGEold_088_B1ones_Gunther_ADI53  = 1088.403
T1_corrected_MP2RAGEold_092_B1ones_Gunther_ADI53 =1057.834
T1_corrected_MP2RAGEold_096_B1ones_Gunther_ADI53 = 1029.815
T1_corrected_MP2RAGEold_1_B1ones_Gunther_ADI53 = 1003.888









T1_corrected_MP2RAGEold_080_B1ones_err_Gunther_ADI53 = 49.622
T1_corrected_MP2RAGEold_084_B1ones_err_Gunther_ADI53 = 46.493
T1_corrected_MP2RAGEold_088_B1ones_err_Gunther_ADI53  = 43.586
T1_corrected_MP2RAGEold_092_B1ones_err_Gunther_ADI53 = 41.28
T1_corrected_MP2RAGEold_096_B1ones_err_Gunther_ADI53 = 39.005
T1_corrected_MP2RAGEold_1_B1ones_err_Gunther_ADI53 = 37.123


plt.errorbar(f_inv,[T1_corrected_MP2RAGEold_1_Gunther_ADI53[0], T1_corrected_MP2RAGEold_096_Gunther_ADI53[0], T1_corrected_MP2RAGEold_092_Gunther_ADI53[0],T1_corrected_MP2RAGEold_088_Gunther_ADI53[0], T1_corrected_MP2RAGEold_084_Gunther_ADI53[0], T1_corrected_MP2RAGEold_080_Gunther_ADI53[0]  ],
             yerr= [T1_corrected_MP2RAGEold_1_err_Gunther_ADI53[0], T1_corrected_MP2RAGEold_096_err_Gunther_ADI53[0], T1_corrected_MP2RAGEold_092_err_Gunther_ADI53[0],T1_corrected_MP2RAGEold_088_err_Gunther_ADI53[0], T1_corrected_MP2RAGEold_084_err_Gunther_ADI53[0],T1_corrected_MP2RAGEold_080_err_Gunther_ADI53[0]]
             ,marker= '^',mfc='y', label='Local Flip Angle Map, Adi-53 Vol.3', markersize= 10)


plt.errorbar(f_inv,[ T1_corrected_MP2RAGEold_1_B1ones_Gunther_ADI53, T1_corrected_MP2RAGEold_096_B1ones_Gunther_ADI53, T1_corrected_MP2RAGEold_092_B1ones_Gunther_ADI53, T1_corrected_MP2RAGEold_088_B1ones_Gunther_ADI53, T1_corrected_MP2RAGEold_084_B1ones_Gunther_ADI53, T1_corrected_MP2RAGEold_080_B1ones_Gunther_ADI53  ],
             yerr= [T1_corrected_MP2RAGEold_1_B1ones_err_Gunther_ADI53, T1_corrected_MP2RAGEold_096_B1ones_err_Gunther_ADI53, T1_corrected_MP2RAGEold_092_B1ones_err_Gunther_ADI53, T1_corrected_MP2RAGEold_088_B1ones_err_Gunther_ADI53, T1_corrected_MP2RAGEold_084_B1ones_err_Gunther_ADI53, T1_corrected_MP2RAGEold_080_B1ones_err_Gunther_ADI53  ]

             ,marker= '^', mec='k', label='Without Local Flip Angle Map, Adi-53 Vol.3', markersize= 10)



plt.title('T1-values vs inverstion efficiency influenced by local flip angles', fontsize=12, fontweight='bold')
plt.xlabel('Assigned Inversion efficiency',fontsize=12, fontweight='bold')
plt.ylabel('Estimated T1-values [ms] ', fontsize=12, fontweight='bold')
plt.grid(color = 'black' , linestyle = ':', linewidth = 0.7)
plt.legend()


#%%

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

labels = ['0.80', '0.84', '0.88', '0.92', '0.96','1.0']
T1_values = [1190.569, 1154.605, 1121.765, 1091.633, 1050.6,1038.201]
T1_goal = 1200
errors =[63.654,59.715,56.259,53.193,50.429,47.943]

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars


plt.bar(labels , T1_values, width, color=['black', 'red', 'green', 'blue', 'cyan','m'],edgecolor='black', yerr=errors, alpha= 0.5)
plt.plot([-0.50,6],[1200,1200], 'r--')

plt.title('T1-values influenced by Assigned Inversion Efficency',fontsize=12, fontweight='bold')
plt.xlim((-0.5, 5.5))
plt.xlabel('Assigned Inversion Efficency',fontsize=12, fontweight='bold')
plt.ylabel('Estimated T1-values [ms]',fontsize=12, fontweight='bold')
plt.legend()


