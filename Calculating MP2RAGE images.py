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



path = r"C:\Users\musti\Desktop\Gunther_NEw_Data_MP2RAGE"
file_1_ST2 = "\Shadow_Gunther_53.nii_shadowreg_DREAM_combined_Gunther.nii.gz"


epi_img_imag_ST2 = nib.load(path + file_1_ST2)
epi_img_imag_ST2_st2_ad = epi_img_imag_ST2.get_fdata() 


B1_ones  = np.ones((epi_img_imag_ST2_st2_ad.shape), dtype=np.int32)
print(B1_ones.shape)
        
        
new_image_17_ST2_ad = nib.Nifti1Image(B1_ones, epi_img_imag_ST2.affine)
nib.save(new_image_17_ST2_ad, path + "\B1_plus_Map_ones.nii.gz")


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


plt.errorbar([6,9,12,17],[T1_corrected_06, T1_corrected_09, T1_corrected_12,T1_corrected_17 ],
             yerr= [T1_corrected_06_err, T1_corrected_09_err, T1_corrected_12_err,T1_corrected_17_err]
             ,marker= 's',mfc='red', label='Local Flip Angle Map')


plt.errorbar([6,9,12,17],[ T1_corrected_06_B1ones, T1_corrected_09_B1ones, T1_corrected_12_B1ones, T1_corrected_17_B1ones ],
             yerr= [T1_corrected_06_B1ones_err, T1_corrected_09_B1ones_err, T1_corrected_12_B1ones_err, T1_corrected_17_B1ones_err]
             ,marker= 's', mec='k', label='Without Local Flip Angle Map')


plt.title('T1-values vs B1-max influenced by local flip angles')
plt.xlabel('B1-max [uT]',fontsize=9, fontweight='bold')
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


T1_corrected_adi_53 = 1130.5
T1_corrected_FOCI_53 = 1137
T1_corrected_FOCI_45 = 1100

T1_corrected_adi_53_err = 53.5
T1_corrected_FOCI_53_err = 59.9
T1_corrected_FOCI_45_err = 57.2



T1_corrected_adi_53_B1ones = 1030.4
T1_corrected_FOCI_53_B1ones = 1035
T1_corrected_FOCI_45_B1ones = 1070.9

T1_corrected_adi_53_B1ones_err = 44
T1_corrected_FOCI_53_B1ones_err = 50.4
T1_corrected_FOCI_45_B1ones_err = 60.05



plt.errorbar(['Full Adiabatic','FOCI 5-3','FOCI 4-5'],[T1_corrected_adi_53, T1_corrected_FOCI_53, T1_corrected_FOCI_45 ],
             yerr= [T1_corrected_adi_53_err, T1_corrected_FOCI_53_err, T1_corrected_FOCI_45_err]
             ,marker= 's',mfc='red', label='Local Flip Angle Map')


plt.errorbar(['Full Adiabatic','FOCI 5-3','FOCI 4-5'],[T1_corrected_adi_53_B1ones, T1_corrected_FOCI_53_B1ones, T1_corrected_FOCI_45_B1ones ],
              yerr= [T1_corrected_adi_53_B1ones_err, T1_corrected_FOCI_53_B1ones_err, T1_corrected_FOCI_45_B1ones_err]
              ,marker= 's', mec='k', label='Without Local Flip Angle Map')


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
plt.xlabel('Inversion efficiency',fontsize=9, fontweight='bold')
plt.ylabel('T1-values [ms] ', fontsize=9, fontweight='bold')
plt.grid(color = 'black' , linestyle = ':', linewidth = 0.7)
plt.legend()


