
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

path = r"C:\Users\musti\OneDrive\Skrivbord\Gunther_New_Data"
file_1_re = "\Serie_603_Real.nii.gz"
file_1_im = "\Serie_604_Imaginary.nii.gz"
file_2_re = "\Serie_703_Real.nii.gz"
file_2_im = "\Serie_704_Imaginary.nii.gz"
file_3_re = "\Serie_803_Real.nii.gz"
file_3_im = "\Serie_804_Imaginary.nii.gz"
# file_4_re = "\Serie_804_Imaginary.nii.gz"
# file_4_im = "\Serie_804_Imaginary.nii.gz"


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


# epi_img_real_06 = nib.load(path  + file_4_re)
# epi_img_real_06_r = epi_img_real_06.get_fdata()

# #print(epi_img_data_real.shape)

# epi_img_imag_06 = nib.load(path  + file_4_re)
# epi_img_imag_06_i = epi_img_imag_06.get_fdata()

#print(epi_img_data_imag.shape)




# S_Mask = nib.load(r'C:\Users\musti\OneDrive\Skrivbord\LinuxNew\wetransfer-9fe4ac\ST2_Mask_Bin_mask.nii.gz')
# S_Mask_data = S_Mask.get_fdata()
#S_Mask_data.astype(np.int32)

#%%


# =============================================================================
# Go through the slices and Multiply by mask (MP2RAGE as Ref for Threshold)  and combine them into one Nii-file 
# =============================================================================      
images_segmented=[]
MP2RAGE_lista=[]

Scalefactor = 4095

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
    
    
    # Sreal_T1_06 = epi_img_real_06_r[s, :, :,0]    
    
    # Sreal_T2_06 = epi_img_real_06_r[s, :, :,1]
    
    # Simg_T1_06 = epi_img_imag_06_i[s, :, :,0]
    
    # Simg_T2_06 = epi_img_imag_06_i[s, :, :,1]    
    
    
    
    
    
    MP2RAGE_17 = ((((Sreal_T1_17*Sreal_T2_17) + (Simg_T1_17*Simg_T2_17))/((Sreal_T1_17**2) + (Sreal_T2_17**2) + Simg_T1_17**2 + Simg_T2_17**2)) + 0.5) * Scalefactor  # ta bort skalle
    MP2RAGE_12 = ((((Sreal_T1_12*Sreal_T2_12) + (Simg_T1_12*Simg_T2_12))/((Sreal_T1_12**2) + (Sreal_T2_12**2) + Simg_T2_12**2 + Simg_T1_12**2)) + 0.5) * Scalefactor 
    MP2RAGE_09 = ((((Sreal_T1_09*Sreal_T2_09) + (Simg_T1_09*Simg_T2_09))/((Sreal_T1_09**2) + (Sreal_T2_09**2) + Simg_T2_09**2 + Simg_T1_09**2)) + 0.5) * Scalefactor
    #MP2RAGE_06 = ((((Sreal_T1_06*Sreal_T2_06) + (Simg_T1_06*Simg_T2_06))/((Sreal_T1_06**2) + (Sreal_T2_06**2) + Simg_T2_06**2 + Simg_T1_06**2)) + 0.5) * Scalefactor #* S_Mask_data[s,:,:]

    MP2RAGE_Masked_17.append(MP2RAGE_17)        
    MP2RAGE_Masked_12.append(MP2RAGE_12)     
    MP2RAGE_Masked_09.append(MP2RAGE_09)     
    #MP2RAGE_Masked_06.append(MP2RAGE_06)     
    
vol_17 = np.stack((MP2RAGE_Masked_17))
vol_12 = np.stack((MP2RAGE_Masked_12))
vol_09 = np.stack((MP2RAGE_Masked_09))
#vol_06 = np.stack((MP2RAGE_Masked_06))
print(vol_17.shape)
        
        
new_image_17 = nib.Nifti1Image(vol_17, epi_img_real_17.affine)
new_image_12 = nib.Nifti1Image(vol_12, epi_img_real_17.affine)
new_image_09 = nib.Nifti1Image(vol_09, epi_img_real_17.affine)
#new_image_06 = nib.Nifti1Image(vol_06, epi_img_real_17.affine)




nib.save(new_image_17, path + "\MP2RAGE_adiabat.nii.gz")
nib.save(new_image_12, path + "\MP2RAGE_FOCI53.nii.gz")
nib.save(new_image_09, path + "\MP2RAGE_FOCI45.nii.gz")
#nib.save(new_image_06, r'C:\Users\musti\OneDrive\Skrivbord\Hampus_NEw_Data_MP2RAGE\MP2RAGE_06.nii.gz')

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

path = r"C:\Users\musti\OneDrive\Skrivbord\Gunther_New_Data"
file_1_ST2 = "\Serie_601_MP2RAGE_adiabaticfull_TFE256Z_FF.nii.gz"
file_2_ST2 = "\Serie_701_MP2RAGE_TRFOCI2230dg_FA5-3dg_TFE256Z_FF.nii.gz"
file_3_ST2 = "\Serie_801_MP2RAGE_TRFOCI2230dg_FA4-5dg_TFE256Z_FF.nii.gz"
#file_4_ST2 = "\Serie_804_Imaginary.nii.gz"



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

Scalefactor = 4095

MP2RAGE_Masked_ST2_ad_1 = []
MP2RAGE_Masked_ST2_FO_53_1 = []
MP2RAGE_Masked_ST2_FO_45_1 = []

for s in range(epi_img_real_17_r.shape[0]):
    
    MP2RAGE_Masked_ST2_ad = epi_img_imag_ST2_st2_ad[s, :, :,1]    
        
    MP2RAGE_Masked_ST2_ad_1.append(MP2RAGE_Masked_ST2_ad)   

    MP2RAGE_Masked_ST2_FO_53 = epi_img_imag_ST2_st2_fo_53[s, :, :,1]    
        
    MP2RAGE_Masked_ST2_FO_53_1.append(MP2RAGE_Masked_ST2_FO_53)  

    MP2RAGE_Masked_ST2_FO_45 = epi_img_imag_ST2_st2_fo_45[s, :, :,1]    
        
    MP2RAGE_Masked_ST2_FO_45_1.append(MP2RAGE_Masked_ST2_FO_45)  

     
 
    
vol_ST2_ad = np.stack((MP2RAGE_Masked_ST2_ad_1))
vol_ST2_FO_53 = np.stack((MP2RAGE_Masked_ST2_FO_53_1))
vol_ST2_FO_45 = np.stack((MP2RAGE_Masked_ST2_FO_45_1))
print(vol_ST2_ad.shape)
        
        
new_image_17_ST2_ad = nib.Nifti1Image(vol_ST2_ad, epi_img_imag_ST2.affine)
new_image_17_ST2_fo_53 = nib.Nifti1Image(vol_ST2_FO_53, epi_img_imag_ST2.affine)
new_image_17_ST2_fo_45 = nib.Nifti1Image(vol_ST2_FO_45, epi_img_imag_ST2.affine)
nib.save(new_image_17_ST2_ad, path + "\MP2RAGE_Gunther_ST2_adi.nii.gz")
nib.save(new_image_17_ST2_fo_53, path + "\MP2RAGE_Gunther_ST2_FO_53.nii.gz")
nib.save(new_image_17_ST2_fo_45, path + "\MP2RAGE_Gunther_ST2_FO_45.nii.gz")


