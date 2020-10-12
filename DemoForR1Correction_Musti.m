% Scripts to remove residual B1 bias from T1 maps calculated with the
% MP2RAGE sequence
% Correction of T1 estimations as sugested on:
% Marques, J.P., Gruetter, R., 2013. New Developments and Applications of the MP2RAGE Sequence - Focusing the Contrast and High Spatial Resolution R1 Mapping. PLoS ONE 8. doi:10.1371/journal.pone.0069294
%
% in this script it is assumed that the B1 maps have been coregistered to the
% space of the MP2RAGE image and that they now have the B1 has in the
% process been interpolated to the same resolution.
% 
% 
%
addpath(genpath('.'))
 
%% Sa2RAGE protocol info and loading the Sa2RAGE data for B1 estimation

    Sa2RAGE.TR=2.4;
    Sa2RAGE.TRFLASH=2.75e-3;
    Sa2RAGE.TIs=[47e-3 1800e-3];
    Sa2RAGE.NZslices=128.*[0.25 0.5]./3+[6 6];% Base Resolution * [PartialFourierInPE-0.5  0.5]/iPATpe+[RefLines/2 RefLines/2]*(1-1/iPATpe )
    Sa2RAGE.FlipDegrees=[4 10];
    Sa2RAGE.averageT1=1.5;
    %Sa2RAGE.B1filename='C:\Users\musti\AppData\Local\Temp\Temp1_wetransfer-b5343e.zip\New folder\DREAM_combined.nii.gz';
    %B1=load_untouch_nii(Sa2RAGE.B1filename);
    
    
%% MP2RAGE protocol info and loading the MP2RAGE dataset 
    %path = 'C:\Users\Hampus\Documents\Linux\MP2RAGE\191003_GuHe\';  %200820 - HO
    path = 'C:\Users\musti\AppData\Local\Temp\Temp1_wetransfer-b5343e.zip\New folder';
    MP2RAGE.B0=7;           % in Tesla
    MP2RAGE.TR=5;           % MP2RAGE TR in seconds 
    MP2RAGE.TRFLASH=6.8e-3; % TR of the GRE readout
    MP2RAGE.TIs=[0.900 2.750];% inversion times - time between middle of refocusing pulse and excitatoin of the k-space center encoding
    MP2RAGE.NZslices=[256];% Slices Per Slab * [PartialFourierInSlice-0.5  0.5]
    MP2RAGE.FlipDegrees=[5 3];% Flip angle of the two readouts in degrees
    %MP2RAGE.filename='C:\Users\Hampus\Documents\Linux\MP2RAGE\191003_GuHe\Serie_803_DelRecReal_MP2RAGE.nii.gz' % file
    MP2RAGE.filename='C:\Users\musti\AppData\Local\Temp\Temp1_wetransfer-b5343e.zip\New folder\MP2RAGE_Stacked0.5SkalleFree_Python.nii.gz';
    
    % check the properties of this MP2RAGE protocol... 

    plotMP2RAGEproperties(MP2RAGE)    %200819 - HO

    % load the MP2RAGE data
    MP2RAGEimg=load_untouch_nii(MP2RAGE.filename);
    
    
    
%% performing the correction    
%[ B1corr T1corrected MP2RAGEcorr] = T1B1correctpackage( [],B1,Sa2RAGE,MP2RAGEimg,[],MP2RAGE,[],0.96)  %200819 - HO
    % saving the data
%     save_untouch_nii(MP2RAGEcorr,'data/MP2RAGEcorr.nii')                                                  %200819 - HO
%     save_untouch_nii(T1corrected,'data/T1corrected.nii')                                                  %200819 - HO
%% if another technique was used to obtain the relative B1 maps 
%  (1 means correctly calibrated B1)
B1=load_untouch_nii('C:\Users\musti\AppData\Local\Temp\Temp1_wetransfer-b5343e.zip\New folder\Serie_1001_DREAM_60dg_REGIS_shadowreg_DREAM_combined.nii.gz');
%     B1.img=double(B1.img)/1000;   %200819 - HO
B1.img=double(B1.img)/100;          %200819 - HO
%brain = load_untouch_nii('C:\Users\musti\AppData\Local\Temp\Temp1_wetransfer-b5343e.zip\New folder\MP2RAGE_StackedINTEN4095_Python.nii.gz');
%brain.img = double(brain.img);      %200820 - HO

[ T1corrected , MP2RAGEcorr] = T1B1correctpackageTFL(B1,MP2RAGEimg,[],MP2RAGE,[],0.96)

    
    % saving the data
save_untouch_nii(MP2RAGEcorr,'C:\Users\musti\AppData\Local\Temp\Temp1_wetransfer-b5343e.zip\New folder\MP2RAGEcorrMustiMatlabbb.nii.gz') 
save_untouch_nii(T1corrected,'C:\Users\musti\AppData\Local\Temp\Temp1_wetransfer-b5343e.zip\New folder\T1corrected_Marques_Musti.nii.gz')
    