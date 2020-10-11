# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 12:37:05 2020

@author: musti
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
start_time = time.time()



RAGE_1 = int(6.8*256) #6.9 * Nz_slice(160) * 6/8
RAGE_2 = int(6.8*256)

cycle = 5000#8000

flip_1 = 5#4
flip_2 = 3#5
Echo_factor = 6.8#6.9
flip_ang_1 = flip_1 *(np.pi/180)
flip_ang_2 = flip_2 *(np.pi/180)


slices = 256 #160
TI_1 = 900 #1000
TI_2 = 2750 #3300
Pulsint = TI_2 - TI_1
epsi = 0.00000001

T1_csf =3900  #ms
T1_gm = 1800 #ms
T1_wm = 1200

f_inv_list = np.round(np.linspace(0.80,1,6), 3)


X1 = abs(TI_1 - RAGE_1//2)

X2 =  Pulsint - 0.5*(RAGE_1 + RAGE_2)

X3= cycle - (RAGE_2/2 + TI_2)

T_1_list =[ T1_csf, T1_gm,T1_wm ]
lession = ['CSF', 'GM', 'WM']

M_csf = np.zeros(cycle)
M_gm = np.zeros(cycle)
M_wm = np.zeros(cycle)
M_tissue = [M_wm]

#tissue = 2

B_plus = np.sort( [1, 1.2 ,1.4 , 0.4, 0.6,0.8])
alpha_1 = [flip_ang_1 * i for i in B_plus]
alpha_2= [flip_ang_2 * i for i in B_plus]

Iterations = 10

f_inv = 0.96 #0.94

MP2RAGE_intens_04 = []
MP2RAGE_intens_06 = []
MP2RAGE_intens_08 = []
MP2RAGE_intens_1 = []
MP2RAGE_intens_1_2 = []
MP2RAGE_intens_1_4 = []

start= 600
end= 4200
jump = 100




#%%
for T1_loop in range(start,end,jump):
    y_K1=[]
    y_K2=[]
    for Bplus in range(len(alpha_1)):
        M0_2 = [1]    
        for m in range(len(M0_2)+Iterations):       
    #------------------
        
          
          # Plottar tiden som finns först i listan 
            x_RAGE_1 = []
            for i in range(1,slices):
                x_RAGE_1.append((i*epsi + (i-1)*6.8) + X1)
                for j in range(1):
                    x_RAGE_1.append(i *(epsi + 6.8) + X1)
            
            # the k1 x-point is at np.round(x_RAGE_1[len(x_RAGE_1)//2], 0) = 900
            #------------------ X1 -----------------------
            
            def Mz(t,T1):
                            
                return f_inv * M0_2[m] * (1-2*np.exp(-t/T1_loop))        
            
            X1_values = [Mz(t,T1_csf) for t in range(X1)]
            Normal_values = [ f_inv * M0_2[0] * (1-2*np.exp(-t/T1_loop)) for t in range(int(X1+ RAGE_1 + X2 + RAGE_2 + X3))]
            
            #------------------ RAGE 1 -----------------------
            
            
            M_RAGE_1=np.zeros(len(x_RAGE_1))
            M_RAGE_1[0]= X1_values[-1]
                        
                        
            for j in range(1,len(x_RAGE_1)-1,2):
                M_RAGE_1[j] = M_RAGE_1[j-1] * np.cos(alpha_1[Bplus])
                for k in range(1):
                    M_RAGE_1[j+k+1] = M_RAGE_1[j] * np.exp(-Echo_factor/T1_loop) + (1-np.exp(-Echo_factor/T1_loop))
                        
            
            
            #------------------ X2 -----------------------
            
            
            M_X_2 = np.zeros(int(X2))
            for j in range(0,len(M_X_2)):
                M_X_2[j] = M_RAGE_1[-2] * np.exp(-j/T1_loop) + (1-np.exp(-j/T1_loop)) #tar sista värdet av zikzak 
                        
            t_X_2 = np.arange( np.round(X1 + RAGE_1 - Echo_factor, 0) , np.round(X1 + RAGE_1 - Echo_factor + X2, 0) )
            
            
            
            #------------------ RAGE 2 -----------------------
            if len(M_X_2) != 0 :
                M_RAGE_2 = np.zeros(len(x_RAGE_1))
                M_RAGE_2[0] = M_X_2[-1]
                                
                for i in range(1,len(M_RAGE_2)-1,2):
                    M_RAGE_2[i] = M_RAGE_2[i-1] * np.cos(alpha_2[Bplus])
                    for j in range(1):
                        M_RAGE_2[i+j+1] = M_RAGE_2[i] * np.exp(-Echo_factor/T1_loop) + (1-np.exp(-Echo_factor/T1_loop))
                            
            
            else:
                M_RAGE_2 = np.zeros(len(x_RAGE_1))
                M_RAGE_2[0] = M_RAGE_1[-2]
                                
                for i in range(1,len(M_RAGE_2)-1,2):
                    M_RAGE_2[i] = M_RAGE_2[i-1] * np.cos(alpha_2[Bplus])
                    for j in range(1):
                        M_RAGE_2[i+j+1] = M_RAGE_2[i] * np.exp(-Echo_factor/T1_loop) + (1-np.exp(-Echo_factor/T1_loop))
                            
            
            
            #------------------ X3 -----------------------
            
            M_X_3 = np.zeros(int(X3))
            for j in range(0,len(M_X_3)):
                M_X_3[j] = M_RAGE_2[-2] * np.exp(-j/T1_loop) + (1-np.exp(-j/T1_loop)) #tar sista värdet av zikzak 
                        
            t_X_3 = np.arange( np.round(X1 + RAGE_1 - Echo_factor + X2 + RAGE_2 - Echo_factor, 0)  , 
                              np.round(X1 + RAGE_1 - Echo_factor + X2 + RAGE_2 - Echo_factor + X3-1, 0))
        
            M0_2.append(M_X_3[-1])
        
        x_K1 = np.round(X1+ RAGE_1//2,0)
        x_K2 = np.round(X1+ RAGE_1 + X2 + RAGE_2//2,0)
        
    
        
        y_K1.append(M_RAGE_1[len(x_RAGE_1[:-1])//2])
        y_K2.append(M_RAGE_2[len(x_RAGE_1[:-1])//2 + 1])
        
        Colors = ['#ff7f0e', '#1f77b4', '#d62728', '#2ca02c', '#e377c2', '#8c564b', '#e377c2', '#bcbd22', '#bcbd22', '#d62728' '#458b74', '#458b74']
        Colors2 = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf' '#458b74', '#d62728']
            
        #------------------ Plotting The Data -----------------------
        
        
        # plt.plot(np.arange(len(X1_values)) , X1_values, '--', linewidth=0.5, label='X1')
        
        # plt.plot(x_RAGE_1[:-1] , M_RAGE_1[:-1], '--', linewidth=0.5, label='RAGE 1 {}'.format(B_plus[Bplus]), color='{}'.format(Colors[Bplus]))
        
        # plt.plot(np.round(X1+ RAGE_1//2,0), M_RAGE_1[len(x_RAGE_1[:-1])//2] ,'*', label = 'k1')
        
        # plt.plot([i + np.round( RAGE_1 - Echo_factor + X2, 0) for i in x_RAGE_1[:-1]] , M_RAGE_2[:-1], '--', linewidth=0.5, label='RAGE 2 {}'.format(B_plus[Bplus]), color='{}'.format(Colors2[Bplus]))
        
        # plt.plot(t_X_2, M_X_2, '--', label='X2', linewidth=0.5)
        
        # plt.plot(t_X_3, M_X_3, 'm--', label='X3', linewidth=0.5)
        
        
        
        # plt.plot(np.round(X1+ RAGE_1 + X2 + RAGE_2//2,0), M_RAGE_2[len(x_RAGE_1[:-1])//2 + 1] ,'*', label = 'k2')
    
    
#    plt.plot(np.arange(X1+ RAGE_1 + X2 + RAGE_2 + X3-1), Normal_values, 'b-.', linewidth=1, label='Normal {} T1 relaxation'.format(lession[tissue]))
    
    #=============================================================================
    
    # fig1 = plt.figure(1)
    # x0 = [0, cycle]
    # y0 = [0, 0]
    # plt.plot(x0,y0, 'k--', linewidth=0.5)
    # plt.xlabel('Time [ms]')
    # plt.ylabel('Mz')
    # plt.title('T1 relaxation of {} (B+ and f_inv influenced) '.format(lession[tissue]))
    
    # plt.axvline(x=TI_1, linewidth=0.4, color='g', linestyle='--')
    # plt.axvline(x=TI_2, linewidth=0.4, color='g', linestyle='--')
    
    
    # plt.text(TI_2-5, 0.94, TI_2, fontsize=8, style='italic', color='r')
    # plt.text(TI_1-5, 0.94, TI_1, fontsize=8, style='italic', color='r')
    
    # plt.xlim((0, cycle))
    # plt.ylim((-1,1))
    
    
    # ax = plt.subplot(111)
    # box = ax.get_position()
    # ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=2)
    
    # plt.show()
    #=============================================================================
    
    
    
    
    #=============================================================================
        
    # fig1 = plt.figure(1)
    markers = ['s', 'o', 'x', '^', '*', 'd']
    df = pd.DataFrame(np.array([y_K1, y_K2 ]), columns=B_plus)
    ss = ['S1', 'S2']
    angle = [alpha_1, alpha_2]
    
    # for row in range(2):
    
    #     plt.plot(B_plus, df.iloc[row] * np.sin(angle[row]), '{}-'.format(markers[row]) , linewidth=0.5)
            
    # plt.xlabel('B+')
    # plt.ylabel('Mz-value [k0]')
    # plt.title('B+ influence on K-values in {}'.format(lession[tissue]))
    # ax = plt.subplot(111)
    # box = ax.get_position()
    # ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)
    # plt.hlines(0, 0.39, 1.4, linewidth = 0.5, colors='k', linestyles='--')
        
        
    # plt.xlim((B_plus[0]-0.01, B_plus[-1]+0.01))
    # plt.show()
                

    
    

#    fig2 = plt.figure(2)
    
    def MP2RAGE_UNI(S1, S2):
        return (S1*S2)/((S1**2) + (S2**2))
    
    cc =[]
    MP2RAGE_values = []
    for col in df.columns:
        for i in range(2):
            cc.append(df.iloc[i][col])
    
    S1 = [cc[i]  for i in range(0,len(cc),2)]
    S1 = [np.sin(angle[0])[i] * S1[i] for i in range(len(S1))]
    S2 = [cc[i]  for i in range(1,len(cc),2)]
    S2 = [np.sin(angle[1])[j] * S2[j] for j in range(len(S2))]
    
    
    
    for i in range(len(S1)):
        MP2RAGE_values.append(MP2RAGE_UNI(S1[i],S2[i]))
    MP2RAGE_intens_04.append(MP2RAGE_values[0])
    MP2RAGE_intens_06.append(MP2RAGE_values[1])
    MP2RAGE_intens_08.append(MP2RAGE_values[2])
    MP2RAGE_intens_1.append(MP2RAGE_values[3])
    MP2RAGE_intens_1_2.append(MP2RAGE_values[4])
    MP2RAGE_intens_1_4.append(MP2RAGE_values[5])






#%%

from scipy.interpolate import interp1d



T1_Y_axes = np.arange(start,end,jump)



plt.plot(MP2RAGE_intens_04, T1_Y_axes, 'd-', color='{}'.format(Colors[0]), label= r'$ B^+$ = {}'.format(B_plus[0]))
plt.plot(MP2RAGE_intens_06, T1_Y_axes, 'H-', color='{}'.format(Colors[1]), label=r'$ B^+$ = {}'.format(B_plus[1]))
plt.plot(MP2RAGE_intens_08, T1_Y_axes, 'o-', color='{}'.format(Colors[2]), label=r'$ B^+$ = {}'.format(B_plus[2]))
plt.plot(MP2RAGE_intens_1, T1_Y_axes, '^-', color='{}'.format(Colors[3]), label=r'$ B^+$ = {}'.format(B_plus[3]))
plt.plot(MP2RAGE_intens_1_2, T1_Y_axes, 'X-', color='{}'.format(Colors[4]), label=r'$ B^+$ = {}'.format(B_plus[4]))
plt.plot(MP2RAGE_intens_1_4, T1_Y_axes, '*-', color='{}'.format(Colors[5]), label=r'$ B^+$ = {}'.format(B_plus[5]))

# =============================================================================
# Interpolation
# =============================================================================

# f04 = interp1d(MP2RAGE_intens_04, T1_Y_axes , kind='cubic',fill_value="extrapolate")
# f06 = interp1d(MP2RAGE_intens_06, T1_Y_axes , kind='cubic',fill_value="extrapolate")
# f08 = interp1d(MP2RAGE_intens_08, T1_Y_axes , kind='cubic',fill_value="extrapolate")
# f1 = interp1d(MP2RAGE_intens_1 , T1_Y_axes , kind='cubic', fill_value="extrapolate")
# f1_2 = interp1d(MP2RAGE_intens_1_2, T1_Y_axes , kind='cubic',fill_value="extrapolate")
# f1_4 = interp1d(MP2RAGE_intens_1_4, T1_Y_axes , kind='cubic',fill_value="extrapolate")


test_inter = np.linspace(-0.5,0.5,20,endpoint=True)
#plt.plot(test_inter, f04(test_inter), '-', color='{}'.format(Colors[1]), label=r' Interpolated $ B^+$ = {}'.format(B_plus[0]))
#plt.plot(test_inter, f06(test_inter), '-', color='{}'.format(Colors[0]), label=r' Interpolated $ B^+$ = {}'.format(B_plus[1]))
# plt.plot(test_inter, f08(test_inter), '-', color='{}'.format(Colors[3]), label=r' Interpolated $ B^+$ = {}'.format(B_plus[2]))
# plt.plot(test_inter, f1(test_inter), '-', color='{}'.format(Colors[6]), label=r' Interpolated $ B^+$ = {}'.format(B_plus[3]))
# plt.plot(test_inter, f1_2(test_inter), '-', color='{}'.format(Colors[5]), label=r' Interpolated $ B^+$ = {}'.format(B_plus[4]))
# plt.plot(test_inter, f1_4(test_inter), '-', color='{}'.format(Colors[7]), label=r' Interpolated $ B^+$ = {}'.format(B_plus[5]))



plt.title(r'$\alpha_1 / \alpha_2 = {}^o /{}^o  $'.format(flip_1,flip_2))
plt.xlabel('MP2RAGE Intensity')
plt.ylabel('T1-values [ms]')
plt.xlim((-0.501,0.501))
plt.ylim((start, end))
plt.xticks(np.linspace(-0.5,0.5,11),['-0.5','-0.4','-0.3','-0.2','-0.1','0','0.1','0.2','0.3','0.4','0.5'])
plt.axhspan(1000, 1100, facecolor='0.9') # WM
plt.axhspan(1700, 1800, facecolor='0.9') # GM
plt.text(0.48, (1200), 'WM', dict(size=10), color='red')
plt.text(0.48, (1969+2161)/2, 'GM', dict(size=10), color='green')


plt.legend()
plt.show()

    
#T1_map_Values = pd.DataFrame(np.array([MP2RAGE_intens_04,MP2RAGE_intens_06, MP2RAGE_intens_08, MP2RAGE_intens_1,MP2RAGE_intens_1_2, MP2RAGE_intens_1_4]) , columns = ['0.4', '0.6', '0.8', '1', '1.2', '1.4'])
# T1_map_Values = {'0.4':MP2RAGE_intens_04 , '0.6': MP2RAGE_intens_06, '0.8': MP2RAGE_intens_08, '1.0': MP2RAGE_intens_1,'1.2': MP2RAGE_intens_1_2,'1.4': MP2RAGE_intens_1_4}
# T1_map_Values = pd.DataFrame(T1_map_Values, index=T1_Y_axes)
# print(T1_map_Values)
























print("--- %s seconds ---" % (time.time() - start_time))




