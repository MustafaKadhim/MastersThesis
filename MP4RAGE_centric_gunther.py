# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 13:32:31 2020

@author: musti
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

RAGE_1 = int(6.4 * 256) #1638
RAGE_2 = int(6.4 * 256) # 1638
RAGE_3 = int(6.4 * 256)
RAGE_4 = int(6.4 * 256)

cycle = 7700 #long , #7200 kort

flip_1 = 12
flip_2 = 4
flip_3 = 4
flip_4 = 4

Echo_factor = 6.4

flip_ang_1 = flip_1 *(np.pi/180)
flip_ang_2 = flip_2 *(np.pi/180)
flip_ang_3 = flip_3 *(np.pi/180)
flip_ang_4 = flip_4 *(np.pi/180)


slices = 256


#Pulsint = TI_2 - TI_1
epsi = 0.00000001

T1_csf =3900  #ms
T1_gm = 1800 #ms
T1_wm = 1200 #ms

f_inv_list = np.round(np.linspace(0.80,1,6), 3)

TI_1 = 1100 # 600 kort
TI_2 = TI_1 + RAGE_1
TI_3 = TI_2 + RAGE_2
TI_4 = TI_3 + RAGE_3

X1 = TI_1

X2 =  abs(TI_2 - (TI_1 + RAGE_1 ))

X3 = abs(TI_3 - (TI_2 + RAGE_2))

X5= abs(cycle - (TI_4 + RAGE_4))

T_1_list =[ T1_csf, T1_gm,T1_wm ]
lession = ['CSF', 'GM', 'WM']

M_csf = np.zeros(cycle)
M_gm = np.zeros(cycle)
M_wm = np.zeros(cycle)
M_tissue = [M_wm]

tissue = 2

B_plus = np.sort( [1, 1.2 ,1.4 , 0.4, 0.6,0.8])
alpha_1 = [flip_ang_1 * i for i in B_plus]
alpha_2= [flip_ang_2 * i for i in B_plus]
alpha_3= [flip_ang_3 * i for i in B_plus]
alpha_4= [flip_ang_4 * i for i in B_plus]

Iterations = 10

f_inv = 0.96 #0.94






y_K1=[]
y_K2=[]
for Bplus in range(len(alpha_1)):

    M0_2 = [1]    
    for m in range(len(M0_2)+Iterations):       
#------------------
    
      
      # Plottar tiden som finns först i listan 
        x_RAGE_1 = []
        for i in range(1,slices):
            x_RAGE_1.append((i*epsi + (i-1)* Echo_factor) + X1)
            for j in range(1):
                x_RAGE_1.append(i *(epsi + Echo_factor) + X1)
        
        # the k1 x-point is at np.round(x_RAGE_1[len(x_RAGE_1)//2], 0) = 900
        #------------------ X1 -----------------------
        
        def Mz(t,T1):
                        
            return f_inv * M0_2[m] * (1-2*np.exp(-t/T_1_list[tissue])) 
        
      
        X1_values = [Mz(t,T_1_list[tissue]) for t in range(X1)]


        
        #------------------ RAGE 1 -----------------------
        
     
        M_RAGE_1=np.zeros(len(x_RAGE_1))
        M_RAGE_1[0]= X1_values[-1]
                    
                    
        for j in range(1,len(x_RAGE_1)-1,2):
            M_RAGE_1[j] = M_RAGE_1[j-1] * np.cos(alpha_1[Bplus])
            for k in range(1):
                M_RAGE_1[j+k+1] = M_RAGE_1[j] * np.exp(-Echo_factor/T_1_list[tissue]) + (1-np.exp(-Echo_factor/T_1_list[tissue]))
                    
        
        
        #------------------ X2 -----------------------
        
        
        M_X_2 = np.zeros(int(X2))
        for j in range(0,len(M_X_2)):
            M_X_2[j] = M_RAGE_1[-2] * np.exp(-j/T_1_list[tissue]) + (1-np.exp(-j/T_1_list[tissue])) #tar sista värdet av zikzak 
                    
        t_X_2 = np.arange( np.round(X1 + RAGE_1 - Echo_factor, 0) , np.round(X1 + RAGE_1 - Echo_factor + X2-1, 0) )
        
        
        
        #------------------ RAGE 2 -----------------------
        if len(M_X_2) != 0 :
            M_RAGE_2 = np.zeros(len(x_RAGE_1))
            M_RAGE_2[0] = M_X_2[-1]
                            
            for i in range(1,len(M_RAGE_2)-1,2):
                M_RAGE_2[i] = M_RAGE_2[i-1] * np.cos(alpha_2[Bplus])
                for j in range(1):
                    M_RAGE_2[i+j+1] = M_RAGE_2[i] * np.exp(-Echo_factor/T_1_list[tissue]) + (1-np.exp(-Echo_factor/T_1_list[tissue]))
                        
        
        else:
            M_RAGE_2 = np.zeros(len(x_RAGE_1))
            M_RAGE_2[0] = M_RAGE_1[-2]
                            
            for i in range(1,len(M_RAGE_2)-1,2):
                M_RAGE_2[i] = M_RAGE_2[i-1] * np.cos(alpha_2[Bplus])
                for j in range(1):
                    M_RAGE_2[i+j+1] = M_RAGE_2[i] * np.exp(-Echo_factor/T_1_list[tissue]) + (1-np.exp(-Echo_factor/T_1_list[tissue]))
                        
        
        
        #------------------ X3 -----------------------
        
        M_X_3 = np.zeros(int(X3))
        for j in range(0,len(M_X_3)):
            M_X_3[j] = M_RAGE_2[-2] * np.exp(-j/T_1_list[tissue]) + (1-np.exp(-j/T_1_list[tissue])) #tar sista värdet av zikzak 
                    
        t_X_3 = np.arange( np.round(X1 + RAGE_1 - Echo_factor + X2 + RAGE_2 - Echo_factor, 0)  , 
                          np.round(X1 + RAGE_1 - Echo_factor + X2 + RAGE_2 - Echo_factor + X3-1, 0))
    
        #M0_2.append(M_X_3[-1])
        
        #------------------ RAGE 3 -----------------------
        if len(M_X_3) != 0 :
            M_RAGE_3 = np.zeros(len(x_RAGE_1))
            M_RAGE_3[0] = M_X_3[-1]
                        
            for i in range(1,len(M_RAGE_2)-1,2):
                M_RAGE_3[i] = M_RAGE_3[i-1] * np.cos(alpha_3[Bplus])
                for j in range(1):
                    M_RAGE_3[i+j+1] = M_RAGE_3[i] * np.exp(-Echo_factor/T_1_list[tissue]) + (1-np.exp(-Echo_factor/T_1_list[tissue]))
                        
    
        else:
            M_RAGE_3 = np.zeros(len(x_RAGE_1))
            M_RAGE_3[0] = M_RAGE_2[-2]
                            
            for i in range(1,len(M_RAGE_3)-1,2):
                M_RAGE_3[i] = M_RAGE_3[i-1] * np.cos(alpha_3[Bplus])
                for j in range(1):
                    M_RAGE_3[i+j+1] = M_RAGE_3[i] * np.exp(-Echo_factor/T_1_list[tissue]) + (1-np.exp(-Echo_factor/T_1_list[tissue]))
                    
#                M_X_3 = np.zeros(int(X3))
        
                
        
        if len(M_RAGE_3) != 0 :
            M_RAGE_4 = np.zeros(len(x_RAGE_1))
            M_RAGE_4[0] = M_RAGE_3[-2]
                        
            for i in range(1,len(M_RAGE_2)-1,2):
                M_RAGE_4[i] = M_RAGE_4[i-1] * np.cos(alpha_4[Bplus])
                for j in range(1):
                    M_RAGE_4[i+j+1] = M_RAGE_4[i] * np.exp(-Echo_factor/T_1_list[tissue]) + (1-np.exp(-Echo_factor/T_1_list[tissue]))
                        
                
   
        else:
            M_RAGE_4 = np.zeros(len(x_RAGE_1))
            M_RAGE_4[0] = M_RAGE_3[-2]
                            
            for i in range(1,len(M_RAGE_4)-1,2):
                M_RAGE_4[i] = M_RAGE_4[i-1] * np.cos(alpha_4[Bplus])
                for j in range(1):
                    M_RAGE_4[i+j+1] = M_RAGE_4[i] * np.exp(-Echo_factor/T_1_list[tissue]) + (1-np.exp(-Echo_factor/T_1_list[tissue]))
                    

   
    
        M_X_5 = np.zeros(int(X5))        
        for j in range(0,len(M_X_5)):
            M_X_5[j] = M_RAGE_4[-2] * np.exp(-j/T_1_list[tissue]) + (1-np.exp(-j/T_1_list[tissue])) #tar sista värdet av zikzak 
                    
        t_X_5 = np.arange( int(TI_4 + RAGE_4)  , int(cycle ))    
        
        M0_2.append(M_X_5[-1])
    
    Colors = ['#ff7f0e', '#1f77b4', '#d62728', '#2ca02c', '#e377c2', '#8c564b', '#e377c2', '#bcbd22', '#bcbd22', '#d62728' '#458b74', '#458b74']
    Colors2 = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf' '#458b74', '#d62728']
        
    #------------------ Plotting The Data -----------------------
    
    
    # plt.plot(np.arange(len(X1_values)) , X1_values, '--', linewidth=0.5, label='X1')
    
    # plt.plot(x_RAGE_1[:-1] , M_RAGE_1[:-1], '--', linewidth=0.5, label='RAGE 1 {}'.format(B_plus[Bplus]), color='{}'.format(Colors[Bplus]))
    
    # plt.plot(x_RAGE_1[0], M_RAGE_1[0] ,'*', label = 'k1')
    
    # plt.plot([i + np.round( RAGE_1 - Echo_factor + X2, 0) for i in x_RAGE_1[:-1]] , M_RAGE_2[:-1], '--', linewidth=0.5, label='RAGE 2 {}'.format(B_plus[Bplus]), color='{}'.format(Colors2[Bplus]))
    
    # plt.plot(t_X_2, M_X_2[:-1], '--', label='X2', linewidth=0.5)
    
    # plt.plot(t_X_3, M_X_3[:-1], 'm--', label='X3', linewidth=0.5)
    
    
    
    # plt.plot(np.round(X1+ RAGE_1 + X2 + RAGE_2//2,0), M_RAGE_2[len(x_RAGE_1[:-1])//2 + 1] ,'*', label = 'k2')


    #X1_values = [Mz(t,T_1_list[tissue]) for t in range(X1)]
    plt.plot(np.arange(len(X1_values)) , X1_values, '--', linewidth=0.5, label='X1')
    plt.plot(x_RAGE_1[:-1] , M_RAGE_1[:-1], '--', linewidth=0.5, label='RAGE 1 {}'.format(B_plus[Bplus]), color='{}'.format(Colors[Bplus]))
    plt.plot(x_RAGE_1[0], M_RAGE_1[0] ,'*', label = 'k1')
    plt.plot([i + np.round(RAGE_1 + X2, 0) for i in x_RAGE_1[:-1]] , M_RAGE_2[:-1], '--', linewidth=0.5, label='RAGE 2 {}'.format(B_plus[Bplus]), color='{}'.format(Colors2[Bplus]))
    plt.plot([i + np.round(RAGE_1 + X2, 0) for i in x_RAGE_1[:-1]][0], M_RAGE_2[0] ,'*', label = 'k2')
    plt.plot([i + np.round(RAGE_1 + X2 + RAGE_2, 0) for i in x_RAGE_1[:-1]] , M_RAGE_3[:-1], '--', linewidth=0.5, label='RAGE 3 {}'.format(B_plus[Bplus]), color='{}'.format(Colors[Bplus]))
    plt.plot([i + np.round(RAGE_1 + X2 + RAGE_2, 0) for i in x_RAGE_1[:-1]][0], M_RAGE_3[0] ,'*', label = 'k3')
    
    plt.plot([i + np.round(RAGE_1 + X2 + RAGE_2 + RAGE_3, 0) for i in x_RAGE_1[:-1]] , M_RAGE_4[:-1], '--', linewidth=0.5, label='RAGE 4 {}'.format(B_plus[Bplus]), color='{}'.format(Colors[Bplus]))
    plt.plot([i + np.round(RAGE_1 + X2 + RAGE_2 + RAGE_3, 0) for i in x_RAGE_1[:-1]][0], M_RAGE_4[0] ,'*', label = 'k4')

    plt.plot(t_X_5, M_X_5, '--', linewidth=0.5, label='X5 {}'.format(B_plus[Bplus]), color='{}'.format(Colors[Bplus]))
    
    
    
# plt.plot(np.arange(cycle), Mz(np.arange(cycle), T_1_list[tissue]), 'b-.', linewidth=1, label='Normal {} T1 relaxation'.format(lession[tissue]) )




# #=============================================================================

fig1 = plt.figure(1)
x0 = [0, cycle]
y0 = [0, 0]
plt.plot(x0,y0, 'k--', linewidth=0.5)
plt.xlabel('Time [ms]')
plt.ylabel('Mz')
plt.title('T1 relaxation influenced by f_inv and B+ in {}, for Alpha_1 =  {}' .format(lession[tissue], flip_1))

plt.axvline(x=TI_1, linewidth=0.4, color='g', linestyle='--')
plt.axvline(x=TI_2, linewidth=0.4, color='g', linestyle='--')
plt.axvline(x=TI_3, linewidth=0.4, color='g', linestyle='--')
plt.axvline(x=TI_4, linewidth=0.4, color='g', linestyle='--')

plt.text(TI_2-5, 0.94, TI_2, fontsize=8, style='italic', color='r')
plt.text(TI_1-5, 0.94, TI_1, fontsize=8, style='italic', color='r')
plt.text(TI_3-5, 0.94, TI_3, fontsize=8, style='italic', color='r')
plt.text(TI_4-5, 0.94, TI_4, fontsize=8, style='italic', color='r')



plt.xlim((0, cycle))
plt.ylim((-1,1))


ax = plt.subplot(111)
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=2)

plt.show()
