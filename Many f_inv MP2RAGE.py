# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 21:12:04 2020

@author: musti
"""



import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

RAGE_1 = 1741
RAGE_2 = 1741
Pulsint = 1850
cycle = 5000

Echo_factor = 6.8
flip_ang_1 = 5*(np.pi/180)
flip_ang_2 = 3*(np.pi/180)


slices = 257
TI_1 = 900
TI_2 = 2750
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

tissue = 2

B_plus = np.sort( [1, 1.2 ,1.4 , 0.4, 0.6,0.8])
alpha_1 = [flip_ang_1 * i for i in B_plus]
alpha_2= [flip_ang_2 * i for i in B_plus]

Iterations = 10


for f_inv in range(len(f_inv_list)):
    y_K1=[]
    y_K2=[]
    for Bplus in range(len(alpha_1)):
        M0_2 = [1]
        M0 = 1
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
                            
                return M0  + (- f_inv_list[f_inv] * M0_2[m] - M0) * np.exp(-t/T_1_list[tissue])       
            
            X1_values = [Mz(t,T_1_list[tissue]) for t in range(X1)]
            #Normal_values = [ f_inv_list[f_inv] * M0_2[0] * (1-2*np.exp(-t/T_1_list[tissue])) for t in range(int(X1+ RAGE_1 + X2 + RAGE_2 + X3))]
            
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
                        
            t_X_2 = np.arange( np.round(X1 + RAGE_1 - Echo_factor, 0) , np.round(X1 + RAGE_1 - Echo_factor + X2, 0) )
            
            
            
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
        
            M0_2.append(M_X_3[-1])
        
        x_K1 = np.round(X1+ RAGE_1//2,0)
        x_K2 = np.round(X1+ RAGE_1 + X2 + RAGE_2//2,0)
        
    
        
        y_K1.append(M_RAGE_1[len(x_RAGE_1[:-1])//2])
        y_K2.append(M_RAGE_2[len(x_RAGE_1[:-1])//2 + 1])
        
        Colors = ['#ff7f0e', '#1f77b4', '#d62728', '#2ca02c', '#e377c2', '#8c564b', '#e377c2', '#bcbd22', '#bcbd22', '#d62728' '#458b74', '#458b74']
        Colors2 = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf' '#458b74', '#d62728']
            
        #------------------ Plotting The Data -----------------------
        
        
    #     plt.plot(np.arange(len(X1_values)) , X1_values, '--', linewidth=0.5, label='X1')
        
    #     plt.plot(x_RAGE_1[:-1] , M_RAGE_1[:-1], '--', linewidth=0.5, label='RAGE 1 {}'.format(B_plus[Bplus]), color='{}'.format(Colors[Bplus]))
        
    #     plt.plot(np.round(X1+ RAGE_1//2,0), M_RAGE_1[len(x_RAGE_1[:-1])//2] ,'*', label = 'k1')
        
    #     plt.plot([i + np.round( RAGE_1 - Echo_factor + X2, 0) for i in x_RAGE_1[:-1]] , M_RAGE_2[:-1], '--', linewidth=0.5, label='RAGE 2 {}'.format(B_plus[Bplus]), color='{}'.format(Colors2[Bplus]))
        
    #     plt.plot(t_X_2, M_X_2, '--', label='X2', linewidth=0.5)
        
    #     plt.plot(t_X_3, M_X_3, 'm--', label='X3', linewidth=0.5)
        
        
        
    #     plt.plot(np.round(X1+ RAGE_1 + X2 + RAGE_2//2,0), M_RAGE_2[len(x_RAGE_1[:-1])//2 + 1] ,'*', label = 'k2')
    
    
    # plt.plot(np.arange(X1+ RAGE_1 + X2 + RAGE_2 + X3-1), Normal_values, 'b-.', linewidth=1, label='Normal {} T1 relaxation'.format(lession[tissue]))
    
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
        
#    fig1 = plt.figure(1)
    markers = ['s', 'o', 'x', '^', '*', 'd']
    df = pd.DataFrame(np.array([y_K1, y_K2 ]), columns=B_plus)
    ss = ['S1', 'S2']
    angle = [alpha_1, alpha_2]
    
    # for row in range(2):
    
    #     plt.plot(B_plus, df.iloc[row] * np.sin(angle[row]), '{}-'.format(markers[row]), label='{}'.format(f_inv_list[f_inv]) , linewidth=0.5)
            
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
                

    
    

    fig2 = plt.figure(2)
    
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
    
    

    plt.title('MP2RAGE "UNI" for {}'.format(lession[tissue]))
    plt.xlabel('B+')
    plt.ylabel('MPRAGE-UNI')
    

    
    MP2RAGE_Experimental_Data = [0.22, 0.258, 0.31, 0.355, 0.392, 0.42]
    plt.plot(B_plus, MP2RAGE_values, '-^' , linewidth=0.6, label='{}'.format(f_inv_list[f_inv]))
plt.plot(B_plus,MP2RAGE_Experimental_Data, 'k-x' ,linewidth=0.6 , label='Experimental')
plt.legend()
plt.show()









