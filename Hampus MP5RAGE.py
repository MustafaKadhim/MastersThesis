# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 20:30:43 2020
@author: musti
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

RAGE_1 = 1822 # 6.8 * 268
RAGE_2 = 1822
RAGE_3 = 1822 
RAGE_4 = 1822
RAGE_5 = 1822
RAGE_6 = 0

Pulsint = 1850


Echo_factor = 6.8
flip_ang_1 = 3*(np.pi/180)
flip_ang_2 = 3*(np.pi/180)
flip_ang_3 = 13*(np.pi/180)
flip_ang_4 = 3*(np.pi/180)
flip_ang_5 = 3*(np.pi/180)
flip_ang_6 = 0*(np.pi/180)

cycle = 9750

slices = 268
TI_1 = 911
TI_2 = 911 + Pulsint
TI_3 = TI_2 + Pulsint
TI_4 = TI_3 + Pulsint
TI_5 = TI_4 + Pulsint


epsi = 0.00000001

T1_csf =3800  #ms
T1_gm = 1700 #ms
T1_wm = 1200

f_inv = 0.96


X1 = 1 #int(abs(TI_1 - abs(RAGE_1/2))) 

X2 =  Pulsint -  RAGE_2 #m

X3 =  Pulsint -  RAGE_2 #ms

X4 = Pulsint -  RAGE_2

X5 = Pulsint -  RAGE_2

X6 = (cycle - (X1+ RAGE_1 + X2 + RAGE_2 + X3 + RAGE_3 + X4 + RAGE_4 + X5 + RAGE_5) )#ms

T_1_list =[ T1_csf, T1_gm,T1_wm ]
lession = ['CSF', 'GM', 'WM']

 #int(X1+ RAGE_1 + X2 + RAGE_2 + X3 + RAGE_3 + RAGE_4 + RAGE_5 + RAGE_6 + X7)

M_csf = np.zeros(cycle)
M_gm = np.zeros(cycle)
M_wm = np.zeros(cycle)
M_tissue = [M_csf]
tissue = 2

B_plus = np.sort( [1, 1.2 ,1.4 , 0.4, 0.6,0.8])
alpha_1 = [flip_ang_1 * i for i in B_plus]
alpha_2= [flip_ang_2 * i for i in B_plus]
alpha_3= [flip_ang_3 * i for i in B_plus]
alpha_4= [flip_ang_4 * i for i in B_plus]
alpha_5= [flip_ang_5 * i for i in B_plus]
alpha_6= [flip_ang_6 * i for i in B_plus]


Iterations = 10

for Bplus in range(len(alpha_1)):
    M0_2 = [1]
    y_K1=[]
    y_K2=[] 
    y_K3=[]
    y_K4=[]  
    y_K5=[]  
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
                        
            return f_inv * M0_2[m] * (1-2*np.exp(-t/T_1_list[tissue]))        
        
        X1_values = [Mz(t,T1_csf) for t in range(X1)]
        
        T1_star = ((1/3900) + (np.log(np.cos(5*(np.pi/180)))/cycle))**-1
        x_long = X1+ RAGE_1 + X2 + RAGE_2 + X3 + RAGE_3 + X4 + RAGE_4 + X5 + RAGE_5 + X6
        Normal_values = [ f_inv * M0_2[0] * (1-2*np.exp(-t/T_1_list[tissue])) for t in range(int(x_long))]
#        facit = [ f_inv * M0_2[0] * (1-2*np.exp(-t/T1_star)) for t in range(cycle)]
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
    
#        M0_2.append(M_X_3[-1])

#================================================================================================================
#================================================================================================================
#================================================================================================================
    
        #------------------ RAGE 3 -----------------------    
        if len(M_X_3) != 0 :
            M_RAGE_3 = np.zeros(len(x_RAGE_1))
            M_RAGE_3[0] = M_X_3[-1]
                            
            for i in range(1,len(M_RAGE_3)-1,2):
                M_RAGE_3[i] = M_RAGE_3[i-1] * np.cos(alpha_3[Bplus])
                for j in range(1):
                    M_RAGE_3[i+j+1] = M_RAGE_3[i] * np.exp(-Echo_factor/T_1_list[tissue]) + (1-np.exp(-Echo_factor/T_1_list[tissue]))
       
        else:
            M_RAGE_3 = np.zeros(len(x_RAGE_1))
            M_RAGE_2[0] = M_RAGE_2[-2]
                            
            for i in range(1,len(M_RAGE_3)-1,2):
                M_RAGE_3[i] = M_RAGE_3[i-1] * np.cos(alpha_3[Bplus])
                for j in range(1):
                    M_RAGE_3[i+j+1] = M_RAGE_3[i] * np.exp(-Echo_factor/T_1_list[tissue]) + (1-np.exp(-Echo_factor/T_1_list[tissue]))
                        
        #------------------ RAGE 4 -----------------------  
        
        if len(M_X_3) != 0 :
            M_RAGE_4 = np.zeros(len(x_RAGE_1))
            M_RAGE_4[0] = M_RAGE_3[-2]
                            
            for i in range(1,len(M_RAGE_4)-1,2):
                M_RAGE_4[i] = M_RAGE_4[i-1] * np.cos(alpha_4[Bplus])
                for j in range(1):
                    M_RAGE_4[i+j+1] = M_RAGE_4[i] * np.exp(-Echo_factor/T_1_list[tissue]) + (1-np.exp(-Echo_factor/T_1_list[tissue]))
                        
    
        
        
        #------------------ RAGE 5 -----------------------  
        if len(M_RAGE_4) != 0 :
            M_RAGE_5 = np.zeros(len(x_RAGE_1))
            M_RAGE_5[0] = M_RAGE_4[-2]
                            
            for i in range(1,len(M_RAGE_5)-1,2):
                M_RAGE_5[i] = M_RAGE_5[i-1] * np.cos(alpha_5[Bplus])
                for j in range(1):
                    M_RAGE_5[i+j+1] = M_RAGE_5[i] * np.exp(-Echo_factor/T_1_list[tissue]) + (1-np.exp(-Echo_factor/T_1_list[tissue]))
 

        #------------------ RAGE 6 -----------------------  
        if len(M_RAGE_5) != 0 :
            M_RAGE_6 = np.zeros(len(x_RAGE_1))
            M_RAGE_6[0] = M_RAGE_5[-2]
                            
            for i in range(1,len(M_RAGE_6)-1,2):
                M_RAGE_6[i] = M_RAGE_6[i-1] * np.cos(alpha_6[Bplus])
                for j in range(1):
                    M_RAGE_6[i+j+1] = M_RAGE_6[i] * np.exp(-Echo_factor/T_1_list[tissue]) + (1-np.exp(-Echo_factor/T_1_list[tissue]))
 
        
        
        #------------------ Normal again -----------------------  


        M_X_7 = np.zeros(int(X6))
        for j in range(0,len(M_X_7)):
            M_X_7[j] = M_RAGE_5[-2] * np.exp(-j/T_1_list[tissue]) + (1-np.exp(-j/T_1_list[tissue])) #tar sista värdet av zikzak 
                    
        t_X_7 = np.arange( np.round(X1 + RAGE_1 - Echo_factor + X2 + RAGE_2 - Echo_factor  + X3 + RAGE_3 - Echo_factor + X4  + RAGE_4 - Echo_factor + RAGE_5 - Echo_factor + X5 , 0)  , 
                          np.round(X1 + RAGE_1 - Echo_factor + X2 + RAGE_2 - Echo_factor  + X3 + RAGE_3 - Echo_factor + X4  + RAGE_4 - Echo_factor + RAGE_5 - Echo_factor + X5 + X6-1, 0))



        M0_2.append(M_X_7[-1])



        
    #------------------ Plotting The Data -----------------------
    Colors = ['#ff7f0e', '#1f77b4', '#d62728', '#2ca02c', '#e377c2', '#8c564b', '#e377c2', '#bcbd22', '#bcbd22', '#d62728' '#458b74', '#458b74']

    Colors2 = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf' '#458b74', '#d62728']
    
    
    fig1 = plt.figure(1)
    
    plt.plot(np.arange(len(X1_values)) , X1_values, '--', linewidth=0.5, label='X1')
    
    plt.plot(x_RAGE_1[:-1] , M_RAGE_1[:-1], '--', linewidth=0.5, label='RAGE 1 {}'.format(B_plus[Bplus]), color='{}'.format(Colors2[Bplus]))
    
    plt.plot(t_X_2, M_X_2, '--', linewidth=0.5)
    
    plt.plot([i + np.round( RAGE_1 - Echo_factor + X2, 0) for i in x_RAGE_1[:-1]] , M_RAGE_2[:-1], '--', linewidth=0.5, label='RAGE 2 {}'.format(B_plus[Bplus]), color='{}'.format(Colors2[Bplus]))

    plt.plot(t_X_3, M_X_3[:-1], 'm--', linewidth=0.5)
    
    plt.plot([i + np.round( RAGE_1 - Echo_factor + X2 + X3 + RAGE_2 - Echo_factor , 0) for i in x_RAGE_1[:-1]] , M_RAGE_3[:-1], '--', linewidth=0.5, label='RAGE 3 {}'.format(B_plus[Bplus]), color='{}'.format(Colors2[Bplus]))
    
    plt.plot([i + np.round( RAGE_1 - Echo_factor + X2 + X3 + RAGE_2 - Echo_factor + RAGE_3 - Echo_factor + X4 , 0) for i in x_RAGE_1[:-1]] , M_RAGE_4[:-1], '--', linewidth=0.5, label='RAGE 4 {}'.format(B_plus[Bplus]), color='{}'.format(Colors2[Bplus]))

    plt.plot([i + np.round( RAGE_1 - Echo_factor + X2 + X3 + RAGE_2 - Echo_factor + RAGE_3 - Echo_factor + RAGE_4 - Echo_factor + X5 , 0) for i in x_RAGE_1[:-1]] , M_RAGE_5[:-1], '--', linewidth=0.5, label='RAGE 5 {}'.format(B_plus[Bplus]), color='{}'.format(Colors2[Bplus]))

    #plt.plot([i + np.round( RAGE_1 - Echo_factor + X2 + X3 + RAGE_2 - Echo_factor + RAGE_3 - Echo_factor + RAGE_4 - Echo_factor + RAGE_5 - Echo_factor , 0) for i in x_RAGE_1[:-1]] , M_RAGE_6[:-1], '--', linewidth=0.5, label='RAGE 6 {}'.format(B_plus[Bplus]), color='{}'.format(Colors2[Bplus]))

    



    plt.plot(np.round(X1+ RAGE_1//2,0), M_RAGE_1[len(x_RAGE_1[:-1])//2] ,'r*', label = 'k1')
    
    plt.plot(np.round(X1+ RAGE_1 + X2 + RAGE_2//2,0), M_RAGE_2[len(x_RAGE_1[:-1])//2 + 1] ,'*', label = 'k2')
    
    plt.plot(np.round(X1+ RAGE_1 + X2 + RAGE_2 + X3 + RAGE_3//2 ,0), M_RAGE_3[len(x_RAGE_1[:-1])//2 + 1] ,'*', label = 'k3')
    
    plt.plot(np.round(X1+ RAGE_1 + X2 + RAGE_2 + X3 + RAGE_3 + X4 +  RAGE_4//2 ,0), M_RAGE_4[len(x_RAGE_1[:-1])//2 + 1] ,'*', label = 'k4')
    
    plt.plot(np.round(X1+ RAGE_1 + X2 + RAGE_2 + X3 + RAGE_3 + X4 + RAGE_4 + X5+ RAGE_5//2 ,0), M_RAGE_5[len(x_RAGE_1[:-1])//2 + 1] ,'*', label = 'k5')
    
    #plt.plot(np.round(X1+ RAGE_1 + X2 + RAGE_2 + X3 + RAGE_3 + RAGE_4 + RAGE_5 + RAGE_6//2 ,0), M_RAGE_6[len(x_RAGE_1[:-1])//2 + 1] ,'*', label = 'k6')
    
    plt.plot(t_X_7,M_X_7[:-1] ,'--', label = 'X7')
    
    
x0 = [0, cycle]
y0 = [0, 0]
plt.plot(x0,y0, 'k--', linewidth=0.5)
plt.xlabel('Time [ms]')
plt.ylabel('Mz')
plt.title('T1 relaxation influenced by f_inv and B+ in {}'.format(lession[tissue]))

# plt.axvline(x=TI_1, linewidth=0.4, color='g', linestyle='--')
# plt.text(TI_1-5, 0.95, TI_1, fontsize=9, style='italic', color='r')

# plt.axvline(x=TI_2, linewidth=0.4, color='g', linestyle='--')
# plt.text(TI_2-5, 0.95, TI_2, fontsize=9, style='italic', color='r')

plt.xlim((0, cycle))
plt.ylim((-1,1))


ax = plt.subplot(111)
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=2)
        
    
    
    
    
    
    
    # fig2 = plt.figure(2)
    # x_K1 = np.round(X1+ RAGE_1//2,0)
    # x_K2 = np.round(X1+ RAGE_1 + X2 + RAGE_2//2,0)
    # x_K3 = np.round(X1+ RAGE_1 + X2 + RAGE_2 + X3 + RAGE_3//2 ,0)
    # x_K4 = np.round(X1+ RAGE_1 + X2 + RAGE_2 + X3 + RAGE_3 + X4 +  RAGE_4//2 ,0)
    # x_K5 = np.round(X1+ RAGE_1 + X2 + RAGE_2 + X3 + RAGE_3 + X4 + RAGE_4 + X5+ RAGE_5//2 ,0)

    
    # y_K1.append(M_RAGE_1[len(x_RAGE_1[:-1])//2] * (flip_ang_1))
    # y_K2.append(M_RAGE_2[len(x_RAGE_1[:-1])//2 + 1] * (flip_ang_2))
    # y_K3.append(M_RAGE_3[len(x_RAGE_1[:-1])//2 + 1] * (flip_ang_3))
    # y_K4.append(M_RAGE_4[len(x_RAGE_1[:-1])//2 + 1] * (flip_ang_4))
    # y_K5.append( M_RAGE_5[len(x_RAGE_1[:-1])//2 + 1] * (flip_ang_5))
    
    # plt.plot([x_K1, x_K2, x_K3, x_K4, x_K5],[y_K1, y_K2, y_K3, y_K4, y_K5], 'o-', label='{}'.format(B_plus[Bplus]))    

#plt.plot(np.arange(X1+ RAGE_1 + X2 + RAGE_2 + X3 + RAGE_3 + RAGE_4  + X4 + RAGE_5 + X5 + X6 ), Normal_values, 'b-.', linewidth=1, label='Normal {} T1 relaxation'.format(lession[tissue]))
#plt.plot(np.arange(X1+ RAGE_1 + X2 + RAGE_2 + X3-1), facit, 'g-.', linewidth=1, label='T1*** CSF facit')






#fig2 = plt.figure(2)

# plt.plot([],[], 'o-')