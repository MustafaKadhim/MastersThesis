"Author: Mustafa Kadhim" 
"Lund University"
"2021-01-17"


import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})
import pandas as pd

#===================================== Sequence Parameters =============================================
RAGE_1 = 1741
RAGE_2 = 1741
Pulsint = 1850
cycle = 5000

Echo_factor = 6.8
flip_ang_1 = 5*(np.pi/180)
flip_ang_2 = 3*(np.pi/180)


slices = 256
TI_1 = 900
TI_2 = 2750


T1_csf =3800  #ms
T1_gm = 1800 #ms
T1_wm = 1200

f_inv_list = np.round(np.linspace(0.80,1.0,6), 3) #different values of the inversion efficiency 
B_plus = np.sort([1, 1.2 ,1.4 , 0.4, 0.6,0.8])    # different values of B1+ 

tissue = 2 # 0=CSF; 1=GM; 2=WM

#==================================================================================================

X1 = abs(TI_1 - RAGE_1//2)

X2 =  Pulsint - 0.5*(RAGE_1 + RAGE_2)

X3= cycle - (RAGE_2/2 + TI_2)

T_1_list =[ T1_csf, T1_gm,T1_wm ]
lession = ['Cerebrospinal fluid (CSF)', 'Gray matter (GM)', 'White matter (WM)']

M_csf = np.zeros(cycle)
M_gm = np.zeros(cycle)
M_wm = np.zeros(cycle)
M_tissue = [M_wm]
alpha_1 = [flip_ang_1 * i for i in B_plus]
alpha_2= [flip_ang_2 * i for i in B_plus]
epsi = 0.00000001
Iterations = 10
#==================================================================================================





for f_inv in range(len(f_inv_list)):
    y_K1=[]
    y_K2=[]
    for Bplus in range(len(alpha_1)):
        M0_2 = [1]    
        for m in range(len(M0_2)+Iterations):       
    #------------------
        
          
            x_RAGE_1 = []

            for i in range(1,slices):
                x_RAGE_1.append((i*epsi + (i-1)*Echo_factor) + X1)
                for j in range(1):
                    x_RAGE_1.append( (j+1)* epsi + (i-1)*Echo_factor + X1)

            #------------------ X1 -----------------------
            
            def Mz(t,T1):
                M0 = 1            
                return M0  + (- f_inv_list[f_inv] * M0_2[m] - M0) * np.exp(-t/T_1_list[tissue])         
            
            X1_values = [Mz(t,T_1_list[tissue]) for t in range(X1+1)]
            Normal_values = [ f_inv_list[f_inv] * M0_2[0] * (1-2*np.exp(-t/T_1_list[tissue])) for t in range(int(X1+ RAGE_1 + X2 + RAGE_2 + X3))]
            
            #------------------ RAGE 1 -----------------------
            
            
            M_RAGE_1=np.zeros(len(x_RAGE_1))
            M_RAGE_1[0]= X1_values[-1]
                        
                        
            for j in range(1,len(x_RAGE_1)-1,2):
                M_RAGE_1[j] = M_RAGE_1[j-1] * (np.cos(alpha_1[Bplus]))
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
        
        orange = '#ff7f0e'
        Colors = ['#2ca02c' , '#bcbd22', '#d62728', '#ff7f0e', '#e377c2', '#44c6b3ff','#d62728', '#bcbd22', '#bcbd22', '#d62728' '#458b74', '#458b74']
        Colors2 = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf' '#458b74', '#d62728']
            
        #------------------ Plotting The Data -----------------------

    #========================== S1 and S2 ========================================
    fig1 = plt.figure(1)
    markers = ['s', 'o', 'x', '^', '*', 'd']
    df = pd.DataFrame(np.array([y_K1, y_K2 ]), columns=B_plus)
    ss = ['S1', 'S2']
    angle = [alpha_1, alpha_2]
    #=============================================================================

  
    #================================= MP2RAGE vs B1+ for finv ==================================
    fontsiz = 24
    width = 5.2
    plt.rcParams.update({'font.size': 26})
    fig2 = plt.figure(1)
    
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
    
    
    colors = ['blue','orange','red','green','deeppink','royalblue']
    facecolor = ['red', 'navajowhite','magenta', 'sandybrown', 'cyan', 'red'] #royalblue
    plt.plot(B_plus, MP2RAGE_values , markersize=20, marker = '^', linewidth=5,color ='{}'.format(colors[f_inv]) ,label='finv = {}'.format(f_inv_list[f_inv]), alpha=0.54) 

plt.title(r'MP2RAGE simulation for {} vs $B_1^+$'.format(lession[tissue]),fontsize=fontsiz, fontweight='bold')
plt.xlabel(r'$B_1^+$-value',fontsize=fontsiz, fontweight='bold')
plt.ylabel('MP2RAGE' ,fontsize=fontsiz, fontweight='bold')
plt.grid(color = 'black' , linestyle = ':', linewidth = 0.7)
plt.legend(loc='lower right', framealpha=0.1)
plt.show()
#=================================================================================================







