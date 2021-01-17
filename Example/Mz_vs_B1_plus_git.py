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

f_inv_list = [0.96] #np.round(np.linspace(0.80,1,6), 3) 
B_plus = np.sort([1, 1.2 ,1.4 , 0.4, 0.6,0.8])

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
    
        width = 3
        marksiz =17

        plt.plot(np.arange(len(X1_values)) , X1_values, '-', linewidth=width)
        
        plt.plot(x_RAGE_1[:-1] , M_RAGE_1[:-1], '-', linewidth=width, label=r'B$_1^+$ = {}'.format(B_plus[Bplus]), color='{}'.format(Colors[Bplus]))
        
        plt.plot(t_X_2, M_X_2, ':', linewidth=width,color='{}'.format(Colors2[Bplus]))
        
        plt.plot([i + np.round( RAGE_1 - Echo_factor + X2, 0) for i in x_RAGE_1[:-1]] , M_RAGE_2[:-1], '-', linewidth=width, color='{}'.format(Colors[Bplus]))
 
        plt.plot(t_X_3, M_X_3, 'm-', linewidth=width)
        
        colo = ['black','black','black']
plt.plot(np.arange(X1+ RAGE_1 + X2 + RAGE_2 + X3-1), Normal_values, '-.',color='{}'.format(colo[tissue]) ,linewidth=width+0.4, label='Normal T1-relaxation')
    
     #=============================================================================
    
fontsiz= 16
x0 = [0, cycle]
y0 = [0, 0]
plt.plot(x0,y0, 'k--', linewidth=0.5)
plt.xlabel('Time [ms]', fontsize=fontsiz, fontweight='bold')
plt.ylabel('Mz', fontsize=fontsiz, fontweight='bold')
plt.title('T1-relaxation of Mz in {}'.format(lession[tissue]), fontsize=fontsiz, fontweight='bold')
plt.axvline(x=TI_1, linewidth=2, color='blue', linestyle='-', label=r'TI$_1$= {}'.format(TI_1))
plt.axvline(x=TI_2, linewidth=2, color='orange', linestyle='-', label=r'TI$_2$= {}'.format(TI_2))


plt.text(TI_2-5, 0.94, TI_2, fontsize=fontsiz, style='italic', color='r')
plt.text(TI_1-5, 0.94, TI_1, fontsize=fontsiz, style='italic', color='r')

plt.xlim((0, cycle))

plt.ylim((-1,1))

ax = plt.subplot(111)
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1, framealpha=0.0)
plt.grid(color = 'black' , linestyle = ':', linewidth = 0.3)

plt.show()