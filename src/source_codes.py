# -*- coding: utf-8 -*-
"""
Created on Sun Apr 27 17:36:15 2025

@author: dmril
"""
import numpy as np
from numpy import vectorize
from scipy.optimize import curve_fit

def objective_function_2(Te,Td,P,delta):
    if P > 3.00: #Pi values larger than 3  are considered irrelevant
        delta=10
    T_s=((Te+ delta -Td) *np.exp(-P)  + Td)
    return T_s
OF_2=vectorize(objective_function_2)


def compute_T_d_matrix(Dates, H_T, default_value=33):
    T_d_matrix = {}
    j = 0
    r = 0
    Delta_day = [0]

    for i in range(len(Dates)):
        delta_time = Dates[i] - Dates[0].replace(hour=0, minute=0)

        if delta_time.days > r:
            data = sorted(H_T[j:i])
            filtered_data = [val for val in data if 33 <= val <= 36]
            mean_val = np.mean(filtered_data) if filtered_data else default_value

            # Only use mean if > 33, else default to 33
            T_d_matrix[Dates[i-1].replace(hour=0, minute=0)] = (
                mean_val if mean_val > 33 else default_value
            )

            Delta_day.append(delta_time.days)
            r += Delta_day[-1] - Delta_day[-2]
            j = i

        elif i == len(Dates) - 1:
            data = sorted(H_T[j:i])
            filtered_data = [val for val in data if 33 <= val <= 36]
            mean_val = np.mean(filtered_data) if filtered_data else default_value
            T_d_matrix[Dates[-1].replace(hour=0, minute=0)] = (
                mean_val if mean_val > 33 else default_value
            )
            
    return T_d_matrix


def get_Td(dates,T_d_matrix):
    dates=dates.replace(hour=0, minute=0)
    if dates in T_d_matrix:
        return T_d_matrix[dates]
    else:
        return 34.5


def Method2(Dates,H_T,E_T,T_desired,tw,tw_limit_1_second,tw_limit_2_second,shift):
    Covariance =[]; Cross_Correlation=[]; Std_H=[]; Std_ET=[] ; T_des=[]; E_T_roll=[]; H_T_roll=[] 
    for s in range(len(Dates)):
        #print(s)
        if Dates[s] <= tw_limit_1_second:
            limit_e=Dates[s] + tw/2
            index_e=np.where(Dates <= limit_e)[0]
            signal_1=E_T[index_e]  #Env Temp
            signal_2=H_T[index_e + shift] #Hiv Temp
            Td_=np.mean(np.array(T_desired)[index_e])

            
        elif Dates[s] >= tw_limit_2_second:
            limit_s=Dates[s]-tw/2
            index_s=np.where(Dates >= limit_s)[0]
            signal_1=E_T[index_s-shift] #Env Temp
            signal_2=H_T[index_s]  #Hiv Temp
            Td_=np.mean(np.array(T_desired)[index_s])
            
        
            
        else:
            limit_s=Dates[s] - tw/2
            limit_e=Dates[s] + tw/2
            index=np.where((limit_s < Dates) & (Dates < limit_e))[0]
            signal_1=E_T[index]  #Env Temp
            signal_2=H_T[index + shift] #Hiv Temp
            Td_=np.mean(np.array(T_desired)[index])
            

        H_T_=np.mean(signal_2)
        E_T_=np.mean(signal_1)
        ####################### METHOD 2 ###################################

        value_2=np.std(signal_2); #Std HT
        value_3=np.std(signal_1); #Std ET
        value=sum((signal_2-H_T_)* (signal_1-E_T_))/(len(signal_1)-1)
        value_c=sum((signal_2-H_T_)* (signal_1-E_T_))/np.sqrt(sum((signal_1-E_T_)**2)*sum((signal_2-H_T_)**2))
        Covariance.append(value); 
        Cross_Correlation.append(value_c);
        Std_H.append(value_2); 
        Std_ET.append(value_3);
        T_des.append(Td_);
        H_T_roll.append(H_T_); 
        E_T_roll.append(E_T_)
    
    return Covariance, Cross_Correlation,Std_H,Std_ET,T_des,E_T_roll,H_T_roll


def Method1(ET_HT_d_sorted,tw,tw_limit_1,tw_limit_2,x_1,x_2,y_1,y_2,z_1,z_2):
    Pi=[]; Td=[]; delta=[ ];ll=[];ul=[];D=[];M=[];B=[] ;Time=[] #For (Method 1)
    delta_0=7;
    for k in range(len(ET_HT_d_sorted)):
    
        if ET_HT_d_sorted[k][2] < tw_limit_1:
            ####################### Method 1 (Index) ###################################
            limit_e=ET_HT_d_sorted[k][2]+tw/2
            index_e=np.where(ET_HT_d_sorted[:,2]<=limit_e)[0]
            index_e_1=np.where(z_1<=limit_e)[0]
            index_e_2=np.where(z_2<=limit_e)[0]
            
            x_1_=x_1[index_e_1] #Env T
            x_2_=x_2[index_e_2] #Env T
            y_1_=y_1[index_e_1] #Hiv T
            y_2_=y_2[index_e_2] #Hiv T
            
            Td_=np.mean(np.array(ET_HT_d_sorted[index_e,3]))
    
        elif ET_HT_d_sorted[k][2] > tw_limit_2:
            ####################### Method 1 (Index) ###################################
            limit_s=ET_HT_d_sorted[k][2]-tw/2
            index_s=np.where(ET_HT_d_sorted[:,2]>=limit_s)[0]
            index_s_1=np.where(z_1>=limit_s)[0]
            index_s_2=np.where(z_2>=limit_s)[0]
            
            x_1_=x_1[index_s_1] #Env T
            x_2_=x_2[index_s_2] #Env T
            y_1_=y_1[index_s_1] #Hiv T
            y_2_=y_2[index_s_2] #Hiv T
            
            Td_=np.mean(np.array(ET_HT_d_sorted[index_s,3]))
    
        else:
            ####################### Method 1 (Index) ###################################
            limit_s=ET_HT_d_sorted[k][2]-tw/2
            limit_e=ET_HT_d_sorted[k][2]+tw/2
            index=np.where((limit_s <= ET_HT_d_sorted[:, 2]) & (ET_HT_d_sorted[:, 2] <= limit_e))[0]
            index_1=np.where((limit_s <= z_1) & (z_1 <= limit_e))[0]
            index_2=np.where((limit_s <= z_2) & (z_2 <= limit_e))[0]
            
            x_1_=x_1[index_1] #Env T
            x_2_=x_2[index_2] #Env T
            y_1_=y_1[index_1] #Hiv T
            y_2_=y_2[index_2] #Hiv T
            
            Td_=np.mean(np.array(ET_HT_d_sorted[index,3]))
            
    
            
        ####################### METHOD 1 ###################################
        if np.any(x_2_<=max(x_1_)):
                indices_remove =np.where(x_2_<=max(x_1_))
                x_2_=np.delete(x_2_,indices_remove)
                y_2_=np.delete(y_2_,indices_remove)
            
        x_=np.concatenate((x_1_,x_2_),axis=0)
        y_=np.concatenate((y_1_,y_2_),axis=0)
            

    
        p, pcov = curve_fit(OF_2, x_.astype('float64'), y_.astype('float64'), p0= [Td_,0.5,delta_0], bounds=([0.99999*Td_, 0,0], [Td_,6,10]),maxfev=1000000)
        Pi_=p[1]; delta_=p[2];m=np.exp(-p[1]);
        b=Td_-m*(Td_-delta_);
        ll_=(33-b)/m ;
        ul_=(36-b)/m;
        D_=np.sum(abs(y_-Td_))/len(y_)
            
            
        Pi.append(Pi_); Td.append(Td_); delta.append(delta_);ll.append(ll_);ul.append(ul_); D.append(D_);M.append(m);B.append(b); Time.append(ET_HT_d_sorted[k][2])
        
    return Pi,Td,delta,ll,ul,m,b,Time




    

