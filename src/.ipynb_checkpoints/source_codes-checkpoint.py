# -*- coding: utf-8 -*-
"""
Created on Sun Apr 27 17:36:15 2025

@author: dmril
"""
import numpy as np
from numpy import vectorize
from scipy.optimize import curve_fit
from scipy.optimize import minimize
from collections import defaultdict

def objective_function_2(Te,Td,P,delta):
    if P > 3.00: #Pi values larger than 3  are considered irrelevant
        delta=10
    T_s=((Te+ delta -Td) *np.exp(-P)  + Td)
    return T_s
OF_2=vectorize(objective_function_2)

############################################################################################
# Define the linear model
def linear_model(x, m, b):
    return m * x + b

# Custom objective function that includes the constraints
def objective(params, x_data, y_data,Td_):
    m, b = params
    # Residuals for least squares fitting
    residuals = y_data - linear_model(x_data, m, b)
    
    # Initialize penalty
    penalty = 0
    
    # Penalize if m < 0.0001
    if m < 0.0025: #max \pi=6
        penalty += 1000  # High penalty for violating the lower bound of m
    
    # Penalize if m > 1
    if m > 1:       #min \pi=0
        penalty += 1000  # High penalty for violating the upper bound of m
    
    # Penalize if b < 0
    if b < 0:
        penalty += 1000  # High penalty for violating the lower bound of b
    
    # Penalize if b > 0.2 * m
    if b > -(Td_-10)*m + Td_:
        penalty += 1000  # High penalty for violating the upper bound of b
    
    # Return the sum of squared residuals plus the penalty
    return np.sum(residuals**2) + penalty

############################################################################################

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
    
    index_omit=np.where(np.array(Cross_Correlation)<0)
    Covariance=np.delete(Covariance,index_omit)
    Cross_Correlation=np.delete(Cross_Correlation,index_omit)
    Std_H=np.delete(Std_H,index_omit)
    Std_ET=np.delete(Std_ET,index_omit)
    T_des=np.delete(T_des,index_omit)
    E_T_roll=np.delete(E_T_roll,index_omit)
    H_T_roll=np.delete(H_T_roll,index_omit)
    Dates_=np.delete(Dates,index_omit)
    
    return Covariance, Cross_Correlation,Std_H,Std_ET,T_des,E_T_roll,H_T_roll,Dates_


def Method1(ET_HT_d_sorted,tw,tw_limit_1,tw_limit_2,x_1,x_2,y_1,y_2,z_1,z_2):
    Pi=[]; Td=[]; delta=[ ];ll=[];ul=[];D=[];M=[];B=[] ;Time=[] #For (Method 1)
    delta_0=10;
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
        # Set bounds for m (slope) and b (intercept)
        #lower_bounds = [0.0001, 0]  # Example: m >= 0.0001, b >= 0
        #upper_bounds = [1, Td_]  # Example: m <= 1, b <= Td_
        #p, pcov =  curve_fit(linear_model, x_.astype('float64'), y_.astype('float64'), bounds=(lower_bounds, upper_bounds))
        #m=p[0]; b=p[1]
        # Minimize the objective function
        #result = minimize(objective, [0.5,20], args=(x_.astype('float64'),y_.astype('float64'),Td_), method='Nelder-Mead')
        #m, b  = result.x
        #Pi_=-np.log(m); delta_=((b-Td_)/m)+Td_
        
        ll_=(33-b)/m ;
        ul_=(36-b)/m;
        D_=np.sum(abs(y_-Td_))/len(y_)
            
            
        Pi.append(Pi_); Td.append(Td_); delta.append(delta_);ll.append(ll_);ul.append(ul_); D.append(D_);M.append(m);B.append(b); Time.append(ET_HT_d_sorted[k][2])
        
    return Pi,Td,delta,ll,ul,m,b,Time


###############################################################################################################################################################
def get_first_6am_index(Dates):
    for i, date in enumerate(Dates):
        if date.hour == 6:
            return i  # Return the index of the first 6:00 AM record
    return -1  # If no 6:00 AM is found, return -1

def get_first_5pm_index(Dates):
    for i, date in enumerate(Dates):
        if date.hour == 17:
            return i  # Return the index of the first 6:00 AM record
    return -1  # If no 6:00 AM is found, return -1



def Index_high_temperatures(Dates,E_T,H_T,gap):
    """ 
    Returns:
    - Index_max: List of indices where the maximum E_T occurs.
    - Index_max_hive: List of indices where the maximum H_T occurs after E_T max.
    - Delta_day: List of day differences for each calculation.
    """
    
    Dates_daylight=[]; E_T_daylight=[]; H_T_daylight=[]
    for i in range(len(Dates)):
        H=Dates[i].hour
        if 6 <= H < 18:
          Dates_daylight.append(Dates[i])
          E_T_daylight.append(E_T[i])
          H_T_daylight.append(H_T[i])

    
    Index_max=[]
    Index_max_hive=[ ]
    Baseline=Dates_daylight[get_first_6am_index(Dates_daylight)] 
    if Baseline > Dates_daylight[0]:
        Delta_day=[((Dates_daylight[0] - Baseline)//3600).days]
        t=((Dates_daylight[0] - Baseline)//3600).days
        #Delta_day=[-1]
        #t=-1 #starting condition
    else:
        Delta_day=[0]
        t=0 #starting condition
        
    j=0
    for i in range(len(Dates_daylight)):
        delta_time = Dates_daylight[i]- Baseline ####!!!! result from Baseline!(The index [28] represents 6:00 am of the first recorded day [Set 2])
        if delta_time.days > t:
            if i-1 > j:
                index=np.where(E_T_daylight[j:i]==np.max(E_T_daylight[j:i]))[0]# It will calculate the maximun value ET from [j to i-1] reason [i-1]>j
                index_aux=np.where(H_T_daylight[int(index[0]+ j):int(index[0]+ j + gap)] == np.max(H_T_daylight[int(index[0]+ j):int(index[0]+ j + gap)]))[0]
                
            else:
                index=[0]
                index_aux=[0]
                
            Index_max.append(int(index[0] +j))
            Index_max_hive.append(int(index[0] +j+ index_aux[0]))
            
            Delta_day.append(delta_time.days)
            t += Delta_day[-1]-Delta_day[-2] #1
            j=i    
        
    return Index_max,Index_max_hive,Dates_daylight,E_T_daylight,H_T_daylight


def Index_low_temperatures(Dates,E_T,H_T,gap):
    """ 
    Returns:
    - Index_max: List of indices where the minimun E_T occurs.
    - Index_max_hive: List of indices where the minimun H_T occurs after E_T max.
    - Delta_day: List of day differences for each calculation.
    """
    
    Index_min=[]
    Index_min_hive=[ ]
    Baseline=Dates[get_first_5pm_index(Dates)] 
    if Baseline > Dates[0]:
        Delta_day=[((Dates[0] - Baseline)//3600).days]
        t=((Dates[0] - Baseline)//3600).days
        #Delta_day=[-1]
        #t=-1 #starting condition
    else:
        Delta_day=[0]
        t=0 #starting condition
        
    j=0
    for i in range(len(Dates)):
        delta_time = Dates[i]- Baseline ####!!!! result from Baseline!(The index [28] represents 6:00 am of the first recorded day [Set 2])
        if delta_time.days > t:
            if i-1 > j:
                index=np.where(E_T[j:i]==np.min(E_T[j:i]))[0]# It will calculate the maximun value ET from [j to i-1] reason [i-1]>j
                index_aux=np.where(H_T[int(index[0]+ j):int(index[0]+ j + gap)] == np.min(H_T[int(index[0]+ j):int(index[0]+ j + gap)]))[0]
                
            else:
                index=[0]
                index_aux=[0]
                
            Index_min.append(int(index[0] +j))
            Index_min_hive.append(int(index[0] +j+ index_aux[0]))
            
            Delta_day.append(delta_time.days)
            t += Delta_day[-1]-Delta_day[-2] #1
            j=i    
        
    return Index_min,Index_min_hive

def find_chop_index(time, results):
    """Stop the calculations when hive remains in a collapse state for more than two days """
    # Convert time to calendar days
    day_labels = time.astype('datetime64[D]')

    # Group indices by day
    day_to_indices = defaultdict(list)
    for idx, day in enumerate(day_labels):
        day_to_indices[day].append(idx)

    # Sort the days chronologically
    sorted_days = np.sort(np.unique(day_labels))

    # Loop through days and check two consecutive days with only zeros
    for i in range(len(sorted_days) - 1):
        d1, d2 = sorted_days[i], sorted_days[i + 1]
        idxs_d1 = day_to_indices[d1]
        idxs_d2 = day_to_indices[d2]

        if all(results[idx] == 0.0 for idx in idxs_d1) and all(results[idx] == 0.0 for idx in idxs_d2):
            # Return index just after the last index in day 2
            return idxs_d2[-1] + 1

    # If no such pair found, return length of results (no chop)
    return len(results)