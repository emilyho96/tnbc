#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 09:35:00 2021

@author: Emily
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.ticker import FuncFormatter
import pandas as pd
import seaborn as sns
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines.plotting import plot_lifetimes
from scipy.optimize import curve_fit
from matplotlib import collections as matcoll
from scipy import stats 
from scipy.special import ndtr
import scipy


# load data; alpha_g calculated with the constants below, then GARD calculated
# with patient-specific n, d
df = pd.read_csv('/Users/Emily/tnbc/data.csv')
df = df.drop(columns=['Unnamed: 0'])
tcc = pd.read_csv('/Users/Emily/tnbc/TCCdata.csv')

# constants/calculations
d = 2
beta = 0.05
n = 1
cut = 22
'''
ag = -np.log(df['RSI'])/(n*d)-beta*d
df['alpha_g'] = ag
gard = df['n']*df['d']*(ag+beta*df['d'])
df['GARD'] = gard

ag = -np.log(tcc['RSI'])/(n*d)-beta*d
tcc['alpha_g'] = ag
gard = n*d*(ag+beta*d)
tcc['GARD'] = gard
'''
# set cut-point, SOC range
gard_t = 22
high = 66 
low = 50 

df = df.sort_values(by='GARD')
nki = df.loc[df['Source'] == 'NKI_wboost']
mcc = df.loc[df['Source'] == 'MCC_wboost']
df['RxRSI'] = gard_t/(df['alpha_g']+beta*d)
tcc['RxRSI'] = gard_t/(tcc['alpha_g']+beta*d)


# KM curves
'''
km_event_all = KaplanMeierFitter()
km_event_all.fit(df['Time'],df['Event'],label='Combined')

km_event_nki = KaplanMeierFitter()
km_event_nki.fit(nki['Time'],nki['Event'],label='NKI')

km_event_mcc = KaplanMeierFitter()
km_event_mcc.fit(mcc['Time'],mcc['Event'],label='MCC')

# plot event KM
a1 = km_event_all.plot(color='black', ci_show=False)
km_event_nki.plot(ax=a1, color='tomato', ci_show=False)
km_event_mcc.plot(ax=a1, color='royalblue', ci_show=False)
plt.xlabel('Years')
plt.ylabel('Event Probability')
plt.ylim([0,1.1])

# plot survival KM
km_surv_mcc = KaplanMeierFitter()
km_surv_mcc.fit(mcc['Time_OS'],mcc['Event_OS'],label='Survival (MCC)')
a2 = km_surv_mcc.plot(ci_show=False, color='royalblue')
plt.xlabel('Years')
plt.ylabel('Survival Probability')
plt.ylim([0,1.1])
'''


# histograms
'''
# stacked histo for RSI
f1 = plt.figure(figsize=(7,5))
ax = f1.add_subplot(1,1,1)
sns.histplot(data=df, ax=ax, stat="count", multiple="stack",
             x="RSI", kde=False,
             palette="deep", hue="Source",
             element="bars", legend=True)
ax.set_title("RSI Histogram")
ax.set_xlabel("RSI")
ax.set_ylabel("Count")

# stacked histo for GARD
f2 = plt.figure(figsize=(7,5))
ax = f2.add_subplot(1,1,1)
sns.histplot(data=df, ax=ax, stat="count", multiple="stack",
             x="GARD", kde=False,
             palette="deep", hue="Source",
             element="bars", legend=True)
ax.set_title("GARD Stacked Histogram")
ax.set_xlabel("GARD")
ax.set_ylabel("Count")
'''


# run comparison KM for dataset based on a GARD cut-point
# returns log-rank stats
def KMbyGARD(time, event, gard, cut, show = False):
   
    # throw error if cut-point is out of the range
    if cut < gard.min() or cut > gard.max():
        print("Cut-point out of range")
        return
   
    temp = pd.DataFrame()
    temp['T'] = time
    temp['E'] = event
    temp['gard'] = gard
    temp = temp.sort_values(by='gard')
    
    above = temp.loc[temp['gard'] > cut]
    below = temp.loc[temp['gard'] <= cut]
    
    km_above = KaplanMeierFitter()
    km_above.fit(above['T'],above['E'],label='GARD > '+str(cut))
    km_below = KaplanMeierFitter()
    km_below.fit(below['T'],below['E'],label='GARD < '+str(cut))
    
    results = logrank_test(above['T'],below['T'],event_observed_A=above['E'], event_observed_B=below['E'])
    # print(results.p_value)
    
    # optional plot
    if show == True:
        
        a2 = km_above.plot(ci_show=False)
        km_below.plot(ax=a2,ci_show=False)
        
    return results, km_above, km_below


# iterates thtrough GARD to minimize p-value
# returns a 1-smaller list of p-values, ordered by GARD
def findCut(time, event, gard, show = False):
    
    p = []
    
    for cut_val in gard:
        
        if cut_val == gard.max():
            break
        
        results, _, _ = KMbyGARD(time, event, gard, cut_val)
        p.append(results.p_value)
        
    if show == True:
        
        a1 = sns.scatterplot(x=gard[:-1], y=p)
        a1.set_yscale('log')
        plt.title("p-value vs GARD cut-point")
    
    return p, gard[:-1].tolist()


# finding gard_t for cohorts
'''p_vals1, gard1 = findCut(nki['Time'], nki['Event'], nki['GARD'])
p_vals2, gard2 = findCut(mcc['Time'], mcc['Event'], mcc['GARD'])
p_vals3, gard3 = findCut(df['Time'], df['Event'], df['GARD'])
p_vals4, gard4 = findCut(mcc['Time_OS'], mcc['Event_OS'], mcc['GARD'])

_, a1, b1 = KMbyGARD(nki['Time'], nki['Event'], nki['GARD'], cut)
_, a2, b2 = KMbyGARD(mcc['Time'], mcc['Event'], mcc['GARD'], cut)
_, a3, b3 = KMbyGARD(df['Time'], df['Event'], df['GARD'], cut)
_, a4, b4 = KMbyGARD(mcc['Time_OS'], mcc['Event_OS'], mcc['GARD'], cut)


# overlying gard_iter plots
fig1 = plt.figure()
plt.title("Event")
plt.scatter(gard1, p_vals1, color='tomato', label='NKI')
plt.scatter(gard2, p_vals2, color='royalblue', label='MCC')
plt.scatter(gard3, p_vals3, color='black', label='Combined')
plt.xlabel('GARD cut-point')
plt.ylabel('p-value')
plt.yscale('log')
plt.legend()

# overlying KM event curves for cutoff GARD
fig2 = a1.plot(color='tomato', ci_show=False, label='NKI above')
b1.plot(ax=fig2, color='tomato', linestyle='dashed', ci_show=False, label='NKI below')
a2.plot(ax=fig2, color='royalblue', ci_show=False, label='MCC above')
b2.plot(ax=fig2, color='royalblue', linestyle='dashed', ci_show=False, label='MCC below')
a3.plot(ax=fig2, color='black', ci_show=False, label='combined above')
b3.plot(ax=fig2, color='black', linestyle='dashed', ci_show=False, label='combined below')
plt.title('KM Event comparison by cohort for GARD_T ' + str(cut))
plt.xlim([0,10])
plt.ylim([0,1.1])
plt.xlabel('Years')
plt.yscale('linear')


# KM survival curves for cutoff GARD
fig3 = a4.plot(color='royalblue', ci_show=False, label='MCC above')
b4.plot(ax=fig2, color='royalblue', linestyle='dashed', ci_show=False, label='MCC below')
plt.title('KM Survavial comparison by cohort for GARD_T ' + str(cut))
plt.xlim([0,10])
plt.ylim([0,1.1])
plt.xlabel('Years')
plt.yscale('linear')

# finding min
coeff = np.polyfit(gard2, p_vals2, 6)
p = np.poly1d(coeff)
x = np.linspace(5, 50, num=100)
y = p(x)
fig = plt.figure()
plt.plot(x,y)
plt.scatter(gard2,p_vals2)
# plt.yscale('log')
plt.xlim([0,50])
plt.ylim([-0.2,1.1])'''



# making figure 1b
# sort by RxRSI
'''df2 = df.sort_values(by='RxRSI').reset_index().drop(columns=['index'])
# group relative to the SOC
hlines = []
llines = []
for i in range(len(df2)):
    y = df2['RxRSI'].loc[i] 
    if y < low:
        llines.append([(i+1,y),(i+1,low)])
    if y > high:
        hlines.append([(i+1,high),(i+1,y)])
# percentages for legend
lperc = round(100*len(llines)/len(df2))
hperc = round(100*len(hlines)/len(df2))
mperc = 100 - lperc - hperc
# below here actually makes the plot
hlinecoll = matcoll.LineCollection(hlines, colors='tomato')
llinecoll = matcoll.LineCollection(llines, colors='royalblue')
fig, ax = plt.subplots()
ax.add_collection(hlinecoll)
ax.add_collection(llinecoll)
plt.scatter(np.linspace(1,len(df2),len(df2)),df2['RxRSI'],c=df2['RxRSI'],cmap='coolwarm')
plt.axhline(y=low,color='gray')
plt.axhline(y=high,color='gray')
plt.xlim([0,len(df2)])
plt.ylim([10,110])
plt.xlabel('Patient ID')
plt.ylabel('RxRSI (Gy) for GARD_T = '+str(gard_t))
plt.text(5, 20, str(lperc)+'% of patients require <'+str(low)+'Gy')
plt.text(15, 52, str(mperc)+'% of patients receive RxRSI \n within SOC range')
plt.text(40, 80, str(hperc)+'% of patients require >'+str(high)+'Gy')
plt.show()'''


# fig 1c (need to run the part for 1b first though)
# CAUTION the color cutoff depends on the bin arrangement
# the PDF is also scaled manually
'''fig, ax = plt.subplots()
x = np.linspace(0,98,50)
N, bins, patches = ax.hist(tcc['RxRSI'],bins=x)
kde = stats.gaussian_kde(tcc['RxRSI'], bw_method='scott')
scale = 350 # idk if this is the right scale but it's eyeballed
curve = scale*kde(x)
for i in range(0,25):
    patches[i].set_facecolor('royalblue')
for i in range(25,33):    
    patches[i].set_facecolor('gray')
for i in range(33,49):
    patches[i].set_facecolor('tomato')    
plt.axvline(x=low, color='gray', linestyle='dashed')
plt.axvline(x=high, color='gray', linestyle='dashed')
plt.xlabel("RxRSI for TCC TNBC")
plt.plot(x, curve, linestyle="dashed", color='black')
    
    
# plot 2a
# the different 'x' may get confusing here
pdf_scaled = kde(x)/max(kde(x))
plt.fill_between(x, y1=pdf_scaled, y2=0, alpha=0.3) #, label="PDF"
pdf = kde.evaluate(x)/sum(kde.evaluate(x))
cdf = np.cumsum(pdf)
plt.plot(x, cdf, label="TCP")
plt.axvline(x=low, color='gray', linestyle='dashed')
plt.axvline(x=high, color='gray', linestyle='dashed')
plt.xlabel("Dose (Gy)")
plt.ylabel("TCP")
# plt.legend()
plt.show()'''


# NTCP calcs; fig 2b
x = np.linspace(0,115,116)
mld = 0.1*x # NEEDS TO BE ADJUSTED
cardiac = pd.read_csv('/Users/Emily/tnbc/MHDdosimetry.csv')
coeffL = np.polyfit(cardiac['Total Dose'], cardiac['MHD_L'], 1) # force y-int 0?
coeffR = np.polyfit(cardiac['Total Dose'], cardiac['MHD_R'], 1)
mhdL = np.poly1d(coeffL)
mhdR = np.poly1d(coeffR)

# constants
b0 = -3.87 # from QUANTEC lung
b1 = 0.126 # from QUANTEC lung

card_base = 0.001 # what should this baseline be? 0?
card_slope = 0.074 

# ALSO NEED TO CHECK THESE ADJUSTMENTS
ntcp_cardL = card_base + card_slope * mhdL(x)
ntcp_cardR = card_base + card_slope * mhdR(x)
ntcp_pulm = np.exp(b0+b1*mld)/(1+np.exp(b0+b1*mld)) # replace x with mld

fig, ax = plt.subplots()
plt.plot(x, ntcp_cardL, label="Major Cardiac Event (L)")
plt.plot(x, ntcp_cardR, label="Major Cardiac Event (R)")
plt.plot(x, ntcp_pulm, label="Pneumonitis")
plt.axvline(x=low, color='gray', linestyle='dashed')
plt.axvline(x=high, color='gray', linestyle='dashed')
plt.xlabel("Dose (Gy)")
plt.ylabel("NTCP")
ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y))) 
plt.legend()


# fig 2c; need to run code for other parts of plot 2 first
'''fig = plt.plot()
x = np.linspace(0,115,116)
scale = 38 # idk if this is the right scale but it's eyeballed

kde = stats.gaussian_kde(df['RxRSI'])
curve1 = kde(x) # idk if this is the right scale but it's eyeballed
plt.fill_between(x, y1=scale*curve1, y2=0, color='tab:blue', alpha=0.3) 
pdf1 = kde.evaluate(x)/sum(kde.evaluate(x))
cdf1 = np.cumsum(pdf1)
plt.plot(x, cdf1, color='tab:blue', label="TCP")

cdf2 = cdf1 - ntcp_card - ntcp_card
plt.plot(x, cdf2, color='tab:orange', label="Adjusted TCP")
pdf2 = np.diff(cdf2) #THIS ASSUMES Y-VALUES ARE 1 APART
plt.fill_between(x[1:], y1=scale*pdf2, y2=0, color='tab:orange', alpha=0.3)

plt.axvline(x=low, color='gray', linestyle='dashed')
plt.axvline(x=high, color='gray', linestyle='dashed')
plt.xlabel("Dose (Gy)")
plt.ylabel("TCP")
plt.legend()
plt.show()'''






