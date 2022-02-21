# -*- coding: utf-8 -*-
"""
Created on Sun Feb 13 23:52:14 2022

@author: eho10
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
from lifelines import WeibullFitter
from lifelines.datasets import load_waltons
from scipy.optimize import curve_fit
from matplotlib import collections as matcoll
from scipy import stats 
from scipy.special import ndtr
import scipy
# import plotly.graph_objs as go
from sklearn.neighbors import KernelDensity
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from scipy.stats import norm
from scipy import special
erf = special.erf
pdf = norm.pdf
cdf = norm.cdf
ppf = norm.ppf


# load data; alpha_g calculated with the constants below, then GARD calculated
# with patient-specific n, d
df = pd.read_csv('data_totBreast.csv')
df_rad = df[df['Received_RT']==1]
nki = pd.read_csv('nki_all.csv')

# constants/calculations
d = 2
beta = 0.05
n = 1
ag = -np.log(nki['RSI'])/(n*d)-beta*d
nki['alpha_g'] = ag
gard = nki['n']*nki['d']*(ag+beta*nki['d'])
nki['GARD'] = gard
df = df.sort_values(by='GARD')


# =============================================================================
# # KM curves
# km_event_all = KaplanMeierFitter()
# km_event_all.fit(df['Time'],df['Event'],label='Combined')
# 
# km_event_rad = KaplanMeierFitter()
# km_event_rad.fit(df_rad['Time'],df_rad['Event'],label='Radiated')
# 
# # plot event KM
# a1 = km_event_all.plot(ci_show=False)
# km_event_rad.plot(ax=a1, linestyle='dashed', ci_show=False)
# plt.xlabel('Years')
# plt.ylabel('Event Probability')
# plt.ylim([0,1.1])
# 
# 
# # histograms 
# f1 = plt.figure(figsize=(7,5))
# sns.histplot(data=df_rad, x='RSI', kde=True, hue="Source", multiple="stack", 
#              palette="deep")
# plt.title("RSI Distribution")
# plt.xlabel("RSI")
# plt.ylabel("Count")
# 
# f2 = plt.figure(figsize=(7,5))
# ax = f2.add_subplot(1,1,1)
# sns.histplot(data=df_rad, ax=ax, stat="count", multiple="layer",
#              x="GARD", kde=True,
#              palette="deep", hue="Source",
#              element="bars", legend=True)
# plt.title("GARD Distribution")
# plt.xlabel("GARD")
# plt.ylabel("Count")
# =============================================================================


# run comparison KM for dataset based on a GARD cut-point
# returns log-rank stats
def KMbyGARD(time, event, sort, cut, show = False):
   
    # throw error if cut-point is out of the range
    if cut < sort.min() or cut > sort.max():
        print("Cut-point out of range")
        return
   
    temp = pd.DataFrame()
    temp['T'] = time
    temp['E'] = event
    temp['sort'] = sort
    temp = temp.sort_values(by='sort')
    
    above = temp.loc[temp['sort'] > cut]
    below = temp.loc[temp['sort'] <= cut]
    
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


df = df_rad[:]
df = df.sort_values(by='GARD')

# =============================================================================
# # finding gard_t for cohorts
# nki = nki.sort_values(by='GARD')
# p_vals, gard = findCut(nki['Time'], nki['Event'], nki['GARD'])
# cut = round(gard[p_vals.index(min(p_vals))])
# 
# _, a1, b1 = KMbyGARD(nki['Time'], nki['Event'], nki['GARD'], cut)
# =============================================================================


# =============================================================================
# # overlying gard_iter plots
# f3 = plt.figure()
# plt.title("Event")
# plt.scatter(gard, p_vals, color='black')
# plt.xlabel('GARD cut-point')
# plt.ylabel('p-value')
# plt.yscale('log')
# plt.legend()
# =============================================================================


# =============================================================================
# # overlying KM event curves for cutoff GARD
# f4 = a1.plot(color='royalblue', ci_show=False, label='above')
# b1.plot(ax=f4, color='royalblue', linestyle='dashed', ci_show=False, label='below')
# plt.title('KM Event comparison by cohort for GARD_T ' + str(cut))
# plt.xlim([0,10])
# plt.ylim([0,1.1])
# plt.xlabel('Years')
# plt.yscale('linear')
# =============================================================================

# =============================================================================
# cut = 22
# # weibull fits for event above and below cut
# # overall event, not survival
# h = df.loc[df['GARD'] > cut]
# s1 = WeibullFitter()
# s1.fit(h['Time'],h['Event'])
# l = df.loc[df['GARD'] <= cut]
# s2 = WeibullFitter()
# s2.fit(l['Time'],l['Event'])
# # save fit parameters
# s1_lambda = s1.lambda_ # adequate dose
# s1_rho = s1.rho_
# s2_lambda = s2.lambda_ # inadequate
# s2_rho = s2.rho_
# # plot weibull fit
# f5 = s1.plot_survival_function(label='above cut')
# s2.plot_survival_function(ax=f5, label='below cut')
# plt.ylim([0,1.1])
# =============================================================================


# evaluate S1 fit at a value t
# S1 is TD > GARD_T
# rho<1 => event likelihood decreases w/time
def s1(t):
    
    # cut = 22, all NKI
    s1_lambda = 28.2737161196875
    s1_rho = 0.9032234811361695
    
# =============================================================================
#     # cut = 22, not TNBC
#     s1_lambda = 27.392902444874665
#     s1_rho = 0.913877859644507
# =============================================================================
    
# =============================================================================
#     # cut = 38, not TNBC
#     s1_lambda = 147.98845867025895
#     s1_rho = 0.7390876088329748
#     
# =============================================================================
#  original TNBC fit  =============================================================================
#     s1_lambda = 147.9029794079022
#     s1_rho = 0.5757462972829996
# =============================================================================

    return np.exp(-np.power(t/s1_lambda, s1_rho))

# evaluate S2 fit at a value t
# below GARD_T
def s2(t):
    
    
    # cut = 22, all NKI
    s2_lambda = 20.33715287727121
    s2_rho = 0.9581215020163584
    
# =============================================================================
#     # cut = 22, not TNBC
#     s2_lambda = 26.181348618625524
#     s2_rho = 0.8270787625348837
# =============================================================================
    
# =============================================================================
#     # cut = 38, not TNBC
#     s2_lambda = 25.10453473507262
#     s2_rho = 0.8654922152346096
# =============================================================================
    
# original TNBC fit =============================================================================
#     s2_lambda = 19.268745101052353
#     s2_rho = 0.6880271300951681
# =============================================================================

    return np.exp(-np.power(t/s2_lambda, s2_rho))


# get coefficients for calculating cardiac/pulm radiation doses from total breast dose
dosi = pd.read_csv('dosi_summ.csv') #dosimetry data for fits
coeffL = np.polyfit(dosi['Total Dose'], dosi['MHD_L'], 1) 
coeffR = np.polyfit(dosi['Total Dose'], dosi['MHD_R'], 1)
coeffLung = np.polyfit(dosi['Total Dose'], dosi['MLD'], 1)

# calc mean heart dose then card RR from total breast dose
def rr_card(td, side=None, CI=None):
    
    mhdL = np.poly1d(coeffL)
    mhdR = np.poly1d(coeffR)
    
    card_slope = 0.074
    if CI == 'upper': card_slope = 0.145
    elif CI == 'lower': card_slope = 0.029
    
    if side==None: rr = 1 + card_slope * (mhdL(td)-mhdL(0)+mhdR(td)-mhdR(0))/2
    if side == 'L': rr = 1 + card_slope * (mhdL(td)-mhdL(0))
    if side == 'R': rr = 1 + card_slope * (mhdR(td)-mhdR(0))
            
    return rr

    
# calc mean lung dose then pulm HR from total breast dose
def hr_pulm(td, CI=None):
    
    mld = np.poly1d(coeffLung)
    MLD = mld(td)
    
    # constants from QUANTEC lung
    b0 = -3.87 
    b1 = 0.126  
    
    # for relative to dose 0
    constant = np.exp(b0)/(1+np.exp(b0))
    
    # TD50 = 30.75 [28.7–33.9] Gy
    if CI == 'upper':
        b0 = -3.33
        b1 = .153
    elif CI == 'lower': 
        b0 = -4.49
        b1 = .100
        
    hr = np.exp(b0+b1*MLD)/(1+np.exp(b0+b1*MLD)) - constant
       
    return hr

    
# individual ntcp

# for 2N patients, draw from RSI distribution
def rsi_sample(N, distr):

    # for some reason this was originally giving identical samples but seems fine now
    kde = stats.gaussian_kde(distr)
    patients = kde.resample(2*N).flatten()
    patients[patients<0] = 0.001
    # rsi_sample = np.random.normal(loc=0.4267245088495575, scale=0.11221246412456044, size=2*N)
    return patients

def adj_surv(surv, td, side=None, CI=None):
    
    rr = rr_card(td, side, CI)
    hr = hr_pulm(td, CI)
    adj = np.power(surv, np.exp(hr)*rr)
        
    return adj


# set cut-point, standard-of-care (SOC) range
cut = 22
high = 66 
low = 50 

rsi_l = np.exp(-n*d*cut/low) # minimum RSI dose
rsi_h = np.exp(-n*d*cut/high) 

# returns penalized survival curves for 2 treatment groups (control and boosted)
# calls s1, s2
def trial(temp, t, style):
    
    N = int(len(temp)/2)
    
    # calculate GARD, RxRSI 
    # assumes 2Gy dose
    temp['RxRSI'] = -n*d*cut/np.log(temp['RSI'])
    # initialize settings
    temp['side'] = list('LR'*N)
    temp['trt'] = 'no boost'
    temp['TD'] = low
    
    # sns.histplot(data=temp, x="RxRSI", kde=True, palette="deep", 
    #               element="bars", legend=True)
    
    grp1 = temp[:N].copy()
    grp2 = temp[N:].copy()
    
    if style == 'random': # randomized trial

        grp2['trt'] = 'boost'
        grp2['TD'] = high
        
    # for boost grp, only RSI in middle range get boost
    if style == 'sorted': 
     
        grp2.loc[((grp2['RSI']>rsi_l) & (grp2['RSI']<rsi_h)),'TD'] = high 
        grp2.loc[(grp2['TD']==high), 'trt'] = 'boost'
        
    # judging by results this may be glitching
    # NEED TO CHECK THESE CLIP MIN/MAX
    if style == 'custom': # for boost grp, TD = RxRSI within range

        grp2['trt'] = 'boost'
        grp2['TD'] = list(grp2['RxRSI'].clip(45, 80))
     
    noboost_count = grp2[grp2['TD']==low].count()
    boost_count = grp2[grp2['TD']>low].count()
    
    # model penalized survival 
    surv1 = []    
    for index, patient in grp1.iterrows():
        
        # select based on whether or not RxRSI is met
        if patient['TD']>=patient['RxRSI']: lc = s1(t)
        else: lc = s2(t)
            
        # adjust for tox (penalized local control)
        plc = adj_surv(lc, patient['TD'], patient['side'])
        surv1.append(plc)
    
    surv2 = []
    for index, patient in grp2.iterrows():
        
        # select based on whether or not RxRSI is met
        if patient['TD']>=patient['RxRSI']: lc = s1(t)
        else: lc = s2(t)
            
        # adjust for tox (penalized local control)
        plc = adj_surv(lc, patient['TD'], patient['side'])
        surv2.append(plc)
    
    return surv1, surv2

nki = pd.read_csv('nki_all.csv')

 
N = 200
rsi_distr = nki['RSI']
tmin = 0
tmax = 10
t = np.linspace(tmin, tmax) # time axis in years
style = 'random'
repeats = 20


curve1 = []
curve2 = []
var1 = []
var2 = []
for i in range(repeats):
    
    patients = pd.DataFrame(rsi_sample(N, rsi_distr), columns=['RSI'])
    surv1, surv2 = trial(patients, t, style)
    curve1.append(np.mean(surv1, axis=0))
    curve2.append(np.mean(surv2, axis=0))
    # var1.append(np.var(surv1, axis=0))
    # var2.append(np.var(surv2, axis=0))
    
plc1 = np.mean(curve1, axis=0)
lower1 = np.percentile(curve1, 2.5, axis=0)
upper1 = np.percentile(curve1, 97.5, axis=0)
plc2 = np.mean(curve2, axis=0)
lower2 = np.percentile(curve2, 2.5, axis=0)
upper2 = np.percentile(curve2, 97.5, axis=0)
# se1 = np.std(curve1, axis=0)
# se2 = np.std(curve2, axis=0)


fig, ax = plt.subplots() # figsize=(30,20)
plt.fill_between(t, lower1, upper1, alpha=.3) 
plt.fill_between(t, lower2, upper2, alpha=.3) 
# plt.fill_between(t, plc1-2*se1, plc1+2*se1, alpha=.3) 
# plt.fill_between(t, plc2-2*se2, plc2+2*se2, alpha=.3) 
plt.plot(t, plc1, label='no boost')
plt.plot(t, plc2, label='boost')
plt.legend()
plt.xlabel('Years')
plt.ylabel('Percent event-free')
plt.title(style+' survival comparison, n='+str(2*N)+', '+str(repeats)+' trials')
ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y))) 
plt.ylim(0.5,0.7)

print(N, repeats)
print(lower1[-1], upper1[-1])
print(lower2[-1], upper2[-1])
# print(plc1[-1]-2*se1[-1], plc1[-1]+2*se1[-1])
# print(plc2[-1]-2*se2[-1], plc2[-1]+2*se2[-1])




'''
N = 200
rsi_distr = tcc['RSI']
t = np.linspace(0,10) # time axis in years
rsi_l = np.exp(-n*d*cut/low) # minimum
rsi_h = np.exp(-n*d*cut/high) 
style = 'custom'

fig = plt.figure(figsize=(35,10))
from matplotlib.gridspec import GridSpec
# gs1 = GridSpec(1, 1, figure=fig)
gs2 = GridSpec(1, 3, figure=fig)
# =============================================================================
# Left panel: unsorted
# =============================================================================
ax = fig.add_subplot(gs2[0,0])
style = 'random'
results = trial(patients, t, style)
# average survival for each group
noboost_surv = np.mean(results.loc[results['trt']=='no boost']['surv'], axis=0)
noboost_err = np.std(list(results.loc[results['trt']=='no boost']['surv']), axis=0)
boost_surv = np.mean(results.loc[results['trt']=='boost']['surv'], axis=0)
boost_err = np.std(list(results.loc[results['trt']=='boost']['surv']), axis=0)
# should this be 2stdev?
plt.fill_between(t, boost_surv-boost_err, boost_surv+boost_err, alpha=.3) 
plt.fill_between(t, noboost_surv-noboost_err, noboost_surv+noboost_err, alpha=.3) 
plt.plot(t, boost_surv, label='boost')
plt.plot(t, noboost_surv, label='no boost')
plt.legend()
plt.xlabel('Years')
plt.ylabel('Percent event-free')
plt.title(style+' survival comparison, n='+str(2*N))
ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y))) 
plt.ylim(0,1)
# =============================================================================
# Middle panel: sorted boost
# =============================================================================
ax = fig.add_subplot(gs2[0,1])
style = 'sorted'
results = trial(patients, t, style)
# average survival for each group
noboost_surv = np.mean(results.loc[results['trt']=='no boost']['surv'], axis=0)
noboost_err = np.std(list(results.loc[results['trt']=='no boost']['surv']), axis=0)
boost_surv = np.mean(results.loc[results['trt']=='boost']['surv'], axis=0)
boost_err = np.std(list(results.loc[results['trt']=='boost']['surv']), axis=0)
# should this be 2stdev?
plt.fill_between(t, boost_surv-boost_err, boost_surv+boost_err, alpha=.3) 
plt.fill_between(t, noboost_surv-noboost_err, noboost_surv+noboost_err, alpha=.3) 
plt.plot(t, boost_surv, label='boost')
plt.plot(t, noboost_surv, label='no boost')
plt.legend()
plt.xlabel('Years')
plt.ylabel('Percent event-free')
plt.title(style+' survival comparison, n='+str(2*N))
ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y))) 
plt.ylim(0,1)
# =============================================================================
# Right panel: custom boost
# =============================================================================
ax = fig.add_subplot(gs2[0,2])
style = 'custom'
results = trial(patients, t, style)
# average survival for each group
noboost_surv = np.mean(results.loc[results['trt']=='no boost']['surv'], axis=0)
noboost_err = np.std(list(results.loc[results['trt']=='no boost']['surv']), axis=0)
boost_surv = np.mean(results.loc[results['trt']=='boost']['surv'], axis=0)
boost_err = np.std(list(results.loc[results['trt']=='boost']['surv']), axis=0)
# should this be 2stdev?
plt.fill_between(t, boost_surv-boost_err, boost_surv+boost_err, alpha=.3) 
plt.fill_between(t, noboost_surv-noboost_err, noboost_surv+noboost_err, alpha=.3) 
plt.plot(t, boost_surv, label='boost')
plt.plot(t, noboost_surv, label='no boost')
plt.legend()
plt.xlabel('Years')
plt.ylabel('Percent event-free')
plt.title(style+' survival comparison, n='+str(2*N))
ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y))) 
plt.ylim(0,1)

'''
