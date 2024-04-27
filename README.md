# tnbc

TNBC data is combined from 2 sources, NKI and MCC. I originally downloaded it from Geoff Sedor's github. 

There are 2 files that generate all of the figures because of different packages. These files were last updated around April 2023.
-Most of the work is in analysis.py. This includes code for basic KM analysis, finding the GARD cutpoints, and an in silico trial to analyze how large a sample would be needed to test the utility of RxRSI in guiding selective dose boosts.
-The file tnbc_r.Rmd was used for coxanalysis and ROC analysis.
