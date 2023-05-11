# -*- coding: utf-8 -*-
"""
Created on Thu May 11 17:24:30 2023

@author: Jie Zhang
"""

import pandas as pd
from plotnine import ggplot,aes,geom_violin,geom_boxplot,geom_jitter,\
                      position_dodge,position_jitter,geom_point,geom_errorbar,geom_tile,geom_text,\
                     scale_y_continuous,scale_x_continuous,scale_fill_hue,theme_matplotlib,theme,\
                     ylab,xlab,xlim,guide_legend,ylim,scale_fill_manual,element_text,geom_line,\
                     guides,scale_color_manual,facet_wrap,element_rect,scale_fill_cmap,scale_shape_manual,\
                     scale_linetype_manual,scale_color_distiller,scale_color_brewer,scale_fill_brewer,geom_hline,geom_vline,\
                     geom_path,annotate,element_line
from plotnine.guides import guide_legend            
import numpy as np
from pandas.api.types import CategoricalDtype
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import AnovaRM
from scipy.stats import ttest_rel,ttest_ind
from matplotlib import pyplot as plt 
import pingouin as pg
#import patchworklib as pw
#https://stackoverflow.com/questions/52331622/plotnine-any-work-around-to-have-two-plots-in-the-same-figure-and-print-it

df_data0=pd.read_csv("Experimental_Records.csv")

df_data=df_data0[df_data0["Test_order"]!=7]
df_data["test_id"]=range(df_data["ID"].shape[0])
print(df_data.columns.values)


#==================================================Condition difference================================


for i in range(-3,4):   
    df_noseComfort_size=df_data[df_data["Nosepad_Interval"]==i]
    #t,p= ttest_rel(df_noseComfort_size.Nosepad_Position_S, df_noseComfort_size.Nosepad_Position_D)#, paired=True)
    D_mean=df_noseComfort_size.Nosepad_Position_S.mean()- df_noseComfort_size.Nosepad_Position_D.mean()
    #print('Nosepad_Position: nosepad_Width={:.1f}, D_mean={:.2f} t = {:.4f}, p = {:.4f}'.format(i, D_mean, t, p))
    pg_model=pg.wilcoxon(df_noseComfort_size.Nosepad_Position_S, df_noseComfort_size.Nosepad_Position_D)#,paired=True)
    print('Nosepad_Position: nosepad_Width={:.1f}, D_mean={:.3f}, W = {:.3f}, p = {:.3f}'.format(i, D_mean,pg_model["W-val"][0],pg_model["p-val"][0]))


for i in range(-3,4):
    df_noseComfort_size=df_data[df_data["Nosepad_Interval"]==i]
    #t,p= ttest_rel(df_noseComfort_size.Nosepad_Comfort_S, df_noseComfort_size.Nosepad_Comfort_D)#, paired=True)
    D_mean=df_noseComfort_size.Nosepad_Comfort_S.mean()- df_noseComfort_size.Nosepad_Comfort_D.mean()
    #print('Nosepad_Comfort,nosepad_Width={:.1f},D_mean={:.2f} t = {:.4f}, p = {:.4f}'.format(i,D_mean, t, p))
    pg_model=pg.wilcoxon(df_noseComfort_size.Nosepad_Comfort_S, df_noseComfort_size.Nosepad_Comfort_D)#,paired=True)
    print('Nosepad_Comfort: nosepad_Width={:.1f}, D_mean={:.3f}, W = {:.3f}, p = {:.3f}'.format(i, D_mean,pg_model["W-val"][0],pg_model["p-val"][0]))

force_interval=0.075
df_data["Int_Force"]=np.round(df_data["Clapping_Force"]/force_interval)*force_interval  

for i in np.arange(force_interval,0.6,force_interval):
    df_templeComfort_size=df_data[df_data["Int_Force"]==np.round(i/force_interval)*force_interval]
    #t,p= ttest_rel(df_templeComfort_size.Temple_Fit_S, df_templeComfort_size.Temple_Fit_D)#, paired=True)
    D_mean=df_templeComfort_size.Temple_Fit_S.mean()- df_templeComfort_size.Temple_Fit_D.mean()
    #print('Temple_Fit, Force={:.3f},D_mean={:.2f}, t = {:.4f}, p = {:.4f}'.format(i,D_mean, t, p))
    pg_model=pg.wilcoxon(df_templeComfort_size.Temple_Fit_S, df_templeComfort_size.Temple_Fit_D)
    print('Temple Fit: Force={:.3f}, D_mean={:.3f}, W = {:.3f}, p = {:.3f}'.format(i, D_mean,pg_model["W-val"][0],pg_model["p-val"][0]))
   
    
for i in np.arange(force_interval,0.6,force_interval):
    df_templeComfort_size=df_data[df_data["Int_Force"]==np.round(i/force_interval)*force_interval]
    #t,p= ttest_rel(df_templeComfort_size.Temple_Comfort_S, df_templeComfort_size.Temple_Comfort_D)#, paired=True)
    D_mean=df_templeComfort_size.Temple_Comfort_S.mean()- df_templeComfort_size.Temple_Comfort_D.mean()
    #print('Temple_Comfort, Force={:.3f},D_mean={:.2f}, t = {:.4f}, p = {:.4f}'.format(i,D_mean, t, p))
   # pg_model=pg.ttest(df_templeComfort_size.Temple_Fit_S, df_templeComfort_size.Temple_Fit_D,paired=True)
   # print(pg_model)    
    pg_model=pg.wilcoxon(df_templeComfort_size.Temple_Comfort_S, df_templeComfort_size.Temple_Comfort_D)#,paired=True)
    print('Temple Comfort: Force={:.3f}, D_mean={:.3f}, W= {:.3f}, p = {:.3f}'.format(i, D_mean,pg_model["W-val"][0],pg_model["p-val"][0]))
    #print(pg_model)    
   

#==================================================Size difference================================
#https://github.com/raphaelvallat/pingouin/blob/master/notebooks/01_ANOVA.ipynb
check_normality=pg.normality(df_data, group='Nosepad_Interval', dv='Nosepad_Position_S')
print(check_normality)
check_normality=pg.normality(df_data, group='Nosepad_Interval', dv='Nosepad_Position_D')
print(check_normality)

check_homoscedasticity=pg.homoscedasticity(df_data, group='Nosepad_Interval', dv='Nosepad_Position_S')
print(check_homoscedasticity)


post_nosePosit_S=pg.pairwise_tests(data=df_data,dv="Nosepad_Position_S",between="Nosepad_Interval",parametric=False).round(3)
post_nosePosit_D=pg.pairwise_tests(data=df_data,dv="Nosepad_Position_D",between="Nosepad_Interval",parametric=False).round(3)

post_nosePosit_S0=pg.pairwise_tukey(data=df_data,dv="Nosepad_Position_S",between="Nosepad_Interval").round(3)
post_nosePosit_D0=pg.pairwise_tukey(data=df_data,dv="Nosepad_Position_D",between="Nosepad_Interval").round(3)
post_nosePosit_S["diff"]=post_nosePosit_S0["diff"]
post_nosePosit_D["diff"]=post_nosePosit_D0["diff"]

temp_A=post_nosePosit_D["A"].copy()
post_nosePosit_D["A"]=post_nosePosit_D["B"].copy()
post_nosePosit_D["B"]=temp_A.copy()
post_nosePosit_D["diff"]*=-1

post_nosePosit=post_nosePosit_S.append(post_nosePosit_D)
print(post_nosePosit)

post_nosePosit["plabel"]=["ns" if x>0.05  else "*" for x in post_nosePosit["p-unc"]]

gg_post_nosePosit=(ggplot(post_nosePosit,aes('A', 'B'))
 +geom_tile(aes(fill='diff'),color="k",size=0.25)
 +geom_text(aes(label="plabel"))
 +scale_fill_cmap(cmap_name="PiYG",name="Mean\nNosepad\nPosition\nDifference",limits=(-3,3))
 +scale_y_continuous(breaks=range(-3, 4),minor_breaks=None,expand=(0,0))
 +scale_x_continuous(breaks=range(-3, 4),minor_breaks=None,expand=(0,0))
+ylab("Scaling nose pads width (2 mm)")
+xlab("Scaling nose pads width (2 mm)")
 +theme_matplotlib()
 +theme(#legend_position="none",#(0.35,0.22),
        dpi=300,
        figure_size=(3.5,3.5)))

print(gg_post_nosePosit)

#=============================
check_normality=pg.normality(df_data, group='Nosepad_Interval', dv='Nosepad_Comfort_S')
print(check_normality)

check_normality=pg.normality(df_data, group='Nosepad_Interval', dv='Nosepad_Comfort_D')
print(check_normality)

post_noseConfort_S=pg.pairwise_tests(data=df_data,dv="Nosepad_Comfort_S",between="Nosepad_Interval",parametric=False).round(3)
post_noseConfort_D=pg.pairwise_tests(data=df_data,dv="Nosepad_Comfort_D",between="Nosepad_Interval",parametric=False).round(3)

post_noseConfort_S0=pg.pairwise_tukey(data=df_data,dv="Nosepad_Comfort_S",between="Nosepad_Interval").round(3)
post_noseConfort_D0=pg.pairwise_tukey(data=df_data,dv="Nosepad_Comfort_D",between="Nosepad_Interval").round(3)
post_noseConfort_S["diff"]=post_noseConfort_S0["diff"]
post_noseConfort_D["diff"]=post_noseConfort_D0["diff"]

temp_A=post_noseConfort_D["A"].copy()
post_noseConfort_D["A"]=post_noseConfort_D["B"].copy()
post_noseConfort_D["B"]=temp_A.copy()
post_noseConfort_D["diff"]*=-1


post_noseComfort=post_noseConfort_S.append(post_noseConfort_D)
print(post_noseComfort)

post_noseComfort["plabel"]=["ns" if x>0.05  else "*" for x in post_noseComfort["p-unc"]]

gg_post_noseComfort=(ggplot(post_noseComfort,aes('A', 'B'))
 +geom_tile(aes(fill='diff'),color="k",size=0.25)
 +geom_text(aes(label="plabel"))
 +scale_fill_cmap(cmap_name="PiYG",name="Mean\nNosepad\nComfort\nDifference",limits=(-3,3))
 +scale_y_continuous(breaks=range(-3, 4),minor_breaks=None,expand=(0,0))
 +scale_x_continuous(breaks=range(-3, 4),minor_breaks=None,expand=(0,0))
+xlab("Scaling nose pads width (2 mm)")
+ylab("Scaling nose pads width (2 mm)")
 +theme_matplotlib()
 +theme(#legend_position="none",#(0.35,0.22),
        dpi=300,
        figure_size=(3.5,3.5)))

print(gg_post_noseComfort)


#=============================temple==============================
force_interval=0.075
df_data["Int_Force"]=np.round(df_data["Clapping_Force"]/force_interval,0).astype(int)*force_interval
df_data_temple=df_data[(df_data["Int_Force"].values>=0.1) & (df_data["Int_Force"].values<0.6)]

#===============

check_normality=pg.normality(df_data_temple, group='Int_Force', dv='Temple_Fit_S')
print(check_normality)

check_normality=pg.normality(df_data_temple, group='Int_Force', dv='Temple_Fit_D')
print(check_normality)


post_templeFit_S=pg.pairwise_tests(data=df_data_temple,dv="Temple_Fit_S",between="Int_Force",parametric=False).round(3)
post_templeFit_D=pg.pairwise_tests(data=df_data_temple,dv="Temple_Fit_D",between="Int_Force",parametric=False).round(3)


post_templeFit_S0=pg.pairwise_gameshowell(data=df_data_temple,dv="Temple_Fit_S",between="Int_Force").round(3)
post_templeFit_D0=pg.pairwise_gameshowell(data=df_data_temple,dv="Temple_Fit_D",between="Int_Force").round(3)
post_templeFit_S["diff"]=post_templeFit_S0["diff"]
post_templeFit_D["diff"]=post_templeFit_D0["diff"]


temp_A=post_templeFit_D["A"].copy()
post_templeFit_D["A"]=post_templeFit_D["B"].copy()
post_templeFit_D["B"]=temp_A.copy()
post_templeFit_D["diff"]*=-1


post_templeFit=post_templeFit_S.append(post_templeFit_D)
print(post_templeFit)

post_templeFit["plabel"]=["ns" if x>0.05  else "*" for x in post_templeFit["p-unc"]]
num_intervals=df_data_temple["Int_Force"].max()/force_interval

gg_post_noseComfort=(ggplot(post_templeFit,aes('A', 'B'))
 +geom_tile(aes(fill='diff',width=force_interval,height=force_interval),color="k",size=0.25)
 +geom_text(aes(label="plabel"))
 +scale_fill_cmap(cmap_name="PiYG",name="Mean\nTemple\nFit\nDifference",limits=(-3,3))
 +scale_x_continuous(breaks=np.arange(1, num_intervals+1,1)*force_interval,minor_breaks=None,expand=(0,0))
 +scale_y_continuous(breaks=np.arange(1, num_intervals+1,1)*force_interval,minor_breaks=None,expand=(0,0))
 +xlab("Temple clamping force (N)")
 +ylab("Temple clamping force (N)")
 +theme_matplotlib()
 +theme(#legend_position="none",#(0.35,0.22),
        dpi=300,
        axis_text_x= element_text(size = 7),
        axis_text_y= element_text(size = 7),
        figure_size=(3.5,3.5)))

print(gg_post_noseComfort)


#===========

check_normality=pg.normality(df_data_temple, group='Int_Force', dv='Temple_Comfort_S')
print(check_normality)

check_normality=pg.normality(df_data_temple, group='Int_Force', dv='Temple_Comfort_D')
print(check_normality)

post_templeComfort_S=pg.pairwise_tests(data=df_data_temple,dv="Temple_Comfort_S",between="Int_Force",parametric=False).round(3)
post_templeComfort_D=pg.pairwise_tests(data=df_data_temple,dv="Temple_Comfort_D",between="Int_Force",parametric=False).round(3)


post_templeComfort_S0=pg.pairwise_gameshowell(data=df_data_temple,dv="Temple_Comfort_S",between="Int_Force").round(3)
post_templeComfort_D0=pg.pairwise_gameshowell(data=df_data_temple,dv="Temple_Comfort_D",between="Int_Force").round(3)
post_templeComfort_S["diff"]=post_templeComfort_S0["diff"]
post_templeComfort_D["diff"]=post_templeComfort_D0["diff"]


temp_A=post_templeComfort_D["A"].copy()
post_templeComfort_D["A"]=post_templeComfort_D["B"].copy()
post_templeComfort_D["B"]=temp_A.copy()
post_templeComfort_D["diff"]*=-1

post_templeComfort=post_templeComfort_S.append(post_templeComfort_D)
print(post_templeComfort)

post_templeComfort["plabel"]=["ns" if x>0.05  else "*" for x in post_templeComfort["p-unc"]]
num_intervals=df_data_temple["Int_Force"].max()/force_interval

gg_post_noseComfort=(ggplot(post_templeComfort,aes('A', 'B'))
 +geom_tile(aes(fill='diff',width=force_interval,height=force_interval),color="k",size=0.25)
 +geom_text(aes(label="plabel"))
 +scale_fill_cmap(cmap_name="PiYG",name="Mean\nTemple\nComfort\nDifference",limits=(-3,3))
 +scale_x_continuous(breaks=np.arange(1, num_intervals+1,1)*force_interval,minor_breaks=None,expand=(0,0))
 +scale_y_continuous(breaks=np.arange(1, num_intervals+1,1)*force_interval,minor_breaks=None,expand=(0,0))
 +xlab("Temple clamping force (N)")
 +ylab("Temple clamping force (N)")
 +theme_matplotlib()
 +theme(#legend_position="none",#(0.35,0.22),
        dpi=300,
        axis_text_x= element_text(size = 7),
        axis_text_y= element_text(size = 7),
        figure_size=(3.5,3.5)))

print(gg_post_noseComfort)
