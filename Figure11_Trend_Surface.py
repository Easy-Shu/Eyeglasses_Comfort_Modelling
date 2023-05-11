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


#==============================overall comfort==============================

print(np.corrcoef(df_data.Nosepad_Comfort_S,df_data.OverallNosepad_Comfort_S)[0,1],
np.corrcoef(df_data.Nosepad_Comfort_D,df_data.OverallNosepad_Comfort_D)[0,1],
np.corrcoef(df_data.Temple_Comfort_S,df_data.OverallFrame_Comfort_S)[0,1],
np.corrcoef(df_data.Temple_Comfort_D,df_data.OverallFrame_Comfort_D)[0,1])

x=np.arange(-3,3.2,0.2)
y=np.arange(0.075,0.602,0.002)

# x=np.arange(-3,3.2,0.002)
# y=np.arange(0.075,0.602,0.0002)

xx,yy=np.meshgrid(x,y)
df_grid=pd.DataFrame(dict(Nosepad=xx.flatten(),Temple=yy.flatten()))

levels=[-2,-1,0,1,2]
vmin=-3
vmax=2
#=================================================static===========================================
df_subNosepad=df_data[["Nosepad_Interval","Force_Nosepad","OverallNosepad_Comfort_S"]].values
df_subTemple=df_data[["Nosepad\nInterval","Clapping_Force","OverallFrame_Comfort_S"]].values

df_subOverallComfortS=pd.DataFrame(data=np.r_[df_subNosepad,df_subTemple],
                                  columns=["Nosepad","Clapping_Force","Comfort"])

force_interval=0.075

df_subOverallComfortS["Temple"]=np.round(df_subOverallComfortS["Clapping_Force"]/force_interval,0)*force_interval

df_subComfortS=df_subOverallComfortS.groupby(["Nosepad","Temple"],as_index=False).agg(mean=('Comfort','mean'), std=('Comfort', 'std'))
df_subComfortS=df_subComfortS[df_subComfortS["mean"]!=0]  
df_subComfortS=df_subComfortS[df_subComfortS["Temple"]>0]

model_S = ols(formula='mean ~Temple+I(Temple**3)+Nosepad+I(Nosepad**2)+I(Nosepad**3)',data=df_subComfortS).fit()
print(model_S.summary())

df_grid["comfort"]=model_S.predict(df_grid)
gg_mapS=(ggplot(df_grid,aes('Nosepad', 'Temple'))
 +geom_tile(aes(fill='comfort'),color="none",size=0.25)
 +scale_fill_cmap(cmap_name="PiYG",name="Static\nPerceived\nOverall\nComfort",limits=(-3,2))
 +scale_y_continuous(minor_breaks=None,expand=(0,0))
 +scale_x_continuous(breaks=range(-3, 4),minor_breaks=None,expand=(0,0))
+ylab("Clapping force (N)")
+xlab("Scaling nosepad width (2 mm)")
 +theme_matplotlib()
 +theme(#legend_position="none",#(0.35,0.22),
        dpi=300,
        figure_size=(3.5,3.5)))

print(gg_mapS)

z=np.clip(df_grid.comfort.values.reshape(-1,len(x)),-3,3)
z[0]=vmax
fig, ax = plt.subplots(figsize=(5,4),dpi =300)  
CS=ax.contour(x, y[1:], z[1:,:], levels=levels, linewidths=0.5, colors='k')
cntr = ax.contourf(x,y,z, levels=50, cmap="PiYG",vmin=vmin, vmax=vmax)
clb=fig.colorbar(cntr,ax=ax,shrink=0.6,aspect=8,anchor=(0.25,0.2),#label="Global Perceived Overall Comfort",
             ticks=range(vmin, vmax+1,1))     
clb.ax.set_title('Static\nPerceived\nOverall\nComfort',size=10)           
CS.levels =levels# [int(val*10)/10 for val in cntr.levels]
ax.clabel(CS, CS.levels, fmt='%.0f', inline=True,  fontsize=10)
plt.ylabel("Temple clamping force (N)")
plt.xlabel("Nosepad width scale factor (2 mm)")
plt.show()

#=================================================Dynamic===========================================
df_subNosepad=df_data[["Nosepad_Interval","Force_Nosepad","OverallNosepad_Comfort_D"]].values
df_subTemple=df_data[["Nosepad\nInterval","Clapping_Force","OverallFrame_Comfort_D"]].values

df_subOverallComfortD=pd.DataFrame(data=np.r_[df_subNosepad,df_subTemple],
                                  columns=["Nosepad","Clapping_Force","Comfort"])

#force_interval=0.005

df_subOverallComfortD["Temple"]=np.round(df_subOverallComfortD["Clapping_Force"]/force_interval,0)*force_interval

df_subComfortD=df_subOverallComfortD.groupby(["Nosepad","Temple"],as_index=False).agg(mean=('Comfort','mean'), std=('Comfort', 'std'))
df_subComfortD=df_subComfortD[df_subComfortD["mean"]!=0] 
df_subComfortD=df_subComfortD[df_subComfortD["Temple"]>0]

model_D = ols(formula='mean ~Temple+I(Temple**3)+Nosepad+I(Nosepad**2)+I(Nosepad**3)',data=df_subComfortD).fit()
print(model_D.summary())

df_grid["comfort"]=model_D.predict(df_grid)
gg_mapD=(ggplot(df_grid,aes('Nosepad', 'Temple'))
 +geom_tile(aes(fill='comfort'),color="none",size=0.25)
 +scale_fill_cmap(cmap_name="PiYG",name="Dynamic\nPerceived\nOverall\nComfort",limits=(-3,2))
 +scale_y_continuous(minor_breaks=None,expand=(0,0))
 +scale_x_continuous(breaks=range(-3, 4),minor_breaks=None,expand=(0,0))
+ylab("Clapping force (N)")
+xlab("Scaling nosepad width (2 mm)")
 +theme_matplotlib()
 +theme(#legend_position="none",#(0.35,0.22),
        dpi=300,
        figure_size=(3.5,3.5)))

print(gg_mapD)

z=np.clip(df_grid.comfort.values.reshape(-1,len(x)),-3,3)
z[0]=vmax
fig, ax = plt.subplots(figsize=(5,4),dpi =300)  
CS=ax.contour(x, y[1:], z[1:,:], levels=levels, linewidths=0.5, colors='k')
cntr = ax.contourf(x,y,z, levels=50, cmap="PiYG",vmin=vmin, vmax=vmax)
clb=fig.colorbar(cntr,ax=ax,shrink=0.6,aspect=8,anchor=(0.25,0.2),#label="Global Perceived Overall Comfort",
             ticks=range(vmin, vmax+1,1))     
clb.ax.set_title('Dynamic\nPerceived\nOverall\nComfort',size=10)           
CS.levels =levels# [int(val*10)/10 for val in cntr.levels]
ax.clabel(CS, CS.levels, fmt='%.0f', inline=True,  fontsize=10)
plt.ylabel("Temple clamping force (N)")
plt.xlabel("Nosepad width scale factor (2 mm)")
plt.show()
#===============================Global===========================
df_subOverallComfortS["type"]="Static"
df_subOverallComfortD["type"]="Dynamic"
df_subOverallComfortG=df_subOverallComfortS.append(df_subOverallComfortD)

model = ols('Comfort ~ C(Nosepad) + C(Temple)+ C(type)+C(Nosepad)*C(Temple)+C(type)*C(Temple)+C(type)*C(Nosepad)', 
            data=df_subOverallComfortG).fit()
#print(model.summary())
print(sm.stats.anova_lm(model,type=2))

model = ols('Comfort ~ C(Nosepad) + C(Temple)+ C(type)', data=df_subOverallComfortG).fit()
#print(model.summary())
print(sm.stats.anova_lm(model))

df_subComfortS["type"]="Static"
df_subComfortD["type"]="Dynamic"
df_subComfortG=df_subComfortS.append(df_subComfortD)
# model_G = ols(formula='mean ~Temple+I(Temple**3)+Nosepad+I(Nosepad**2)+I(Nosepad**3)+I(Nosepad*Temple)',
#               data=df_subComfortG).fit()
# print(model_G.summary())
model_G = ols(formula='mean ~Temple+I(Temple**3)+Nosepad+I(Nosepad**2)+I(Nosepad**3)',
              data=df_subComfortG).fit()
print(model_G.summary())

df_grid["comfort"]=model_G.predict(df_grid)

print(df_grid["comfort"].max(),df_grid.iloc[df_grid["comfort"].argmax(),:])

gg_map=(ggplot(df_grid,aes('Nosepad', 'Temple'))
 +geom_tile(aes(fill='comfort'),color="none",size=0.25)
 +scale_fill_cmap(cmap_name="PiYG",name="Global\nPerceived\nOverall\nComfort",limits=(-3,2))
 +scale_y_continuous(minor_breaks=None,expand=(0,0))
 +scale_x_continuous(breaks=range(-3, 4),minor_breaks=None,expand=(0,0))
+ylab("Temple clapping force (N)")
+xlab("Nosepad width scale factor (2 mm)")
 +theme_matplotlib()
 +theme(#legend_position="none",#(0.35,0.22),
        dpi=300,
        figure_size=(3.5,3.5)))

print(gg_map)


z=np.clip(df_grid.comfort.values.reshape(-1,len(x)),-3,3)
z[0]=vmax
fig, ax = plt.subplots(figsize=(5,4),dpi =300)  
CS=ax.contour(x, y[1:], z[1:,:], levels=levels, linewidths=0.5, colors='k')
cntr = ax.contourf(x,y,z, levels=50, cmap="PiYG",vmin=vmin, vmax=vmax)
clb=fig.colorbar(cntr,ax=ax,shrink=0.6,aspect=8,anchor=(0.25,0.2),#label="Global Perceived Overall Comfort",
             ticks=range(vmin, vmax+1,1))     
clb.ax.set_title('Global\nPerceived\nOverall\nComfort',size=10)           
CS.levels =levels# [int(val*10)/10 for val in cntr.levels]
ax.clabel(CS, CS.levels, fmt='%.0f', inline=True,  fontsize=10)

plt.ylabel("Temple clamping force (N)")
plt.xlabel("Nosepad width scale factor (2 mm)")
plt.show()
