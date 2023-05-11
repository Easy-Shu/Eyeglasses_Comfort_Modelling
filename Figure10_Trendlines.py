# -*- coding: utf-8 -*-
"""
Created on Thu May 11 19:44:13 2023

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

#============================================nonlinear regression======================= 
#=======nosepad position
degree=1 
force_interval=0.025

polyline_x = np.sort(np.r_[0,np.linspace(-3, 3, 10000)])
df_polylinex=pd.DataFrame(dict(Nosepad_Interval=polyline_x))


df_noseposit=pd.melt(df_data[["ID","Nosepad_Interval",'Nosepad_Position_S','Nosepad_Position_D']],
                   id_vars=["ID","Nosepad_Interval"])

df_noseposit["x1"]=df_noseposit["Nosepad_Interval"].astype(str)
df_noseposit['x1'] = pd.Categorical(df_noseposit['x1'], categories=['-3','-2','-1','0',"1","2","3"])


# print(sm.stats.anova_lm(model,type=1))

# model = AnovaRM(data=df_noseComfort,depvar="value",subject="x1",within=["variable"],aggregate_func='mean').fit()
# print(model)

#model = AnovaRM(data=df_noseComfort,depvar="value",subject="ID",within=["x1","variable"],aggregate_func='mean').fit()
#print(model)

#df_noseposit['x2']= df_noseposit.apply(lambda x: x['Nosepad_Interval']-gap if x['variable']=="Nosepad_Comfort_S" else   x['Nosepad_Interval']+gap, axis=1)

df_noseposit_stat=df_noseposit.groupby(['Nosepad_Interval','variable'], as_index=False).agg(mean=('value','mean'), std=('value', 'std'))

df_noseposit_stat["variable"][df_noseposit_stat["variable"]=="Nosepad_Position_S"]="Static conditions"
df_noseposit_stat["variable"][df_noseposit_stat["variable"]=="Nosepad_Position_D"]="Dynamic conditions"
# degree 2 polynomial fit or quadratic fit

df_temp_D=df_noseposit_stat[df_noseposit_stat['variable']=="Dynamic conditions"]
model_D = ols(formula='mean ~ Nosepad_Interval', data=df_temp_D).fit()#+C(x1):C(variable)
print(model_D.summary())
df_lineD=pd.DataFrame(dict(x=polyline_x,y=model_D.predict(df_polylinex)))
#model_D = np.poly1d(np.polyfit(df_temp_D["Nosepad_Interval"],df_temp_D["mean"], degree))   
#df_lineD=pd.DataFrame(dict(x=polyline_x,y=model_D(polyline_x)))
df_lineD["group"]="Trendline for dynamic conditions"#Cubic 
#print(model_D, np.corrcoef(df_temp_D["mean"],  model_D(df_temp_D["Nosepad_Interval"]))[0,1]**2)

df_temp_S=df_noseposit_stat[df_noseposit_stat['variable']=="Static conditions"]
model_S = ols(formula='mean ~ Nosepad_Interval', data=df_temp_S).fit()#+C(x1):C(variable)
print(model_S.summary())
df_lineS=pd.DataFrame(dict(x=polyline_x,y=model_S.predict(df_polylinex)))
#model_S = np.poly1d(np.polyfit(df_temp_S["Nosepad_Interval"],df_temp_S["mean"], degree))   
#df_lineS=pd.DataFrame(dict(x=polyline_x,y=model_S(polyline_x)))
df_lineS["group"]="Trendline for static conditions"#Cubic 
#print(model_S, np.corrcoef(df_temp_S["mean"],  model_S(df_temp_S["Nosepad_Interval"]))[0,1]**2)

model_G = ols(formula='mean ~ Nosepad_Interval', data=df_noseposit_stat).fit()#+C(x1):C(variable)
print(model_G.summary())
df_lineG=pd.DataFrame(dict(x=polyline_x,y=model_G.predict(df_polylinex)))
#model_Global = np.poly1d(np.polyfit(df_noseposit_stat["Nosepad_Interval"],df_noseposit_stat["mean"], degree))   
#df_lineG=pd.DataFrame(dict(x=polyline_x,y=model_Global(df_polylinex)))
df_lineG["group"]="Trendline for global conditions"#Cubic 
#print(model_Global, np.corrcoef(df_noseposit_stat["mean"],  model_Global(df_noseposit_stat["Nosepad_Interval"]))[0,1]**2)

print(df_lineG.y[np.abs(df_lineG.y).argmin()],
      df_lineG.x[np.abs(df_lineG.y).argmin()])

df_line_NosePosit=df_lineD.append(df_lineS).append(df_lineG)
df_line_NosePosit["type"]="Nose Pads Position"
df_noseposit_stat["type"]="Nose Pads Position"
# g_nosepadposi=(ggplot()
# +geom_point(df_noseposit_stat,aes(x="Nosepad_Interval",y="mean",fill="variable"),shape="o",size=2.55,stroke=0.25)
# #+geom_line(df_lineSD,aes(x="x",y="y",color="group"),size=1)
# +geom_line(df_line_NosePosit,aes(x="x",y="y",color="group"),size=1)
# +scale_fill_manual(values=["#FC8619","#02ACF4"],guide=False)
# +scale_color_manual(values=["#FC8619","#02ACF4"],guide=False)
# +guides(fill = guide_legend( direction = "vertical",nrow=2,title="",order =2),
#         color = guide_legend( direction = "vertical",nrow=2,title="",order =1)
#         )
# +scale_y_continuous(breaks=range(-3, 4),limits =(-3,2.5))
# +scale_x_continuous(breaks=np.arange(-3,4,1))
# #+guide_legend(nrow=2)
# +xlab("Nosepad width difference (2mm)")
# +ylab("Perceived position of nosepad")
# +theme_matplotlib()
# +theme(legend_position="right",#(0.67,0.275),
#        #legend_direction = "vertical",
#        legend_box = "vertical",
#        #legend_box_margin=0,
#        legend_margin =-12,
#        #legend_entry_spacing_y=-10,
#        legend_key_size =3,
#        #aspect_ratio =1.05,
#        #axis_text_x=element_text(size=8),
#        dpi=300,
#        figure_size=(4.,4)))
# print(g_nosepadposi)

#=======nosepad comfort
degree=3
force_interval=0.025
polyline_x = np.sort(np.r_[0,np.linspace(-3, 3, 10000)])
df_polylinex=pd.DataFrame(dict(Nosepad_Interval=polyline_x))

df_noseComfort=pd.melt(df_data[["ID","Nosepad_Interval",'Nosepad_Comfort_S','Nosepad_Comfort_D']],
                   id_vars=["ID","Nosepad_Interval"])

df_noseComfort["x1"]=df_noseComfort["Nosepad_Interval"].astype(str)
df_noseComfort['x1'] = pd.Categorical(df_noseComfort['x1'], categories=['-3','-2','-1','0',"1","2","3"])

# model = ols('value ~ C(x1) + C(variable)', data=df_noseComfort).fit()#+C(x1):C(variable)
# #print(model.summary())
# print(sm.stats.anova_lm(model,type=1))

# model = AnovaRM(data=df_noseComfort,depvar="value",subject="x1",within=["variable"],aggregate_func='mean').fit()
# print(model)

#model = AnovaRM(data=df_noseComfort,depvar="value",subject="ID",within=["x1","variable"],aggregate_func='mean').fit()
#print(model)
gap=0.15
df_noseComfort['x2']= df_noseComfort.apply(lambda x: x['Nosepad_Interval']-gap if x['variable']=="Nosepad_Comfort_S" else   x['Nosepad_Interval']+gap, axis=1)

df_noseComfort_stat=df_noseComfort.groupby(['Nosepad_Interval','variable'], as_index=False).agg(mean=('value','mean'), std=('value', 'std'))

df_noseComfort_stat["variable"][df_noseComfort_stat["variable"]=="Nosepad_Comfort_S"]="Static conditions"
df_noseComfort_stat["variable"][df_noseComfort_stat["variable"]=="Nosepad_Comfort_D"]="Dynamic conditions"
# degree 2 polynomial fit or quadratic fit

df_temp_D=df_noseComfort_stat[df_noseComfort_stat['variable']=="Dynamic conditions"]
#model_D = np.poly1d(np.polyfit(df_temp_D["Nosepad_Interval"],df_temp_D["mean"], degree))   
#df_lineD=pd.DataFrame(dict(x=polyline_x,y=model_D(polyline_x)))
model_D = ols(formula='mean ~ Nosepad_Interval+I(Nosepad_Interval**2)+I(Nosepad_Interval**3)', data=df_temp_D).fit()#+C(x1):C(variable)
print(model_D.summary())
df_lineD=pd.DataFrame(dict(x=polyline_x,y=model_D.predict(df_polylinex)))
df_lineD["group"]="Trendline for dynamic conditions"#Cubic 
#print(model_D, np.corrcoef(df_temp_D["mean"],  model_D(df_temp_D["Nosepad_Interval"]))[0,1]**2)

print(df_lineD.x[df_lineD.y.argmax()],df_lineD.y.max())


df_temp_S=df_noseComfort_stat[df_noseComfort_stat['variable']=="Static conditions"]
#model_S = np.poly1d(np.polyfit(df_temp_S["Nosepad_Interval"],df_temp_S["mean"], degree))   
model_S = ols(formula='mean ~ Nosepad_Interval+I(Nosepad_Interval**2)+I(Nosepad_Interval**3)',data=df_temp_S).fit()#+C(x1):C(variable)
print(model_S.summary())
df_lineS=pd.DataFrame(dict(x=polyline_x,y=model_S.predict(df_polylinex)))
df_lineS["group"]="Trendline for static conditions"#Cubic 
#print(model_S, np.corrcoef(df_temp_S["mean"],  model_S(df_temp_S["Nosepad_Interval"]))[0,1]**2)

print(df_lineS.x[df_lineS.y.argmax()],df_lineS.y.max())


model_G = ols(formula='mean ~Nosepad_Interval+I(Nosepad_Interval**2)+I(Nosepad_Interval**3)', data=df_noseComfort_stat).fit()#+C(x1):C(variable)
print(model_G.summary())
df_lineG=pd.DataFrame(dict(x=polyline_x,y=model_G.predict(df_polylinex)))
# model_Global = np.poly1d(np.polyfit(df_noseComfort_stat["Nosepad_Interval"],df_noseComfort_stat["mean"], degree))   
# df_lineG=pd.DataFrame(dict(x=polyline_x,y=model_Global(polyline_x)))
df_lineG["group"]="Trendline for global conditions"#Cubic 


print(df_lineG.x[df_lineG.y.argmax()],df_lineG.y.max())

print(df_noseComfort_stat.groupby("Nosepad_Interval").mean(),
      df_lineG.x[np.abs(df_lineG.x).argmin()],
      df_lineG.y[np.abs(df_lineG.x).argmin()])
thredhold= df_lineG.y[np.abs(df_lineG.x).argmin()]
print(df_lineG.x[df_lineG.y>=thredhold].min(), df_lineG.x[df_lineG.y>=thredhold].max(),np.ptp(df_lineG.x[df_lineG.y>=thredhold])*2)


df_line_NoseComfort=df_lineD.append(df_lineS).append(df_lineG)
df_line_NoseComfort["type"]="Nose Pads Comfort"
df_noseComfort_stat["type"]="Nose Pads Comfort"

df_noses=df_noseComfort_stat.append(df_noseposit_stat)
df_lines=df_line_NoseComfort.append(df_line_NosePosit)

cat_type = CategoricalDtype(categories=["Nose Pads Position","Nose Pads Comfort"], ordered=True)
df_noses["type"] = df_noses["type"].astype(cat_type)
df_lines["type"] = df_lines["type"].astype(cat_type)

cat_type = CategoricalDtype(categories=["Trendline for global conditions",
                                        "Trendline for dynamic conditions",
                                        "Trendline for static conditions"], ordered=True)
df_lines["group"] = df_lines["group"].astype(cat_type)

g_nosepad=(ggplot()
+geom_point(df_noses,aes(x="Nosepad_Interval",y="mean",fill="variable",shape="variable"),size=2,stroke=0.15)
#+geom_line(df_lineSD,aes(x="x",y="y",color="group"),size=1)
+geom_line(df_lines,aes(x="x",y="y",color="group"),size=0.5)
+scale_fill_manual(values=["#FC8619","#02ACF4"],guide=False)
+scale_color_manual(values=["k","#FC8619","#02ACF4"],guide=False)
+scale_shape_manual(values=["s","o"],guide=False)
+guides(shape = guide_legend( direction = "vertical",nrow=2,title="",order =2),
        color= guide_legend( direction = "vertical",nrow=3,title="",order =1),
        fill=None)
+scale_y_continuous(breaks=range(-3, 4),limits =(-3,2.5))
+scale_x_continuous(breaks=np.arange(-3,4,1),expand=(0,0))
+facet_wrap("type",nrow=2)
#+guide_legend(nrow=2)
+xlab("Nose pads width scale factor (2 mm)")
+ylab("Perception scores")
+theme_matplotlib()
+theme(legend_position="bottom",#(0.67,0.275),
       #legend_direction = "vertical",
       legend_box = "horizontal",
       #legend_box_margin=0,
       legend_margin =-5,
       #legend_entry_spacing_y=-10,
       legend_key_height =15,
       strip_background=element_rect(color="k"),
       strip_text_x = element_text(size = 10),
       #axis_text_x= element_text(size = 6),
       dpi=300,
       figure_size=(5.,5)))
print(g_nosepad)


# g_nosepadcomf=(ggplot()
# +geom_point(df_noseComfort_stat,aes(x="Nosepad_Interval",y="mean",fill="variable"),shape="o",size=2.55,stroke=0.25)
# #+geom_line(df_lineSD,aes(x="x",y="y",color="group"),size=1)
# +geom_line(df_line_NoseComfort,aes(x="x",y="y",color="group"),size=1)
# +scale_fill_manual(values=["#FC8619","#02ACF4"],guide=False)
# +scale_color_manual(values=["#FC8619","#02ACF4"],guide=False)
# +guides(fill = guide_legend( direction = "vertical",nrow=2,title="",order =2),
#         color = guide_legend( direction = "vertical",nrow=2,title="",order =1)
#         )
# +scale_y_continuous(breaks=range(-3, 4),limits =(-3,2.5))
# +scale_x_continuous(breaks=np.arange(-3,4,1))
# #+guide_legend(nrow=2)
# +xlab("Nosepad width difference (2mm)")
# +ylab("Perceived comfort of nosepad")
# +theme_matplotlib()
# +theme(legend_position="right",#(0.67,0.275),
#        #legend_direction = "vertical",
#        legend_box = "vertical",
#        #legend_box_margin=0,
#        legend_margin =-12,
#        #legend_entry_spacing_y=-10,
#        legend_key_size =3,
#        #aspect_ratio =1.05,
#        #axis_text_x=element_text(size=8),
#        dpi=300,
#        figure_size=(4.,4)))
# print(g_nosepadcomf)


#=======temple comfort
degree=3   
force_interval=0.075
polyline_x = np.linspace(force_interval, 0.6, 10000)
df_polylinet=pd.DataFrame(dict(Int_Force=polyline_x))


df_data["Int_Force"]=np.round(df_data["Clapping_Force"]/force_interval,0)*force_interval

df_templeComfort=pd.melt(df_data[["ID","Int_Force",'Temple_Comfort_S','Temple_Comfort_D']],
                   id_vars=["ID","Int_Force"])
df_templeComfort=df_templeComfort[df_templeComfort["Int_Force"].values!=0]

df_templeComfort["x1"]=df_templeComfort["Int_Force"].astype(str)
#df_templeComfort['x1'] = pd.Categorical(df_noseComfort['x1'], categories=['-3','-2','-1','0',"1","2","3"])

# model = ols('value ~ C(Int_Force) + C(variable)', data=df_templeComfort).fit()
# print(sm.stats.anova_lm(model))

#df_templeComfort['x2']= df_templeComfort.apply(lambda x: x['Int_Force']-gap if x['variable']=="Temple_Comfort_S" else   x['Int_Force']+gap, axis=1)

df_templeComfort_stat=df_templeComfort.groupby(['Int_Force','variable'], as_index=False).agg(mean=('value','mean'), std=('value', 'std'))

df_templeComfort_stat["variable"][df_templeComfort_stat["variable"]=="Temple_Comfort_S"]="Static conditions"
df_templeComfort_stat["variable"][df_templeComfort_stat["variable"]=="Temple_Comfort_D"]="Dynamic conditions"
df_templeComfort_stat["type"]="Temple Comfort"
# degree 2 polynomial fit or quadratic fit

df_temp_D=df_templeComfort_stat[df_templeComfort_stat['variable']=="Dynamic conditions"]
# model_D = np.poly1d(np.polyfit(df_temp_D["Int_Force"],df_temp_D["mean"], degree))   
# df_lineD=pd.DataFrame(dict(x=polyline_x,y=model_D(polyline_x)))
model_D = ols(formula='mean ~ Int_Force+I(Int_Force**3)', data=df_temp_D).fit()#+C(x1):C(variable)
print(model_D.summary())
df_lineD=pd.DataFrame(dict(x=polyline_x,y=model_D.predict(df_polylinet)))
df_lineD["group"]="Trendline for dynamic conditions"#Cubic 

df_temp_S=df_templeComfort_stat[df_templeComfort_stat['variable']=="Static conditions"]
model_S = ols(formula='mean ~ Int_Force+I(Int_Force**3)', data=df_temp_S).fit()#+C(x1):C(variable)
print(model_S.summary())
df_lineS=pd.DataFrame(dict(x=polyline_x,y=model_S.predict(df_polylinet)))
df_lineS["group"]="Trendline for static conditions"#Cubic 



model_Global = ols(formula='mean ~ Int_Force+I(Int_Force**3)', data=df_templeComfort_stat).fit()#+C(x1):C(variable)
print(model_Global.summary())
df_lineG=pd.DataFrame(dict(x=polyline_x,y=model_Global.predict(df_polylinet)))
#model_Global = np.poly1d(np.polyfit(df_templeComfort_stat["Int_Force"],df_templeComfort_stat["mean"], degree))   
#df_lineG=pd.DataFrame(dict(x=polyline_x,y=model_Global(polyline_x)))
df_lineG["group"]="Trendline for global conditions"#Cubic 
#print(model_Global, np.corrcoef(df_templeComfort_stat["mean"],  model_Global(df_templeComfort_stat["Int_Force"]))[0,1]**2)
print("best force:", df_lineG[df_lineG.y==df_lineG.y.max()])


print(df_lineG.x[df_lineG.y.argmax()],df_lineG.y.max())

print(df_templeComfort_stat.groupby("Int_Force").mean(),
      df_lineG.x[np.abs(df_lineG.x-0.3).argmin()],
      df_lineG.y[np.abs(df_lineG.x-0.3).argmin()])
thredhold= df_lineG.y[np.abs(df_lineG.x-0.3).argmin()]
print(df_lineG.x[df_lineG.y>=thredhold].min(), df_lineG.x[df_lineG.y>=thredhold].max(),np.ptp(df_lineG.x[df_lineG.y>=thredhold]))


df_templeComfort_Line=df_lineD.append(df_lineS).append(df_lineG)
df_templeComfort_Line["type"]="Temple Comfort"

# g_templeComfort=(ggplot()
# +geom_point(df_templeComfort_stat,aes(x="Int_Force",y="mean",fill="variable"),shape="o",size=2.55,stroke=0.25)
# +geom_line(df_templeComfort_Line,aes(x="x",y="y",color="group"),size=1)
# +scale_fill_manual(values=["#FA6263","#CDDA29"],guide=False)
# +scale_color_manual(values=["#AB141C","#9AA51D"],guide=False)
# +guides(fill = guide_legend( direction = "vertical",nrow=2,title="",order =2),
#         color = guide_legend( direction = "vertical",nrow=2,title="",order =1)
#         )
# +scale_y_continuous(breaks=range(-3, 4),limits =(-3,2.5))
# #+scale_x_continuous(breaks=np.arange(0.1, 0.7, force_interval))
# #+guide_legend(nrow=2)
# +xlab("Temple clapping force (N)")
# +ylab("Perceived comfort of temple")
# +theme_matplotlib()
# +theme(legend_position="right",#(0.57,0.275),
#         #legend_direction = "vertical",
#         legend_box = "vertical",
#         #legend_box_margin=0,
#         legend_margin =-12,
#         #legend_entry_spacing_y=-10,
#         legend_key_size =3,
#         #aspect_ratio =1.05,
#         #axis_text_x=element_text(size=8),
#         dpi=300,
#         figure_size=(4,4)))
# print(g_templeComfort)

#========temple fit
degree=1
#force_interval=0.02

df_data["Int_Force"]=np.round(df_data["Clapping_Force"]/force_interval)*force_interval

df_templeFit=pd.melt(df_data[["ID","Int_Force",'Temple_Fit_S','Temple_Fit_D']],
                   id_vars=["ID","Int_Force"])
df_templeFit=df_templeFit[df_templeFit["Int_Force"].values!=0]

df_templeFit["x1"]=df_templeFit["Int_Force"].astype(str)
#df_templeComfort['x1'] = pd.Categorical(df_noseComfort['x1'], categories=['-3','-2','-1','0',"1","2","3"])

# model = ols('value ~ C(Int_Force) + C(variable)', data=df_templeFit).fit()
# print(sm.stats.anova_lm(model))

#df_templeComfort['x2']= df_templeFit.apply(lambda x: x['Int_Force']-gap if x['variable']=="Temple_Comfort_S" else   x['Int_Force']+gap, axis=1)

df_templeFit_stat=df_templeFit.groupby(['Int_Force','variable'], as_index=False).agg(mean=('value','mean'), std=('value', 'std'))

df_templeFit_stat["variable"][df_templeFit_stat["variable"]=="Temple_Fit_S"]="Static conditions"
df_templeFit_stat["variable"][df_templeFit_stat["variable"]=="Temple_Fit_D"]="Dynamic conditions"
df_templeFit_stat["type"]="Temple Fit"
# degree 2 polynomial fit or quadratic fit

df_temp_D=df_templeFit_stat[df_templeFit_stat['variable']=="Dynamic conditions"]
model_D = np.poly1d(np.polyfit(df_temp_D["Int_Force"],df_temp_D["mean"], degree))   
df_lineD=pd.DataFrame(dict(x=polyline_x,y=model_D(polyline_x)))
df_lineD["group"]="Trendline for dynamic conditions"#Cubic 
print(model_D, np.corrcoef(df_temp_D["mean"],  model_D(df_temp_D["Int_Force"]))[0,1]**2)

df_temp_S=df_templeFit_stat[df_templeFit_stat['variable']=="Static conditions"]
model_S = np.poly1d(np.polyfit(df_temp_S["Int_Force"],df_temp_S["mean"], degree))   
df_lineS=pd.DataFrame(dict(x=polyline_x,y=model_S(polyline_x)))
df_lineS["group"]="Trendline for static conditions"#Cubic 
print(model_S, np.corrcoef(df_temp_S["mean"],  model_S(df_temp_S["Int_Force"]))[0,1]**2)

model_Global = np.poly1d(np.polyfit(df_templeFit_stat["Int_Force"],df_templeFit_stat["mean"], degree))   
df_lineG=pd.DataFrame(dict(x=polyline_x,y=model_Global(polyline_x)))
df_lineG["group"]="Trendline for global conditions"#Cubic 
print(model_Global, np.corrcoef(df_templeFit_stat["mean"],  model_Global(df_templeFit_stat["Int_Force"]))[0,1]**2)

print(df_lineG.y[np.abs(df_lineG.y).argmin()],
      df_lineG.x[np.abs(df_lineG.y).argmin()])

df_templeFit_Line=df_lineD.append(df_lineS).append(df_lineG)
df_templeFit_Line["type"]="Temple Fit"

#=============overall========
df_temple_Lines=df_templeComfort_Line.append(df_templeFit_Line)
df_temple_stats=df_templeComfort_stat.append(df_templeFit_stat)


cat_type = CategoricalDtype(categories=["Temple Fit","Temple Comfort"], ordered=True)
df_temple_stats["type"] = df_temple_stats["type"].astype(cat_type)
df_temple_Lines["type"] = df_temple_Lines["type"].astype(cat_type)

cat_type = CategoricalDtype(categories=["Trendline for global conditions",
                                        "Trendline for dynamic conditions",
                                        "Trendline for static conditions"], ordered=True)
df_temple_Lines["group"] = df_temple_Lines["group"].astype(cat_type)


g_templefit=(ggplot()
+geom_point(df_temple_stats,aes(x="Int_Force",y="mean",fill="variable",shape="variable"),size=2,stroke=0.15)
+geom_line(df_temple_Lines,aes(x="x",y="y",color="group"),size=0.5)
+scale_fill_manual(values=["#FA6263","#CDDA29"],guide=False)
+scale_color_manual(values=["k","#AB141C","#9AA51D"],guide=False)
+scale_shape_manual(values=["s","o"],guide=False)
+guides(shape = guide_legend( direction = "vertical",nrow=2,title="",order =2),
        color= guide_legend( direction = "vertical",nrow=3,title="",order =1),
        fill=None)
#+scale_y_continuous(breaks=range(-3, 4),limits =(-3,2.5))
+facet_wrap("type",nrow=2)
+scale_y_continuous(breaks=range(-3, 4),limits =(-3,2.5))
+scale_x_continuous(breaks=np.arange(force_interval, 0.7, force_interval),minor_breaks=None,expand=(0,0))
#+xlim(0.1,0.55)
#+guide_legend(nrow=2)
+xlab("Temple clamping force (N)")
+ylab("Perception scores")
+theme_matplotlib()
+theme(legend_position="bottom",#(0.67,0.275),
       #legend_direction = "vertical",
       legend_box = "horizontal",
       #legend_box_margin=0,
       legend_margin =-5,
       #legend_entry_spacing_y=-10,
       legend_key_height =15,
       strip_background=element_rect(color="k"),
       strip_text_x = element_text(size = 10),
       #axis_text_x= element_text(size = 6),
       dpi=300,
       figure_size=(5.,5)))
print(g_templefit)

# g1 = pw.load_ggplot(g_templefit, figsize=(2,3))
# g2 = pw.load_ggplot(g_templeComfort, figsize=(2,3))
# g_temple = (g1|g2)
# print(g_temple)
