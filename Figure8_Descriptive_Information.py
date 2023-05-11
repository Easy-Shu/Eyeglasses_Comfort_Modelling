# -*- coding: utf-8 -*-
"""
Created on Thu May 11 17:27:41 2023

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
# Index(['ID', 'Gender', 'Wearing_Eyeglass_Years', 'Wearing_Eyeglass_Fit',
#        'Test_order', 'Nosepad_Interval', 'Frame-Nosepad_Width',
#        'Force-Nosepad', 'Nosepad_Withs', 'Nosepad_Width', 'Nosepad_Comfort_S',
#        'Nosepad_Position_S', 'OverallNosepad_Comfort_S', 'Nosepad_Comfort_D',
#        'Nosepad_Position_D', 'OverallNosepad_Comfort_D', 'GAP',
#        'Frame_Interval', 'Nosepad_Frame_Width', 'Nosepad\nInterval',
#        'Frame_Width', 'Clapping_Force', 'Int_Force', 'Frame_Widths',
#        'Temple_Comfort_S', 'Temple_Fit_S', 'OverallFrame_Comfort_S',
#        'Temple_Comfort_D', 'Temple_Fit_D', 'OverallFrame_Comfort_D',
#        'Temple_Length', 'Expension\nDistance', 'EI', 'Estimated_Force'],
#       dtype='object')


#===========================nosepad comfort================================
gap=0.15


df_noseComfort=pd.melt(df_data[["test_id","Nosepad_Interval",'Nosepad_Comfort_S','Nosepad_Comfort_D']],
                   id_vars=["test_id","Nosepad_Interval"])

df_noseComfort["x1"]=df_noseComfort["Nosepad_Interval"].astype(str)
df_noseComfort['x1'] = pd.Categorical(df_noseComfort['x1'], categories=['-3','-2','-1','0',"1","2","3"])

df_noseComfort["value"]=df_noseComfort["value"].astype(float)


# df_noseComfort2=df_noseComfort[(df_noseComfort["Nosepad_Interval"]>=-2) & (df_noseComfort["Nosepad_Interval"]<=2)]
# model = pg.rm_anova(dv='value',within=["Nosepad_Interval","variable"], subject='test_id', data=df_noseComfort2, detailed=True)#+C(x1):C(variable)
# #print(model.summary())
# print(model)#sm.stats.anova_lm(model,type=1))


# 
# model = AnovaRM(data=df_noseComfort2,depvar="value",subject="ID",within=["variable","x1"]).fit()#aggregate_func='mean'
# print(model)
#df_noseComfort2=df_noseComfort[(df_noseComfort["Nosepad_Interval"]>=-2) & (df_noseComfort["Nosepad_Interval"]<=2)]
# model = AnovaRM(data=df_noseComfort,depvar="value",subject="ID",within=["x1","variable"],aggregate_func='mean').fit()
# print(model)

df_noseComfort['x2']= df_noseComfort.apply(lambda x: x['Nosepad_Interval']-gap if x['variable']=="Nosepad_Comfort_S" else   x['Nosepad_Interval']+gap, axis=1)

df_noseComfort_stat=df_noseComfort.groupby(['Nosepad_Interval','variable'], as_index=False).agg(mean=('value','mean'), std=('value', 'std'))
df_noseComfort_stat['x2']= df_noseComfort_stat.apply(lambda x: x['Nosepad_Interval']-gap if x['variable']=="Nosepad_Comfort_S" else   x['Nosepad_Interval']+gap, axis=1)

df_noseComfort["variable"][df_noseComfort["variable"]=="Nosepad_Comfort_S"]="Static condition"
df_noseComfort["variable"][df_noseComfort["variable"]=="Nosepad_Comfort_D"]="Dynamic condition"
df_noseComfort_stat["variable"][df_noseComfort_stat["variable"]=="Nosepad_Comfort_S"]="Static condition"
df_noseComfort_stat["variable"][df_noseComfort_stat["variable"]=="Nosepad_Comfort_D"]="Dynamic condition"

jitter= position_jitter(width = 0.1, height = 0.1)

cat_type = CategoricalDtype(categories=["Static condition","Dynamic condition"], ordered=True)
df_noseComfort["variable"] = df_noseComfort["variable"].astype(cat_type)
df_noseComfort_stat["variable"] = df_noseComfort_stat["variable"].astype(cat_type)

gg_nosecomfort=(ggplot()
+geom_boxplot(df_noseComfort,aes(x='x1',y="value",fill="variable"),
              size=0.5,width=0.7)#position = position_dodge(0.7),
#+geom_jitter(df_noseComfort,aes(x='x2',y="value",fill="variable"),position =jitter,shape = "o",size=2,stroke=0.2,alpha=0.65)
#+geom_errorbar(df_noseComfort_stat,aes(x="x2",ymin="mean-std", ymax="mean+std"), width=0.0001,size=0.75)
+geom_line(df_noseComfort_stat,aes(x="Nosepad_Interval+4",y="mean",group="variable"),
           size=2)
+geom_point(df_noseComfort_stat,aes(x="Nosepad_Interval+4",y="mean",group="variable"),
            shape="o",size=2.25,fill="w",stroke=0.5)
+scale_fill_manual(values=["#02ACF4","#FC8619"])#s = 0.90, l = 0.65, h=0.0417,color_space='husl',
                #guide = guide_legend( direction = "vertical",nrow=2,title=""))
+scale_y_continuous(breaks=range(-3, 4),limits =(-3,3.25))
+facet_wrap("variable")
#+scale_x_continuous(breaks=range(-3, 4))
#+guide_legend(nrow=2)
+xlab("Nosepad width scale factor (2 mm)")
+ylab("Perceived comfort of nosepad")
+theme_matplotlib()
+theme(legend_position='none',#(0.65,0.22),
       #aspect_ratio =1.05,
       strip_background=element_rect(color="k"),
       strip_text_x = element_text(size = 10),
       #strip_margin_y=500,
       dpi=300,
       figure_size=(5,3)))
print(gg_nosecomfort)


#=====================nosepad position============================================

df_noseFit=pd.melt(df_data[["test_id","Nosepad_Interval",'Nosepad_Position_S','Nosepad_Position_D']],
                id_vars=["test_id","Nosepad_Interval"])

df_noseFit["x1"]=df_noseFit["Nosepad_Interval"].astype(str)
df_noseFit['x1'] = pd.Categorical(df_noseFit['x1'], categories=['-3','-2','-1','0',"1","2","3"])

df_noseFit["value"]=df_noseFit["value"].astype(float)

#model = ols('value ~ C(x1) + C(variable)', data=df_noseFit).fit()
#print(sm.stats.anova_lm(model))

# df_noseFit2=df_noseFit[(df_noseFit["Nosepad_Interval"]>=-2) & (df_noseFit["Nosepad_Interval"]<=2)]
# model = AnovaRM(data=df_noseFit2,depvar="value",subject="ID",within=["x1","variable"]).fit()
# print(model)

# model = AnovaRM(data=df_noseFit,depvar="value",subject="variable",within=["x1"],aggregate_func='mean').fit()
# print(model)

df_noseFit['x2']= df_noseFit.apply(lambda x: x['Nosepad_Interval']-gap if x['variable']=="Nosepad_Position_S" else   x['Nosepad_Interval']+gap, axis=1)

df_noseFit_stat=df_noseFit.groupby(['Nosepad_Interval','variable'], as_index=False).agg(mean=('value','mean'), std=('value', 'std'))
df_noseFit_stat['x2']= df_noseFit_stat.apply(lambda x: x['Nosepad_Interval']-gap if x['variable']=="Nosepad_Position_S" else   x['Nosepad_Interval']+gap, axis=1)

df_noseFit["variable"][df_noseFit["variable"]=="Nosepad_Position_S"]="Static condition"
df_noseFit["variable"][df_noseFit["variable"]=="Nosepad_Position_D"]="Dynamic condition"
df_noseFit_stat["variable"][df_noseFit_stat["variable"]=="Nosepad_Position_S"]="Static condition"
df_noseFit_stat["variable"][df_noseFit_stat["variable"]=="Nosepad_Position_D"]="Dynamic condition"

jitter= position_jitter(width = 0.1, height = 0.1)

cat_type = CategoricalDtype(categories=["Static condition","Dynamic condition"], ordered=True)
df_noseFit["variable"] = df_noseFit["variable"].astype(cat_type)
df_noseFit_stat["variable"] = df_noseFit_stat["variable"].astype(cat_type)

gg_nosefit=(ggplot()
+geom_boxplot(df_noseFit,aes(x='x1',y="value",fill="variable"),
              size=0.5,width=0.7)#position = position_dodge(0.7),
#+geom_jitter(df_noseComfort,aes(x='x2',y="value",fill="variable"),position =jitter,shape = "o",size=2,stroke=0.2,alpha=0.65)
#+geom_errorbar(df_noseComfort_stat,aes(x="x2",ymin="mean-std", ymax="mean+std"), width=0.0001,size=0.75)
+geom_line(df_noseFit_stat,aes(x="Nosepad_Interval+4",y="mean",group="variable"),
           size=2)
+geom_point(df_noseFit_stat,aes(x="Nosepad_Interval+4",y="mean",group="variable"),
            shape="o",size=2.25,fill="w",stroke=0.5)
+scale_fill_manual(values=["#02ACF4","#FC8619"])#s = 0.90, l = 0.65, h=0.0417,color_space='husl',
                #guide = guide_legend( direction = "vertical",nrow=2,title=""))
+scale_y_continuous(breaks=range(-3, 4),limits =(-3,3.25))
#+scale_x_continuous(breaks=range(-3, 4))
+facet_wrap("variable")
#+guide_legend(nrow=2)
+xlab("Nosepad width scale factor (2 mm)")
+ylab("Perceived positions of nosepad")
+theme_matplotlib()
+theme(legend_position="none",#(0.35,0.22),
      #aspect_ratio =1.05,
      strip_background=element_rect(color="k"),
       strip_text_x = element_text(size = 10),
      dpi=300,
       figure_size=(5,3)))
print(gg_nosefit)


#=========temple comfort
force_interval=0.075
gap=0.01

df_data["Int_Force"]=np.round(df_data["Clapping_Force"]/force_interval,0).astype(int)*force_interval

df_templeComfort=pd.melt(df_data[["ID","Int_Force",'Temple_Comfort_S','Temple_Comfort_D']],
                   id_vars=["ID","Int_Force"])
df_templeComfort=df_templeComfort[(df_templeComfort["Int_Force"].values>=0.1) & (df_templeComfort["Int_Force"].values!=0.6)]

#df_templeComfort["x1"]=["0."+x for x in (df_templeComfort["Int_Force"]*10).astype(int).astype(str)]

df_templeComfort["x1"]=[str(np.round(x,4))[:5] for x in df_templeComfort["Int_Force"]]
#df_templeComfort['x1'] = pd.Categorical(df_noseComfort['x1'], categories=['-3','-2','-1','0',"1","2","3"])

# model = ols('value ~ C(Int_Force) + C(variable)', data=df_templeComfort).fit()
# print(sm.stats.anova_lm(model))
# model = AnovaRM(data=df_templeComfort,depvar="value",subject="ID",within=["x1","variable"],aggregate_func='mean').fit()
# print(model)

#df_templeComfort['x2']= df_templeComfort.apply(lambda x: x['Int_Force']-gap if x['variable']=="Temple_Comfort_S" else   x['Int_Force']+gap, axis=1)

df_templeComfort_stat=df_templeComfort.groupby(['Int_Force','variable'], as_index=False).agg(mean=('value','mean'), std=('value', 'std'))


#df_templeComfort_stat['x2']= df_templeComfort_stat.apply(lambda x: x['Int_Force']-gap if x['variable']=="Temple_Comfort_S" else   x['Int_Force']+gap, axis=1)

df_templeComfort["variable"][df_templeComfort["variable"]=="Temple_Comfort_S"]="Static condition"
df_templeComfort["variable"][df_templeComfort["variable"]=="Temple_Comfort_D"]="Dynamic condition"
df_templeComfort_stat["variable"][df_templeComfort_stat["variable"]=="Temple_Comfort_S"]="Static condition"
df_templeComfort_stat["variable"][df_templeComfort_stat["variable"]=="Temple_Comfort_D"]="Dynamic condition"

jitter= position_jitter(width = gap*0.5, height = 0.1)

cat_type = CategoricalDtype(categories=["Static condition","Dynamic condition"], ordered=True)
df_templeComfort["variable"] = df_templeComfort["variable"].astype(cat_type)
df_templeComfort_stat["variable"] = df_templeComfort_stat["variable"].astype(cat_type)
#df_templeComfort_stat["x_order"]=np.arange(np.unique(df_templeComfort["x1"]).shape[0])

df_templeComfort_stat["x"]=np.repeat(range(int(df_templeComfort_stat.shape[0]/2)),2)+1

gg_templecomfort=(ggplot()
+geom_boxplot(df_templeComfort,aes(x='x1',y="value",fill="variable"),position = position_dodge(0.9),width=0.7,size=0.5)
#+geom_jitter(df_templeComfort,aes(x='x2',y="value",fill="variable"),position =jitter,shape = "o",size=1.5,stroke=0.2,alpha=0.65)
#+geom_errorbar(df_templeComfort_stat,aes(x="x2",ymin="mean-std", ymax="mean+std"), width=0.0001,size=0.75)
+geom_line(df_templeComfort_stat,aes(x="x",y="mean",group="variable"),
           size=2)
+geom_point(df_templeComfort_stat,aes(x="x",y="mean",group="variable"),
            shape="o",size=2.25,fill="w",stroke=0.75)
+scale_fill_manual(values=["#CDDA29","#FA6263"],#s = 0.90, l = 0.65, h=0.0417,color_space='husl',
                guide = guide_legend( direction = "vertical",nrow=2,title=""))
+scale_y_continuous(breaks=range(-3, 4),limits =(-3,3.25))
#+scale_x_continuous(breaks=np.arange(0.1, 0.7,force_interval))
#+guide_legend(nrow=2)
+facet_wrap("variable")
+xlab("Temple clamping force (N)")
+ylab("Perceived comfort of temple")
+theme_matplotlib()
+theme(legend_position="none",#(0.35,0.22),
      #aspect_ratio =1.05,
      strip_background=element_rect(color="k"),
       strip_text_x = element_text(size = 10),
       axis_text_x= element_text(size = 8),
      dpi=300,
       figure_size=(5,3)))
print(gg_templecomfort)


#=========temple fit
df_templeFit=pd.melt(df_data[["ID","Int_Force",'Temple_Fit_S','Temple_Fit_D']],
                   id_vars=["ID","Int_Force"])
df_templeFit=df_templeFit[(df_templeFit["Int_Force"].values>=0.1) & (df_templeFit["Int_Force"].values!=0.6)]

#df_templeFit=df_templeFit[df_templeFit["Int_Force"].values!=0]

df_templeFit["x1"]=[str(np.round(x,4))[:5] for x in df_templeFit["Int_Force"]]
#df_templeFit["x1"]=df_templeFit["Int_Force"].astype(str)
#df_templeComfort['x1'] = pd.Categorical(df_noseComfort['x1'], categories=['-3','-2','-1','0',"1","2","3"])

# model = ols('value ~ C(Int_Force) + C(variable)', data=df_templeFit).fit()
# print(sm.stats.anova_lm(model))

# model = AnovaRM(data=df_templeComfort,depvar="value",subject="ID",within=["x1","variable"],aggregate_func='mean').fit()
# print(model)

df_templeFit['x2']= df_templeFit.apply(lambda x: x['Int_Force']-gap if x['variable']=="Temple_Fit_S" else   x['Int_Force']+gap, axis=1)

df_templeFit_stat=df_templeFit.groupby(['Int_Force','variable'], as_index=False).agg(mean=('value','mean'), std=('value', 'std'))
df_templeFit_stat['x2']= df_templeFit_stat.apply(lambda x: x['Int_Force']-gap if x['variable']=="Temple_Fit_S" else   x['Int_Force']+gap, axis=1)

df_templeFit["variable"][df_templeFit["variable"]=="Temple_Fit_S"]="Static condition"
df_templeFit["variable"][df_templeFit["variable"]=="Temple_Fit_D"]="Dynamic condition"
df_templeFit_stat["variable"][df_templeFit_stat["variable"]=="Temple_Fit_S"]="Static condition"
df_templeFit_stat["variable"][df_templeFit_stat["variable"]=="Temple_Fit_D"]="Dynamic condition"

jitter= position_jitter(width = gap*0.5, height = 0.1)

cat_type = CategoricalDtype(categories=["Static condition","Dynamic condition"], ordered=True)
df_templeFit["variable"] = df_templeFit["variable"].astype(cat_type)
df_templeFit_stat["variable"] = df_templeFit_stat["variable"].astype(cat_type)

df_templeFit_stat["x"]=np.repeat(range(int(df_templeFit_stat.shape[0]/2)),2)+1

gg_templefit=(ggplot()
+geom_boxplot(df_templeFit,aes(x='x1',y="value",fill="variable"),position = position_dodge(0.9),size=0.5)
#+geom_jitter(df_templeFit,aes(x='x2',y="value",fill="variable"),position =jitter,shape = "o",size=1.5,stroke=0.2,alpha=0.65)
#+geom_errorbar(df_templeFit_stat,aes(x="x2",ymin="mean-std", ymax="mean+std"), width=0.0001,size=0.75)
+geom_line(df_templeFit_stat,aes(x="x",y="mean",group="variable"),
           size=2)
+geom_point(df_templeFit_stat,aes(x="x",y="mean",group="variable"),shape="o",fill="w",size=2.25,stroke=0.75)
+scale_fill_manual(values=["#CDDA29","#FA6263"],#s = 0.90, l = 0.65, h=0.0417,color_space='husl',
                guide = guide_legend( direction = "vertical",nrow=2,title=""))
+scale_y_continuous(breaks=range(-3, 4),limits =(-3,3.25))
#+scale_x_continuous(breaks=np.arange(0.1, 0.7, force_interval))
#+guide_legend(nrow=2)
+facet_wrap("variable")
+xlab("Temple clamping force (N)")
+ylab("Perceived fit of temple")
+theme_matplotlib()
+theme(legend_position="none",#(0.35,0.22),
      #aspect_ratio =1.05,
      strip_background=element_rect(color="k"),
       strip_text_x = element_text(size = 10),
       axis_text_x= element_text(size =8),
      dpi=300,
       figure_size=(5,3)))
print(gg_templefit)
