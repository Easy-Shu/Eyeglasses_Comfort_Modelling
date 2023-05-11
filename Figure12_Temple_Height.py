# -*- coding: utf-8 -*-
"""
Created on Thu May 11 19:50:17 2023


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


#====================================temple height analysis================================================

df_head0=pd.read_csv("Head_info.csv")

df_head=df_head0[df_head0.Wearing_Force>0]

a=0.3585/5.83
temp_df=df_head[df_head.Comfort=="Comfortable"]
print(np.corrcoef(temp_df.Wearing_Force,temp_df.Wearing_TempleSideW*a)[0,1])

polyline_x = np.linspace(0, 7, 70)
df_polylinet=pd.DataFrame(dict(x=polyline_x,y=polyline_x*a))
df_polylinet["type"]="Estimated Trendline"


model_temple = ols(formula='Wearing_Force ~ Wearing_TempleSideW-1', data=df_head[df_head.Comfort=="Comfortable"]).fit()#+C(x1):C(variable)
print(model_temple.summary())

df_linetemple=pd.DataFrame(dict(x=polyline_x,y=model_temple.predict(pd.DataFrame(dict(Wearing_TempleSideW=np.linspace(0, 7, 70))))))
df_linetemple["type"]="Fitted Trendline"

df_polylinet=df_polylinet.append(df_linetemple)

g_templeSD=(ggplot()
+geom_point(df_head,aes(x="Wearing_TempleSideW",y="Wearing_Force",fill="Comfort",shape="Comfort"),size=2.55,stroke=0.25)
+geom_line(df_polylinet,aes(x="x",y="y",linetype="type"),size=1)
+scale_fill_manual(values=["#FA6263","#CDDA29"],guide=False)
+scale_color_manual(values=["#AB141C","#9AA51D"],guide=False)
+scale_linetype_manual(values=["solid","dashed"],guide=False)
+scale_shape_manual(values=["o","s"],guide=False)
+guides(shape = guide_legend( direction = "vertical",nrow=2,title="",order =2),
        #color = guide_legend( direction = "vertical",nrow=2,title="",order =1),
        linetype = guide_legend( direction = "vertical",nrow=2,title="",order =1),
        fill=None
        )
+scale_y_continuous(breaks=np.arange(0,0.7,0.1))#,limits =(-3,2.5))
+scale_x_continuous(breaks=np.arange(0, 7.5, 1))
#+guide_legend(nrow=2)
+xlab("Temple height (mm)")
+ylab("Temple clamping force (N)")
+theme_matplotlib()
+theme(legend_position=(0.685,0.81),#"right",
       #legend_text_align="left",
        #legend_direction = "vertical",
        legend_box = "vertical",
        #legend_box_margin=0,
        legend_margin =-15,
        #legend_entry_spacing_y=-10,
        legend_key_size =12,
        #aspect_ratio =1.05,
        #legend_text=element_text(size=9),
        legend_text=element_text(size=8),
        
        #axis_text=element_text(color="k"),
        #axis_title=element_text(color="k"),
        #panel_background=element_rect(size=0.5,color="k"),
        #axis_line = element_line(size=0.5, color="k"),
        #axis_ticks= element_line(size=0.5, color="k"),
  
        dpi=300,
        figure_size=(4,4)))
print(g_templeSD)
