# -*- coding: utf-8 -*-
"""
Created on Thu May 11 19:53:43 2023


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



#========================================================================
y_n=lambda x: 0.034*x**3-0.221*x**2-0.290*x+1.234

y_t=lambda x: -10135.247*x**3+116.763*x-3.239

y_tx = np.linspace(0.075, 0.6, 10000)/5.83
y_ts=y_t(y_tx)


y_nx = np.linspace(-3, 3, 10000)
y_ns=y_n(y_nx)


df_linesn=pd.DataFrame(dict(x=y_nx,y=y_ns))

th_ns=1.23 
g_nosepad=(ggplot()
+geom_line(df_linesn,aes(x="x",y="y"),size=1)
+geom_hline(yintercept=th_ns, linetype="dashed",  color = "red", size=1)
+annotate('text',x=2,y=-1,label="Prediction\nCurve", color = "k")
+annotate('text',x=2,y=1.65,label="satisfactory\nthreshold", color = "red")
+scale_y_continuous(breaks=range(-3, 4),limits =(-3,3),expand=(0,0))
+scale_x_continuous(breaks=np.arange(-3,4,1),expand=(0,0))
#+facet_wrap("type",nrow=2)
#+guide_legend(nrow=2)
+xlab("Nose pads parameter $(W_N-W_n)/2$ (mm)")
+ylab("Estimated nose pads comfort scores")
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
       figure_size=(4,4)))
print(g_nosepad)

x0=-0.58
y0=1.32
df_pointsn=pd.DataFrame(dict(x=[x0],y=[y0]))
g_nosepad=(ggplot()
+geom_line(df_linesn,aes(x="x",y="y"),size=1)
+geom_hline(yintercept=y0, linetype="dashed",  color = "red", size=1)
+geom_vline(xintercept=x0, linetype="dashed",  color = "red", size=1)
+geom_point(df_pointsn,aes(x="x",y="y"),size=4,shape="o",fill="w",stroke=0.75)
+annotate('text',x=x0+1,y=y0+0.35,label="("+str(x0)+","+str(y0)+")", color = "k")
#+annotate('text',x=2,y=1.65,label="satisfactory\nthreshold", color = "red")
+scale_y_continuous(breaks=range(-3, 4),limits =(-3,3),expand=(0,0))
+scale_x_continuous(breaks=np.arange(-3,4,1),expand=(0,0))
#+facet_wrap("type",nrow=2)
#+guide_legend(nrow=2)
+xlab("Nose pads parameter $(W_N-W_n)/2$ (mm)")
+ylab("Estimated nose pads comfort scores")
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
       figure_size=(4,4)))
print(g_nosepad)


df_linest=pd.DataFrame(dict(x=y_tx,y=y_ts))
th_ts=1.388
#force_interval=0.1

g_nosepad=(ggplot()
+geom_line(df_linest,aes(x="x",y="y"),size=1)
+geom_hline(yintercept=th_ts, linetype="dashed",  color = "red", size=1)
+annotate('text',x=0.03,y=-1.5,label="Prediction\nCurve", color = "k")
+annotate('text',x=0.03,y=1.75,label="satisfactory\nthreshold", color = "red")
+scale_y_continuous(breaks=range(-3, 4),limits =(-3,3),expand=(0,0))
+scale_x_continuous(breaks=np.arange(0.025, 0.1, 0.025),expand=(0,0))#,minor_breaks=None,
#+facet_wrap("type",nrow=2)
#+guide_legend(nrow=2)
+xlab("Temple parameter $F/H_t$ (N/mm)")
+ylab("Estimated temple comfort scores")
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
       figure_size=(4.1,4)))
print(g_nosepad)



x0=0.0617#0.36/5.83
y0=1.58
df_pointst=pd.DataFrame(dict(x=[x0],y=[y0]))
                        
g_nosepad=(ggplot()
+geom_line(df_linest,aes(x="x",y="y"),size=1)
+geom_hline(yintercept=y0, linetype="dashed",  color = "red", size=1)
+geom_vline(xintercept=x0, linetype="dashed",  color = "red", size=1)
+geom_point(df_pointst,aes(x="x",y="y"),size=4,shape="o",fill="w",stroke=0.75)
+annotate('text',x=x0+0.017,y=y0+0.35,label="("+str(x0)+","+str(y0)+")", color = "k")
+scale_y_continuous(breaks=range(-3, 4),limits =(-3,3),expand=(0,0))
+scale_x_continuous(breaks=np.arange(0.025, 0.1, 0.025),expand=(0,0))#,minor_breaks=None,
#+facet_wrap("type",nrow=2)
#+guide_legend(nrow=2)
+xlab("Temple parameter $F/H_t$ (N/mm)")
+ylab("Estimated temple comfort scores")
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
       figure_size=(4.1,4)))
print(g_nosepad)
