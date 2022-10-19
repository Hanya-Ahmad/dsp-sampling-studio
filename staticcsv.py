from distutils.command.upload import upload
from email.policy import default
from matplotlib.axis import XAxis,Axis
from matplotlib.patches import Polygon
import streamlit as st 
from streamlit_option_menu import option_menu
import matplotlib.pyplot as plt 
import matplotlib as mpl
from matplotlib.widgets import Slider, Button, RadioButtons
from mpl_toolkits import axisartist
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import itertools
import streamlit.components.v1 as components
import mpld3
from matplotlib.animation import FuncAnimation
from tkinter import HORIZONTAL, Menu
from turtle import color, width
mpl.pyplot.ion()


menus= option_menu(menu_title="Select a page.",options=["Sample","Compose"],default_index=0,orientation=HORIZONTAL)


def generate ():
    global uploaded_file
    uploaded_file = st.file_uploader(label="Upload your CSV file", type=['csv', 'xlsx'])


    global df
    if uploaded_file is not None:

        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            df = pd.read_excel(uploaded_file)


    try:
        interactive_plot(df)

    except Exception as e:
        st.write("You haven't uploaded a signal yet")  
            

           


        
  
  
def interactive_plot(dataframe):
    
      time= df['time'].tolist()
      amplitude = df['amplitude'].tolist()
      col = st.color_picker('Select a plot color')
    
      plot = px.line(dataframe,x=time,y=amplitude,width=800,height=600,title=uploaded_file.name,range_x=[9, 10.2],range_y=[-1,1.5],
       template="plotly_dark")
      plot.update_traces(marker=dict(color=col))

      plot.update_xaxes(title_text='Time')
      plot.update_yaxes(title_text='amplitude')
     
     
      st.plotly_chart(plot)
     



if menus=="Sample":
    generate(),
    
