from distutils.command.upload import upload
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
    uploaded_file = st.file_uploader(label="Upload your CSV file", type=['csv', 'xlsx'])


    global df
    if uploaded_file is not None:

        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            df = pd.read_excel(uploaded_file)
            st.write("Please upload a valid CSV file.")

    try:
       
        
        time= df['time'].tolist()
        amplitude = df['amplitude'].tolist()
        fig, ax = plt.subplots()
        ax.plot(time, amplitude,color='b')

        ax.grid(True, linestyle='-.')
        ax.tick_params(labelcolor='r', labelsize='medium', width=3)
        ax.set_title(uploaded_file.name)
        
        plt.xlim(9, 10.2)
        plt.ylim(-1, 1.5)
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.style.use('dark_background')
        # ax.xaxis.zoom(3)
        
        with plt.ion():
        # interactive mode will be on
        # figures will automatically be shown
         fig2 = plt.figure()

        plt.show()
        st.write(fig)



      
       
        # # plt.subplot(211)
    
        
        # # plt.legend(fontsize=10, loc='upper right')
        
        
        # plt.style.use('dark_background')


        
        # plt.plot(time, amplitude)
        
        
        
    #comment
        
    except Exception as e:
        st.write("Please upload the signal.")
    



if menus=="Sample":
    generate()