import streamlit as st 
from streamlit_option_menu import option_menu
import matplotlib.pyplot as plt 
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
from turtle import width


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
        fig = plt.figure()
        plt.subplot(211)
        plt.plot(time, amplitude,label=uploaded_file)
        plt.xlabel('time', fontsize=15)
        plt.ylabel('Amplitude', fontsize=15)
        plt.legend(fontsize=10, loc='upper right')
        plt.xlim(9, 10.2)
        plt.ylim(-1, 1.5)

        fig_html = mpld3.fig_to_html(fig)
        plt.tight_layout()
        components.html(fig_html,height=600)
        
    #comment
        
    except Exception as e:
        st.write("Please upload the signal.")
    



if menus=="Sample":
    generate()