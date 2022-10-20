from distutils.command.upload import upload
from email.policy import default
from matplotlib.axis import XAxis,Axis
from matplotlib.patches import Polygon
from pyparsing import line
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



st.set_page_config(
    
    page_title="Sampling Studio.",
    page_icon="tv",
    layout='centered'
    
)

st.title("Sampling Sudio Web App.")

menus= option_menu(menu_title="Select a page.",options=["Sample","Compose"],default_index=0,orientation=HORIZONTAL)
 

def generate ():
    global uploaded_file
    uploaded_file = st.file_uploader(label="Upload your CSV file", type=['csv', 'xlsx'])

    global df
    global noise_checkbox
    if uploaded_file is not None:
        noise_checkbox=st.checkbox("Add noise",value=False)
        
        try:
            df = pd.read_csv(uploaded_file)
            
            
        except Exception as e:
            df = pd.read_excel(uploaded_file)


    try:
        
        interactive_plot(df)

    except Exception as e:
        st.write("You haven't uploaded a signal yet")  
        print(e)
            

           


        
  
  
def interactive_plot(dataframe):
    snr_db=0
    if(noise_checkbox):
        snr_db=st.number_input("SNR level",value=0,min_value=0,max_value=120,step=5)
    amplitude = df['amplitude'].tolist()
    col = st.color_picker('Select a plot color','#0827F5')
    mean=df['amplitude'].mean()
    std_deviation=df['amplitude'].std()
    power=df['amplitude']**2
    signal_average_power=np.mean(power)
    signal_averagePower_db=10*np.log10(signal_average_power)
    noise_db=signal_averagePower_db-snr_db
    noise_watts=10**(noise_db/10)
    mean_noise=0
    noise=np.random.normal(mean_noise,np.sqrt(noise_watts),len(df['amplitude']))

    #resulting signal with noise
    noise_signal=df['amplitude']+noise
    if(noise_checkbox):
        plot = px.line(dataframe,x=time,y=noise_signal,width=800,height=600,title=uploaded_file.name,range_x=[9, 10.2],range_y=[-1,1.5], template="plotly_dark")
    else:
        plot = px.line(dataframe,x=time,y=amplitude,width=800,height=600,title=uploaded_file.name,range_x=[9, 10.2],range_y=[-1,1.5], template="plotly_dark")
    plot.update_traces(line=dict(color=col))
    plot.update_xaxes(title_text='Time')
    plot.update_yaxes(title_text='amplitude')
     
     
    st.plotly_chart(plot)
     



if menus=="Sample":
    generate()