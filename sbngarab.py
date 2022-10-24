import streamlit as st
import pandas as pd
import mpld3
import streamlit.components.v1 as components
import csv
from io import StringIO
import numpy as np
import plotly.graph_objects as go
import math
# from signal_studio_app import *

# fsample = st.slider("Fs",1,300)
time = np.linspace(0,3,1000)
class Functions:
    # st.session_state.sig = pd.DataFrame({})
    numberSignalsAdded=-1
    ADDED_FREQUENCES=[]
    ADDED_AMPLITUDES=[]
    # ADDED_PHASES=[]
    ADDED_SIGNALS=[]
    amplitude_SUM = np.zeros(1000)  # Sum of AMPLITUDE of signals
    Current_amplitude=np.zeros(1000)
    



def ADD_SIGNAL(added_magnitude,added_frequency):
    new_y_amplitude = np.zeros(1000)  # Array for saving sin signals values
    for i in range(1000): 
         new_y_amplitude[i] = (added_magnitude * (math.sin((2 * np.pi * added_frequency * time[i])))) 
    #updating lists
    Functions.numberSignalsAdded+=1
    Functions.ADDED_FREQUENCES.append(added_frequency)
    Functions.ADDED_AMPLITUDES.append(added_magnitude)
    Functions.ADDED_SIGNALS.append(new_y_amplitude)
    Functions.Current_amplitude=np.add(Functions.Current_amplitude,new_y_amplitude)
    return time, Functions.Current_amplitude

def Clean_intialize():
    #number of signals added
    Functions.numberSignalsAdded=-1
    #lists of added features of the signals
    Functions.ADDED_FREQUENCES=[]
    Functions.ADDED_AMPLITUDES=[]
    Functions.ADDED_PHASES=[]
    Functions.ADDED_SIGNALS=[]
    Functions.amplitude_SUM = np.zeros(1000)  # Sum of AMPLITUDE of signals
    Functions.Current_amplitude=np.zeros(1000)
    Functions.Current_amplitude=np.zeros(1000)



def DELETE_SIGNAL(index_todelete): 
    if(Functions.numberSignalsAdded==0):
        Clean_intialize()
        return time, Functions.Current_amplitude
    else:
        print('here')
        Functions.Current_amplitude=np.subtract(Functions.Current_amplitude,Functions.ADDED_SIGNALS[index_todelete]  )
        Functions.ADDED_FREQUENCES.pop(index_todelete)
        Functions.ADDED_AMPLITUDES.pop(index_todelete)
        # Functions.ADDED_PHASES.pop(index_todelete)
        Functions.ADDED_SIGNALS.pop(index_todelete)
        Functions.numberSignalsAdded-=1
        return time, Functions.Current_amplitude


def readCsv(uploaded_file):
    if uploaded_file is not None:
        # To read file as bytes:
        bytes_data = uploaded_file.getvalue()
        # To convert to a string based IO:
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        # To read file as string:
        string_data = stringio.read()
        df =  pd.read_csv(uploaded_file)
        x = df.iloc[:,1]
        y = df.iloc[:,2]
        # st.write(x,y)
        # interpolatedSignal = df.iloc[:,3]
        Functions.Current_amplitude=np.add(Functions.Current_amplitude,y)
        return x,Functions.Current_amplitude


def sampling(fsample,t,sin):
    samp_frq=fsample
    time_range=(max(t)-min(t))
    samp_rate=int((len(t)/time_range)/((fsample)))
    samp_time=t[::samp_rate]
    samp_amp= sin[::samp_rate]
    return samp_time,samp_amp


def sinc_interp(fsample,t,sin):
    samp_time,samp_amp=sampling(fsample,t,sin)
    samp_time=np.array(samp_time)
    samp_amp=np.array(samp_amp)
    # st.write(samp_time)
    time_matrix= np.resize(t,(len(samp_time),len(t)))
    print(time_matrix.size)
    print(samp_time)
    fsample=np.array(fsample)
    t=np.array(t)
    sin=np.array(sin)
    k = (time_matrix.T - samp_time)/(samp_time[1]-samp_time[0])
    resulted_matrix = samp_amp* np.sinc(k)
    reconstucted_seg = np.sum(resulted_matrix, axis=1)
    return  reconstucted_seg,samp_time,samp_amp

def noise(signal,snrRatio):
    power = signal**2
    snr_db = 10 * np.log10(snrRatio)
    signal_avg_power=np.mean(power)
    signal_avg_power_db=10 * np.log10(signal_avg_power)
    noise_db=signal_avg_power_db - snr_db
    noise_watts=10 ** (noise_db/10)
    mean_noise=0
    noise=np.random.normal(mean_noise, np.sqrt(noise_watts),len(signal))
    noise_signal = signal+noise
    return noise_signal

def save(sin):
    signalAndInterpolation = {"x-axis":time, "y-axis":Functions.Current_amplitude,
                               }
    df= pd.DataFrame(signalAndInterpolation)

    st.session_state.sig = df
    


    @st.experimental_memo
    def convert_df(df):
        df.to_csv(index = False).encode('utf-8')

    st.download_button(
    "Press to Download",
    st.session_state.sig.to_csv(),
    "file.csv",
    "text/csv",
    key='download-csv'
    )