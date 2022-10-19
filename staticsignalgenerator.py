from ctypes.wintypes import PLARGE_INTEGER
import itertools
import time as tm
from itertools import count
from signal import signal

import matplotlib.pyplot as plt
import mpld3
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
from matplotlib.animation import FuncAnimation

import streamlit as st
import streamlit.components.v1 as components

st.title("Sampling Studio")
st.sidebar.title("Options")


# org_freq=st.sidebar.slider("Frequency",min_value=0,max_value=100,value=20,step=5)
noise_checkbox=st.sidebar.checkbox("Add noise",value=False) 
sampling_checkbox=st.sidebar.checkbox("Sampling", value=False)
if(sampling_checkbox):
    samp_freq=st.sidebar.slider("Sampling Frequency",min_value=1,max_value=100,value=20)

#wave variables
frequency = st.sidebar.slider('frequency',1, 10, 1, 1)  # freq (Hz)
amplitude=st.sidebar.slider('amplitude',1,10,1,1)
time= np.linspace(0, 3, 1200) #time steps
sine = amplitude * np.sin(2 * np.pi * frequency* time) # sine wave 



snr_db=0
if noise_checkbox:
     snr_db=st.sidebar.number_input("SNR level",value=0,min_value=0,max_value=120,step=5)

st.markdown("""
<tr>
""",True)
st.markdown("""
<style>
.css-d1b1ld.edgvbvh6
{
 visibility:hidden;
}
</style>
""",unsafe_allow_html=True)
st.markdown("""
<style>
.css-12gp8ed.eknhn3m4
{
 visibility:hidden;
}
</style>
""",unsafe_allow_html=True)

st.write('''### Sine Wave''')
# Generating time data using arange function from numpy
# time = np.arange(0, 20*np.pi, 0.01) 

plt.subplot(3,1,1)
#power signal 
power=sine**2
signal_average_power=np.mean(power)
signal_averagePower_db=10*np.log10(signal_average_power)
noise_db=signal_averagePower_db-snr_db
noise_watts=10**(noise_db/10)

mean_noise=0
noise=np.random.normal(mean_noise,np.sqrt(noise_watts),len(sine))
noise_signal=sine+noise
# Finding amplitude at each time
#sampling variables
if (sampling_checkbox):
    T=1/samp_freq 
    n=np.arange(0,3/T)
    nT=n*T
    if(noise_checkbox):
        sine_with_noise=amplitude* np.sin(2 * np.pi * frequency * nT)
        noise=np.random.normal(mean_noise,np.sqrt(noise_watts),len(sine_with_noise))
        sampled_amplitude=noise+sine_with_noise
    else:
        sampled_amplitude=amplitude*np.sin(2 * np.pi * frequency * nT )


# Finally displaying the plot
#plt.show()
def cm_to_inch(value):
    return value/2.54

fig=plt.figure(figsize=(cm_to_inch(60),cm_to_inch(45)))


plt.subplot(211)

# Setting x axis label for the plot
plt.xlabel('Time'+ r'$\rightarrow$')

# Setting y axis label for the plot
plt.ylabel('Sin(time) '+ r'$\rightarrow$')

# Showing grid
plt.grid()

# Highlighting axis at x=0 and y=0
plt.axhline(y=0, color='k')
plt.axvline(x=0, color='k')

# Plotting time vs amplitude using plot function from pyplot
if noise_checkbox:
    plt.plot(time, noise_signal,label='signal with noise')
else:
        plt.plot(time, sine,label='signal')
plt.legend(fontsize=20, loc='upper right')



if(sampling_checkbox):
    plt.subplot(212)
    plt.xlabel('Time'+ r'$\rightarrow$')

# Setting y axis label for the plot
    plt.ylabel('Sin(time) '+ r'$\rightarrow$')
        # Showing grid
    plt.grid()

    # Highlighting axis at x=0 and y=0
    plt.axhline(y=0, color='k')
    plt.axvline(x=0, color='k')
    plt.plot(nT,sampled_amplitude,'r*',label='sampled points')
    plt.plot(nT,sampled_amplitude,label='reconstructed wave')
    plt.legend(fontsize=20, loc='upper right')

st.pyplot(fig)