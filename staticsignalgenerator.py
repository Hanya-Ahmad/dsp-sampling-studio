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
"st.session_state object:", st.session_state
if 'added_signals' not in st.session_state:
    	st.session_state.added_signals = []


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

st.write("First")
st.write(st.session_state)


#power signal 
power=sine**2
signal_average_power=np.mean(power)
signal_averagePower_db=10*np.log10(signal_average_power)
noise_db=signal_averagePower_db-snr_db
noise_watts=10**(noise_db/10)

mean_noise=0
noise=np.random.normal(mean_noise,np.sqrt(noise_watts),len(sine))
noise_signal=sine+noise

if(noise_checkbox):
    st.session_state.added_signals = [{'name':'main','x':time,'y':noise_signal}]
else:
   st.session_state.added_signals = [{'name':'main','x':time,'y':sine}] 

#add a signal

def add_signal(label,x,y):
    st.session_state.added_signals.append({'name':label, 'x':x, 'y':y})
    

#remove a signal
def remove_signal(deleted_name):
    for i in range(len(st.session_state.added_signals)):
        if st.session_state.added_signals[i]['name']==deleted_name:
            del st.session_state.added_signals[i]
            break

adding_waves_checkbox=st.sidebar.checkbox("adding waves", value=False)

    #add_signal(added_label,time,added_sine)
# Finding amplitude at each time
#sampling variables
if (sampling_checkbox):
    T=1/samp_freq 
    n=np.arange(0,3/T)
    nT=n*T
    nT_array=np.array(nT)
    if(noise_checkbox):
        sine_with_noise=amplitude* np.sin(2 * np.pi * frequency * nT)
        noise=np.random.normal(mean_noise,np.sqrt(noise_watts),len(sine_with_noise))
        sampled_amplitude=noise+sine_with_noise
        sampled_amplitude_array=np.array(sampled_amplitude)

    else:
        sampled_amplitude=amplitude*np.sin(2 * np.pi * frequency * nT )
        sampled_amplitude_array=np.array(sampled_amplitude)

   

def sinc_interp(nt_array, sampled_amplitude , time):
    # if len(nt_array) != len(sampled_amplitude):
    #     raise Exception('x and s must be the same length')
    T = (sampled_amplitude[1] - sampled_amplitude[0])
    sincM = np.tile(time, (len(sampled_amplitude), 1)) - np.tile(sampled_amplitude[:, np.newaxis], (1, len(time)))
    yNew = np.dot(nt_array, np.sinc(sincM/T))
    plt.subplot(212)
    plt.plot(time,yNew,'r-')

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

# if(adding_waves_checkbox):
# plt.subplot(212)
#plt.plot(time, final_added_wave,label='added signals') '''

if(sampling_checkbox):
    sinc_interp(nT_array, sampled_amplitude , time)
    plt.xlabel('Time'+ r'$\rightarrow$')
 #Setting y axis label for the plot
    plt.ylabel('sampled wave'+ r'$\rightarrow$')
        # Showing grid
    plt.grid()

    # Highlighting axis at x=0 and y=0
    plt.axhline(y=0, color='k')
    plt.axvline(x=0, color='k')
    # plt.plot(nT,sampled_amplitude,'g-', label='reconstructed wave')
    plt.legend(fontsize=20, loc='upper right')
#print(st.session_state.added_signals)
if(adding_waves_checkbox):
    added_frequency = st.sidebar.slider('frequency for added wave',1, 10, 1, 1)  # freq (Hz)
    added_amplitude=st.sidebar.slider('amplitude for added wave',1,10,1,1)
    added_sine=added_amplitude*np.sin(2*np.pi*added_frequency*time)
    added_label=str(np.random.normal(0,100))
    add_wave_button=st.sidebar.button("Add Wave")
    if(add_wave_button):
        add_signal(added_label,time,added_sine)
for signal in st.session_state.added_signals:
    plt.subplot(211)
    plt.plot(signal['x'], signal['y'],
            label=signal['name'])
    plt.legend()
   
st.write(st.session_state)


st.pyplot(fig)