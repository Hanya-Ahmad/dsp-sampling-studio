from ctypes.wintypes import PLARGE_INTEGER
import itertools
from itertools import count
from signal import signal

import matplotlib.pyplot as plt
import mpld3
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import streamlit as st
import streamlit.components.v1 as components

from starter import sinc_interpolation


st.title("Sampling Studio")
st.sidebar.title("Options")

#wave variables
st.markdown(
    """
<style>
.sidebar .sidebar-content {
    background-image: linear-gradient(#2e7bcf,#2e7bcf);
    color: ""#FF4B4B";
}
</style>
""",
    unsafe_allow_html=True,
)
frequency = st.sidebar.slider('frequency',1, 10, 1, 1)  # freq (Hz)
amplitude=st.sidebar.slider('amplitude',1,10,1,1)
time= np.linspace(0, 3, 1200) #time steps
sine = amplitude * np.sin(2 * np.pi * frequency* time) # sine wave 
snr_db=0
noise_checkbox=st.sidebar.checkbox("Add noise..",value=False) 
#show snr slider when noise checkbox is true
if noise_checkbox:
     snr_db=st.sidebar.number_input("SNR level",value=20,min_value=0,max_value=120,step=5)
     components.html(
    """
    <script>
    const elements = window.parent.document.querySelectorAll('.stNumberInput div[data-baseweb="input"] > div')
    console.log(elements)
    elements[0].style.backgroundColor ="#F64848"
    </script>
    """,
        height=0,
        width=0,
    )

     
sampling_checkbox=st.sidebar.checkbox("Sampling", value=False)


#noise variables
power=sine**2
signal_average_power=np.mean(power)
signal_averagePower_db=10*np.log10(signal_average_power)
noise_db=signal_averagePower_db-snr_db
noise_watts=10**(noise_db/10)
mean_noise=0
noise=np.random.normal(mean_noise,np.sqrt(noise_watts),len(sine))
noise_signal=sine+noise

#show fs slider when sampling checkbox is true
if(sampling_checkbox):
    samp_freq=st.sidebar.slider("Sampling Frequency",min_value=1,max_value=100,value=20)
    reconstruct_checkbox=st.sidebar.checkbox("reconstruct Sampling Signal", value=False)
    
adding_waves_checkbox=st.sidebar.checkbox("adding waves", value=False)

#if session state has no 'added signals' object then create one 
if 'added_signals' not in st.session_state:
    st.session_state['added_signals'] = []
    st.session_state.frequencies_list=[]
    #if noise checkbox is true then plot the main signal with noise
    if(noise_checkbox):
        signal_label="signal with noise"
        st.session_state.added_signals = [{'name':signal_label,'x':time,'y':noise_signal}]

    #else plot the main signal without noise
    else:
        signal_label="signal"
        st.session_state.added_signals = [{'name':signal_label,'x':time,'y':sine}] 


st.markdown("""
<style>
.css-12gp8ed.eknhn3m4
{
 visibility:hidden;
}
</style>
""",unsafe_allow_html=True)

st.write('''### Sine Wave''')

# function to add a signal
def add_signal(label,x,y):
    st.session_state.added_signals.append({'name':label, 'x':x, 'y':y})
    

#function to remove a signal
def remove_signal(deleted_name):
    for i in range(len(st.session_state.added_signals)):
        if st.session_state.added_signals[i]['name']==deleted_name:
            del st.session_state.added_signals[i]
            break

#sampling code
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
    st.write("len(sampled_amplitude)", len(sampled_amplitude) )
    st.write("len(nT)", len(nT )) 


def sinc_interp(nt_array, sampled_amplitude , time):
    if len(nt_array) != len(sampled_amplitude):
        raise Exception('x and s must be the same length')
    T = (sampled_amplitude[1] - sampled_amplitude[0])
    sincM = np.tile(time, (len(sampled_amplitude), 1)) - np.tile(sampled_amplitude[:, np.newaxis], (1, len(time)))
    yNew = np.dot(nt_array, np.sinc(sincM/T))
    plt.subplot(4,1,2)
    plt.title("Sampled Wave")
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.plot(time,yNew,'r-',label='Reconstructed wave')

#helper function
def cm_to_inch(value):
    return value/2.54

#change plot size
fig=plt.figure(figsize=(cm_to_inch(60),cm_to_inch(45)))

#set plot parameters
plt.subplot(4,1,1)
plt.title("Sine Wave(s)")
plt.xlabel('Time'+ r'$\rightarrow$',fontsize=20)
plt.ylabel('Sin(time) '+ r'$\rightarrow$',fontsize=20)
plt.grid()
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.axhline(y=0, color='k')
plt.axvline(x=0, color='k')

# if noise checkbox is clicked plot noise signal against time
signal_label=""
if (noise_checkbox):
    signal_label="signal with noise"
    plt.plot(time, noise_signal,label=signal_label)
    plt.legend(fontsize=20, loc='upper right')

else:
    signal_label="signal"
    plt.plot(time, sine,label=signal_label)
    plt.legend(fontsize=20, loc='upper right')

#execute sampling function if sampling checkbox is true
if(sampling_checkbox):
    signal_label="sampled points"
    if reconstruct_checkbox:
        st.write("lengths comparison: ", len(sampled_amplitude), len(nT))
        sinc_interp( sampled_amplitude,nT_array , time)
    plt.subplot(4,1,2)
    plt.title("Sampled Wave")
    plt.xlabel('Time'+ r'$\rightarrow$',fontsize=20)
 #Setting y axis label for the plot
    plt.ylabel('Sin(time) '+ r'$\rightarrow$',fontsize=20)
        # Showing grid
    plt.grid()
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    # Highlighting axis at x=0 and y=0
    plt.axhline(y=0, color='k')
    plt.axvline(x=0, color='k')
    plt.stem(nT,sampled_amplitude,'b',label=signal_label,linefmt='b',basefmt=" ")
    plt.legend(fontsize=16, loc='upper right')

#execute adding wave function if adding wave checkbox is true 

if(adding_waves_checkbox):
    
    added_frequency = st.sidebar.slider('frequency for added wave',1, 10, 1, 1)  # freq (Hz)
    added_amplitude=st.sidebar.slider('amplitude for added wave',1,10,1,1)
    added_sine=added_amplitude*np.sin(2*np.pi*added_frequency*time)
    added_label=st.sidebar.text_input(label="enter wave name", max_chars=50)
    add_wave_button=st.sidebar.button("Add Wave")
    
    #call the add_signal function when button is clicked
    if(add_wave_button):
        add_signal(added_label,time,added_sine)
        st.session_state.frequencies_list.append(added_frequency)

st.write("frequencies list: ", st.session_state.frequencies_list)

sum_amplitude=[]

#loop over each item in added_signals and plot them all on the same plot   
added_signals_list=st.session_state.added_signals
remove_options=[]
if(adding_waves_checkbox):
    for dict in added_signals_list:
        remove_options.append(dict['name'])
    if(sampling_checkbox):
        plt.subplot(4,1,4)
    else:
        plt.subplot(4,1,2)
    plt.title("Resulting Signal")
    
    st.write(remove_options)
    print(remove_options)
    if(len(st.session_state.added_signals)>1):
        remove_wave_selectbox=st.sidebar.selectbox('Remove Wave',remove_options)
        remove_wave_button=st.sidebar.button('Remove')
        if(remove_wave_button):
            remove_signal(remove_wave_selectbox)
    plt.xlabel('Time'+ r'$\rightarrow$',fontsize=20)
    plt.ylabel('Sin(time) '+ r'$\rightarrow$',fontsize=20)
    plt.grid()
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.axhline(y=0, color='k')
    plt.axvline(x=0, color='k')
    y0=(added_signals_list[0])['y']
    for index in range(len(y0)):
        sum=0
        for dict in added_signals_list:
            sum+=dict['y'][index]
        sum_amplitude.append(sum)

    plt.plot(time,sum_amplitude,label="total")
    plt.legend()


if(sampling_checkbox & adding_waves_checkbox):
    max_frequency=max(st.session_state.frequencies_list)
    added_samp_frequency=st.sidebar.slider("Sampling frequency for resulting signsl", min_value=0.5*max_frequency, max_value=float(5*max_frequency), step=0.5*max_frequency)
    total_T=1/added_samp_frequency
    total_n=np.arange(0,3/T)
    total_nT=total_n*total_T
    total_nT_array=np.array(total_nT)
    st.write("max freq: ", max_frequency)
    st.write("len(sum_amplitude)", len(sum_amplitude) )
    st.write("len(total_nT)", len(total_nT )) 
    signal_label="sampled points new"
    total_sampled_amplitude=amplitude*np.sin(2 * np.pi * max_frequency * total_nT )
    total_sampled_amplitude_array=np.array(total_sampled_amplitude)
    if reconstruct_checkbox:
        sinc_interp(total_sampled_amplitude,total_nT_array,time)
    else:
         plt.subplot(4,1,3)
         
    plt.title("Sampled Wave")
    plt.xlabel('Time'+ r'$\rightarrow$',fontsize=20)
 #Setting y axis label for the plot
    plt.ylabel('Sin(time) '+ r'$\rightarrow$',fontsize=20)
        # Showing grid
    plt.grid()
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    # Highlighting axis at x=0 and y=0
    plt.axhline(y=0, color='k')
    plt.axvline(x=0, color='k')
    plt.stem(total_nT,total_sampled_amplitude,'b',label=signal_label,linefmt='b',basefmt=" ")
    plt.legend(fontsize=16, loc='upper right')
        
if(len(st.session_state.added_signals)>1):
    for i in range (1,len(st.session_state.added_signals)):
        plt.subplot(4,1,1)
        plt.plot(st.session_state.added_signals[i]['x'], st.session_state.added_signals[i]['y'],
        label=st.session_state.added_signals[i]['name'])
        plt.legend(fontsize=16)
else:
    plt.subplot(4,1,2)
    plt.close()
st.write(st.session_state.added_signals)
st.pyplot(fig)