import pandas as pd
import matplotlib.pyplot as plt 
import plotly.figure_factory as ff
import plotly.express as px
import streamlit as st
import numpy as np
import itertools
import streamlit.components.v1 as components
import mpld3
from matplotlib.animation import FuncAnimation
from itertools import count


df= pd.read_csv('ECG.csv')
st.write(df)
global time
global amplitude
time= df['time'].tolist()
amplitude = df['amplitude'].tolist()
fig = plt.figure()
plt.subplot(211)
plt.plot(time, amplitude,label='ECG Signal')
plt.xlabel('time', fontsize=15)
plt.ylabel('Amplitude', fontsize=15)
plt.legend(fontsize=10, loc='upper right')
plt.xlim(9, 10.2)
plt.ylim(-1, 1.5)
fig_html = mpld3.fig_to_html(fig)
plt.tight_layout()
components.html(fig_html, height=600)

checked=st.sidebar.checkbox("Add noise",value=False)
sampling_checkbox=st.sidebar.checkbox("sampling",value=False) 

snr_db=0
if checked:
     snr_db=st.sidebar.number_input("SNR level",value=0,min_value=0,max_value=150,step=5)

power=[]
for i in range (0,len(amplitude),1):
    power.append(amplitude[i]*2)

signal_average_power=np.mean(power)
signal_averagePower_db=10*np.log10(signal_average_power)
noise_db=signal_averagePower_db-snr_db
noise_watts=10**(noise_db/10)
mean_noise=0
amp_noise=np.random.normal(mean_noise,np.sqrt(noise_watts) ,len(amplitude)).tolist()
final_withnoise=[]
for i in range (0,len(amplitude),1):
    final_withnoise.append(amp_noise[i]+amplitude[i])

plt.subplot(212)
plt.plot(time, final_withnoise, label='ECG with noise')
plt.xlabel('time', fontsize=15)
plt.ylabel('noise amp', fontsize=15)
plt.legend(fontsize=10, loc='upper right')
fig_html = mpld3.fig_to_html(fig)

plt.tight_layout()
def sampling(dataframe):
        frequency=1
        period=1/frequency
        no_cycles=10/period
        freq_sampling=2*frequency
        no_points=dataframe.shape[0]
        points_per_cycle=no_points/no_cycles
        step=points_per_cycle/freq_sampling
        sampling_time=[]
        sampling_amplitude=[]
        for i in range(int(step/2), int(no_points), int(step)):
          sampling_time.append(dataframe.iloc[i, 0])
          sampling_amplitude.append(dataframe.iloc[i, 1])
        sampling_points=pd.DataFrame(
            {"time": sampling_time, "amplitude": sampling_amplitude})
        sampling=px.scatter(
            sampling_points, x=sampling_points.columns[0], y=sampling_points.columns[1], title="sampling")
        sampling.update_traces( marker=dict(size=12, line=dict(width=2, color= 'DarkSlateGrey')),
                                                            selector=dict(mode='markers'))
        st.plotly_chart(sampling, use_container_width=True)
        return sampling_points
if(sampling_checkbox):
        sampling(df)

def sinc_interpolation(signal, sample):
    time=signal.iloc[:, 0]
    sampled_amplitude= sample.iloc[:, 1]
    sampled_time= sample.iloc[:, 0]
    T=(sampled_time[1]-sampled_time[0])
    sincM=np.tile(time, (len(sampled_time), 1))-np.tile(sampled_time[:,np.newaxis],(1, len(time)))
    yNew=np.dot(sampled_amplitude, np.sinc(sincM/T))
    fig, ax= plt.subplots()
    ax.plot(time, yNew, label="Reconstructed signal")
    ax.scatter(sampled_time, sampled_amplitude, color='r', label="sampling points", marker='x')
    fig.legend()
    plt.grid(True)
    plt.title("Reconstructed signal")
    st.pyplot(fig)


    

components.html(fig_html, height=600)
 