from distutils.command.upload import upload
from email.policy import default
from time import time
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
import plotly
mpl.pyplot.ion()



st.set_page_config(
    
    page_title="Sampling Studio.",
    page_icon="tv",
    layout='centered'
    
)

# st.title("Sampling Sudio Web App.")

# menus= option_menu(menu_title="Select a page.",options=["Sample","Compose"],default_index=0,orientation=HORIZONTAL)
 

# def generate ():
global uploaded_file
uploaded_file = st.file_uploader(label="", type=['csv', 'xlsx'])

global df
global noise_checkbox
global sampling_checkbox
global reconstruction_checkbox
global sampling_freq
global snr_db
global options

if uploaded_file is not None:
    options=st.sidebar.multiselect(label='select csv optins ',options=['sampling','noise','reconstruct'])
    noise_checkbox=st.sidebar.checkbox("Add noise",value=False)
    snr_db=st.sidebar.number_input("SNR level",value=15,min_value=0,max_value=120,step=5)
    sampling_checkbox=st.sidebar.checkbox("sampling",value=False)
    sampling_freq=st.sidebar.slider(label="Sampling frequency",min_value=1,max_value=10,value=5)
    reconstruction_checkbox=st.sidebar.checkbox("reconstruction",value=False)
    try:
        df = pd.read_csv(uploaded_file)
        
        
    except Exception as e:
        df = pd.read_excel(uploaded_file)



            

# st.markdown("<style> ul {display: none;} </style>", unsafe_allow_html=True)

def interactive_plot(dataframe):
    amplitude = df['amplitude'].tolist()
    time = df['time'].tolist()
    # col = st.sidebar.color_picker('Select a plot color','#0827F5')
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
    if('noise' in options):
        fig, ax= plt.subplots()
        ax.plot(time, noise_signal,color='r' ,label="original signal")
        fig.legend()
        ax.set_facecolor("#F3F3E2")
        plt.grid(True)
        plt.xlabel("Time")
        plt.ylabel("amplitude")
        plt.xlim([0, 1])
        plt.ylim([-1, 1])
        if 'sampling' in options:
            pass
        else:    
            st.pyplot(fig)
    else:
        fig, ax= plt.subplots()
        ax.plot(time, amplitude,color='r' ,label="original signal")
        fig.legend()
        plt.grid(True)
        ax.set_facecolor("#F3F3E2")
        plt.xlabel("Time")
        plt.ylabel("amplitude")
        plt.xlim([0, 1])
        plt.ylim([-1, 1])
        if 'sampling' in options:
            pass
        else:    
            st.pyplot(fig)  
    def sampling(dataframe): 
        frequency=sampling_freq
        period=1/frequency
        no_cycles=dataframe.iloc[:,0].max()/period
        freq_sampling=2*frequency
        no_points=dataframe.shape[0]
        points_per_cycle=no_points/no_cycles
        step=points_per_cycle/freq_sampling
        sampling_time=[]
        sampling_amplitude=[]
        for i in range(int(step/2), int(no_points), int(step)):
          sampling_time.append(dataframe.iloc[i, 0])
          sampling_amplitude.append(dataframe.iloc[i, 1])
        global sampling_points
        if('noise' in options):
            sampling_points=pd.DataFrame({"time": sampling_time, "amplitude": noise_signal})
        else:
            sampling_points=pd.DataFrame({"time": sampling_time, "amplitude": sampling_amplitude})

        # plt.scatter(sampling_points.x, sampling_points.y)
        ax.stem(sampling_time, sampling_amplitude,'b',linefmt='b',basefmt=" ",label="sampling points")
        fig.legend()
        if 'reconstruct' in options:
            pass
        else:    
            st.pyplot(fig)
        return sampling_points

    if('sampling' in options):
        sampling(df)

    def sinc_interpolation(signal, sample):
      time = signal.iloc[:, 0]
      sampled_amplitude= sample.iloc[:, 1]
      sampled_time= sample.iloc[:, 0]
      T=(sampled_time[1]-sampled_time[0])
      sincM=np.tile(time, (len(sampled_time), 1))-np.tile(sampled_time[:,np.newaxis],(1, len(time)))
      yNew=np.dot(sampled_amplitude, np.sinc(sincM/T))
      fig, ax= plt.subplots()
      plt.plot(time, yNew,color='k' ,label="Reconstructed signal")
      ax.stem(sampled_time, sampled_amplitude,'b',linefmt='b',basefmt="b",label="sampling points")
      ax.plot(time, amplitude,color='r' ,label="original signal")
      fig.legend()
      ax.set_facecolor("#F3F3E2")
      plt.grid(True)
      plt.title("Signals",fontsize=10)
      plt.xlabel("Time")
      plt.ylabel("amplitude")
      plt.xlim([0, 1])
      plt.ylim([-1, 1])

      st.pyplot(fig)

    if('reconstruct' in options):
        sinc_interpolation(df,sampling_points)
        
     
    
try:
    
    interactive_plot(df)

except Exception as e:
    # st.write("You haven't uploaded a signal yet")  
    print(e)

    

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
optins_sel=st.sidebar.multiselect(label="sin options",options=['sampling','noise','reconstruct'])
# noise_checkbox=st.sidebar.checkbox("Add noise..",value=False) 
# sampling_checkbox=st.sidebar.checkbox("Sampling", value=False)
# reconstruct_checkbox=st.sidebar.checkbox("reconstruct Sampling Signal", value=False)

frequency = st.sidebar.slider('frequency',1, 10, 1, 1)  # freq (Hz)
amplitude=st.sidebar.slider('amplitude',1,10,1,1)
snr_db=st.sidebar.slider("SNR level",value=20,min_value=0,max_value=120,step=5)

time= np.linspace(0, 3, 1200) #time steps
sine = amplitude * np.sin(2 * np.pi * frequency* time) # sine wave 
#show snr slider when noise checkbox is true

samp_freq=st.sidebar.slider("Sampling Frequency",min_value=1,max_value=100,value=20)

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

    


#if session state has no 'added signals' object then create one 
if 'added_signals' not in st.session_state:
    st.session_state['added_signals'] = []
    st.session_state.frequencies_list=[]
    
st.markdown("""
<style>
.css-12gp8ed.eknhn3m4
{
visibility:hidden;
}
</style>
""",unsafe_allow_html=True)

# st.write('''### Sine Wave''')

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
if ('sampling' in optins_sel):
    T=1/samp_freq 
    n=np.arange(0,3/T)
    nT=n*T
    nT_array=np.array(nT)
    if('noise' in optins_sel):
        sine_with_noise=amplitude* np.sin(2 * np.pi * frequency * nT)
        noise=np.random.normal(mean_noise,np.sqrt(noise_watts),len(sine_with_noise))
        sampled_amplitude=noise+sine_with_noise
        sampled_amplitude_array=np.array(sampled_amplitude)

    else:
        sampled_amplitude=amplitude*np.sin(2 * np.pi * frequency * nT )
        sampled_amplitude_array=np.array(sampled_amplitude)

def sinc_interp(nt_array, sampled_amplitude , time):
    if len(nt_array) != len(sampled_amplitude):
        raise Exception('x and s must be the same length')
    T = (sampled_amplitude[1] - sampled_amplitude[0])
    sincM = np.tile(time, (len(sampled_amplitude), 1)) - np.tile(sampled_amplitude[:, np.newaxis], (1, len(time)))
    yNew = np.dot(nt_array, np.sinc(sincM/T))
    # plt.subplot(4,1,2)
    plt.title("Sampled Wave")
    plt.xticks(fontsize=40)
    plt.yticks(fontsize=40)
    plt.plot(time,yNew,'r-',label='Reconstructed wave')
    plt.legend(fontsize=40,loc='upper right')
def sampling(fsample,t,sin):
    samp_frq=fsample
    time_range=(max(t)-min(t))
    samp_rate=int((len(t)/time_range)/((fsample)))
    global samp_time, samp_amp
    samp_time=t[::samp_rate]
    samp_amp= sin[::samp_rate]
    return samp_time,samp_amp

# fsample = st.slider('Fs', 1,20)
#helper function
def cm_to_inch(value):
    return value/2.54

#change plot size
fig=plt.figure()
fig.set_figwidth(40)
fig.set_figheight(70)
#set plot parameters
plt.subplot(4,1,1)
plt.title("Sine Wave(s)")
plt.xlabel('Time'+ r'$\rightarrow$',fontsize=40)
plt.ylabel('Sin(time) '+ r'$\rightarrow$',fontsize=40)
plt.grid(True)
plt.xticks(fontsize=40)
plt.yticks(fontsize=40)
plt.axhline(y=0, color='k')
plt.axvline(x=0, color='k')

# if noise checkbox is clicked plot noise signal against time
signal_label=""

 

#execute sampling function if sampling checkbox is true
if('sampling' in optins_sel):
    signal_label="sampled points"
    if 'reconstruct' in optins_sel :
        sinc_interp( sampled_amplitude,nT_array , time)
    # plt.subplot(4,1,2)
    plt.title("Sampled Wave",fontsize=40)
    plt.xlabel('Time'+ r'$\rightarrow$',fontsize=40)
#Setting y axis label for the plot
    plt.ylabel('Sin(time) '+ r'$\rightarrow$',fontsize=40)
        # Showing grid
    plt.grid(True)
    plt.xticks(fontsize=40)
    plt.yticks(fontsize=40)
    # Highlighting axis at x=0 and y=0
    plt.axhline(y=0, color='k')
    plt.axvline(x=0, color='k')
    plt.stem(nT,sampled_amplitude,'b',label=signal_label,linefmt='b',basefmt=" ")
    plt.legend(fontsize=40, loc='upper right')

#execute adding wave function if adding wave checkbox is true 



added_frequency = st.sidebar.slider('frequency for added wave',1, 10, 1, 1)  # freq (Hz)
added_amplitude=st.sidebar.slider('amplitude for added wave',1,10,1,1)
added_sine=added_amplitude*np.sin(2*np.pi*added_frequency*time)
added_label=st.sidebar.text_input(label="enter wave name", max_chars=50)
add_wave_button=st.sidebar.button("Add Wave")

#call the add_signal function when button is clicked
if(add_wave_button):
    add_signal(added_label,time,added_sine)
    st.session_state.frequencies_list.append(added_frequency)


sum_amplitude=[]

#loop over each item in added_signals and plot them all on the same plot   
added_signals_list=st.session_state.added_signals
remove_options=[]

for dict in added_signals_list:
    remove_options.append(dict['name'])

plt.title("Resulting Signal",fontsize=40)

print(remove_options)
if(len(st.session_state.added_signals)>1):
    remove_wave_selectbox=st.sidebar.selectbox('Remove Wave',remove_options)
    remove_wave_button=st.sidebar.button('Remove')
    if(remove_wave_button):
        remove_signal(remove_wave_selectbox)
plt.xlabel('Time'+ r'$\rightarrow$',fontsize=40)
plt.ylabel('Sin(time) '+ r'$\rightarrow$',fontsize=40)
plt.grid(True)
plt.xticks(fontsize=40)
plt.yticks(fontsize=40)
plt.axhline(y=0, color='k')
plt.axvline(x=0, color='k')
y0=(added_signals_list[0])['y']
for index in range(len(y0)):
    sum=0
    for dict in added_signals_list:
        if(noise_checkbox):
            sum+=dict['y'][index]+noise[index]
        else:
            sum+=dict['y'][index]
    sum_amplitude.append(sum)
sum_amplitude_array=np.array(sum_amplitude)
plt.plot(time,sum_amplitude,label="total")
plt.legend(fontsize=40, loc='upper right')


if(("sampling" in optins_sel ) & len(st.session_state.added_signals)>1):
    max_frequency=max(st.session_state.frequencies_list)
    added_samp_frequency=st.sidebar.slider("Sampling frequency for resulting signsl", min_value=0.5*max_frequency, max_value=float(5*max_frequency), step=0.5*max_frequency)
    sampling(added_samp_frequency, time, sum_amplitude)
    
    if 'reconstruct' in optins_sel:
        sinc_interp(samp_amp,samp_time,time)
    else:
        pass
        # plt.subplot(4,1,3)
        
    plt.title("Sampled Wave",fontsize=40)
    plt.xlabel('Time'+ r'$\rightarrow$',fontsize=40)
#Setting y axis label for the plot
    plt.ylabel('Sin(time) '+ r'$\rightarrow$',fontsize=40)
        # Showing grid
    plt.grid()
    plt.xticks(fontsize=40)
    plt.yticks(fontsize=40)
    # Highlighting axis at x=0 and y=0
    plt.axhline(y=0, color='k')
    plt.axvline(x=0, color='k')
    plt.stem(samp_time, samp_amp,'b',label=signal_label,linefmt='b',basefmt=" ")
    plt.legend(fontsize=16, loc='upper right')
        
if(len(st.session_state.added_signals)>1):
    for i in range (1,len(st.session_state.added_signals)):
        # plt.subplot(4,1,1)
        plt.plot(st.session_state.added_signals[i]['x'], st.session_state.added_signals[i]['y'],
        label=st.session_state.added_signals[i]['name'])
        plt.legend(fontsize=25, loc='upper right')
else:
    # plt.subplot(4,1,2)
    plt.close()
st.pyplot(fig)

# if menus=="Compose":
#     generate_2()


# if menus=="Sample":
#     generate()
