from distutils.command.upload import upload
import streamlit as st 
from streamlit_option_menu import option_menu
import matplotlib.pyplot as plt 
import numpy as np
# css-hxt7ib e1fqkh3o4
st.set_page_config(
    layout='wide'
)

import pandas as pd
st.markdown("""
        <style>
                .css-hxt7ib{
                    margin-top: -55px;
                    }
                
                .css-1q6lfs0 eczokvf0, .withScreencast  {
                    padding-top: 0rem;
                    padding-bottom: 0rem;
                    padding-left: 5rem;
                    padding-right: 5rem;
                }
               .css-18e3th9 {
                    padding-top: 0rem;
                    padding-bottom: 0rem;
                    padding-left: 5rem;
                    padding-right: 5rem;
                }
               .css-1d391kg {
                    padding-top: 3.5rem;
                    padding-right: 1rem;
                    padding-bottom: 3rem;
                    padding-left: 1rem;
                }
        </style>
        """, unsafe_allow_html=True)
global df
global sampling_freq
global snr_db
global csv_options
sidebar_col1,sidebar_col2 = st.sidebar.columns((125,125))
left, middle,right = st.columns(3)
sidebar_col1.title("Signal Viewer Controls")
sidebar_col2.title("File and Viewer Controls")
uploaded_file = st.file_uploader(label="", type=['csv', 'xlsx'])
noise_checkbox=sidebar_col2.checkbox(label='Noise')
sampling_checkbox=sidebar_col2.checkbox(label='Sampling')
reconstruct_checkbox=sidebar_col2.checkbox(label='Reconstruction')
snr_db=sidebar_col2.slider("SNR",value=15,min_value=0,max_value=120,step=5)
if uploaded_file is not None:
    
   
    sampling_freq=sidebar_col2.slider(label="Sampling Frequency",min_value=float(1),max_value=float(10),value=float(5))
    try:
        df = pd.read_csv(uploaded_file)
        
        
    except Exception as e:
        df = pd.read_excel(uploaded_file)


def csv_plot(df):
    amplitude = df['amplitude'].tolist()
    time = df['time'].tolist()

    power=df['amplitude']**2
    signal_average_power=np.mean(power)
    signal_averagePower_db=10*np.log10(signal_average_power)
    noise_db=signal_averagePower_db-snr_db
    noise_watts=10**(noise_db/10)
    mean_noise=0
    noise=np.random.normal(mean_noise,np.sqrt(noise_watts),len(df['amplitude']))

    #resulting signal with noise
    noise_signal=df['amplitude']+noise
    dataframe_noise=pd.DataFrame({"time": time, "amplitude": noise_signal})

    if noise_checkbox:
        fig, ax= plt.subplots(figsize=(8,4))
        ax.plot(time, noise_signal,color='gray' ,label="Original Signal")
        plt.grid(True)
        plt.xlabel("Time")
        plt.ylabel("amplitude")
        plt.xlim([0, 1])
        plt.ylim([-1, 1])
        if sampling_checkbox==0:
            fig.legend(fontsize=8.5, bbox_to_anchor=(0.9, 0.9))
            st.pyplot(fig)
    else:
        fig, ax= plt.subplots(figsize=(8,4))
        ax.plot(time, amplitude,color='gray' ,label="Original Signal")
        plt.grid(True)
        plt.xlabel("Time")
        plt.ylabel("amplitude")
        plt.xlim([0, 1])
        plt.ylim([-1, 1])
        if sampling_checkbox==0:
            fig.legend(fontsize=8.5, bbox_to_anchor=(0.9, 0.9))
            st.pyplot(fig)  
    def csv_sampling(dataframe): 
        frequency=sampling_freq
        period=1/frequency
        no_cycles=dataframe.iloc[:,0].max()/period
        nyquist_freq=2*frequency
        no_points=dataframe.shape[0]
        points_per_cycle=no_points/no_cycles
        step=points_per_cycle/nyquist_freq
        sampling_time=[]
        sampling_amplitude=[]
        for i in range(int(step/2), int(no_points), int(step)):
          sampling_time.append(dataframe.iloc[i, 0])
          sampling_amplitude.append(dataframe.iloc[i, 1])
        global sampling_points
        sampling_points=pd.DataFrame({"time": sampling_time, "amplitude": sampling_amplitude})
        ax.stem(sampling_time, sampling_amplitude,'k',linefmt='k',basefmt=" ",label="Sampling Points")
        fig.legend(fontsize=8.5, bbox_to_anchor=(0.9, 0.9))
        if reconstruct_checkbox==0:
            st.pyplot(fig)

        return sampling_points

    if sampling_checkbox:
        if noise_checkbox:
          csv_sampling(dataframe_noise)
        else:
          csv_sampling(df)


    def csv_interpolation(signal, sample):
      time = signal.iloc[:, 0]
      sampled_amplitude= sample.iloc[:, 1]
      sampled_time= sample.iloc[:, 0]
      T=(sampled_time[1]-sampled_time[0])
      sincM=np.tile(time, (len(sampled_time), 1))-np.tile(sampled_time[:,np.newaxis],(1, len(time)))
      yNew=np.dot(sampled_amplitude, np.sinc(sincM/T))
      fig, ax= plt.subplots(figsize=(8,4))
      plt.plot(time, yNew,color='orange' ,label="Reconstructed Signal")
      ax.stem(sampled_time, sampled_amplitude,'k',linefmt='k',basefmt="k",label="Sampling Points")
      if noise_checkbox:
         ax.plot(time, noise_signal,color='gray' ,label="Original Signal")
      else:
        ax.plot(time, amplitude,color='gray' ,label="Original Signal")
      fig.legend(fontsize=7.5, bbox_to_anchor=(0.91, 0.94))
      plt.grid(True)
      plt.title("Uploaded Signal",fontsize=10)
      plt.xlabel("Time")
      plt.ylabel("amplitude")
      plt.xlim([0, 1])
      plt.ylim([-1, 1])

      st.pyplot(fig)

    if reconstruct_checkbox:
      if noise_checkbox:
           csv_interpolation(dataframe_noise,sampling_points)
      else:
          csv_interpolation(df,sampling_points)


     
    
try:
    
    csv_plot(df)

except Exception as e:
    print(e)

    
#start of composer code



time= np.linspace(-3, 3, 1200) #time steps
sine =  np.cos(2 * np.pi *1* time) # sine wave 
#show snr slider when noise checkbox is true


#noise variables
power=sine**2
signal_average_power=np.mean(power)
signal_averagePower_db=10*np.log10(signal_average_power)
noise_db=signal_averagePower_db-snr_db
noise_watts=10**(noise_db/10)
mean_noise=0
noise=np.random.normal(mean_noise,np.sqrt(noise_watts),len(sine))
noise_signal=sine+noise

if 'added_signals' not in st.session_state:
    st.session_state['added_signals'] = []
    st.session_state.frequencies_list=[1]
    signal_label="First Signal"
  



# function to add a signal
def add_signal(label,x,y):   
    st.session_state.added_signals.append({'name':label, 'x':x, 'y':y})
   

#function to remove a signal
def remove_signal(deleted_name):
    for i in range(len(st.session_state.added_signals)):
        if st.session_state.added_signals[i]['name']==deleted_name:
            del st.session_state.added_signals[i]
            break


def composer_interpolation(nt_array, sampled_amplitude , time):
    if len(nt_array) != len(sampled_amplitude):
        raise Exception('x and s must be the same length')
    T = (sampled_amplitude[1] - sampled_amplitude[0])
    sincM = np.tile(time, (len(sampled_amplitude), 1)) - np.tile(sampled_amplitude[:, np.newaxis], (1, len(time)))
    yNew = np.dot(nt_array, np.sinc(sincM/T))
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    
    ax.plot(time,yNew,'orange',label='Reconstructed Signal')
    if(added_frequency>1 ):
        ax.legend(loc="upper right",fontsize=8.5,bbox_to_anchor=(0.4, 0.8))
    elif(added_frequency>1 and added_amplitude>1):
        ax.legend(fontsize=8.5,bbox_to_anchor=(0.4, 0.8),loc="upper right")
    else:    
         ax.legend(loc="upper right",fontsize=8.5,bbox_to_anchor=(1.1, 1.05))
    
def composer_sampling(fsample,t,sin):
    time_range=(max(t)-min(t))
    samp_rate=int((len(t)/time_range)/((fsample)))
    global samp_time, samp_amp
    samp_time=t[::samp_rate]
    samp_amp= sin[::samp_rate]
    return samp_time,samp_amp

fig=plt.figure()
#change plot size
fig.set_figwidth(40)
fig.set_figheight(70)
#set plot parameters
fig, ax = plt.subplots(figsize=(8.5, 5))
plt.title("Signal Viewer")
plt.xlabel('Time'+ r'$\rightarrow$',fontsize=10)
plt.ylabel('cos(time) '+ r'$\rightarrow$',fontsize=10)
plt.grid(True)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.axhline(y=0, color='k')
plt.axvline(x=0, color='k')

if 'default_wave_name' not in st.session_state:
    st.session_state.default_wave_name=1

added_frequency = sidebar_col1.slider('Frequency',1, 10, 1, 1)  # freq (Hz)
added_amplitude=sidebar_col1.slider('Amplitude',1,10,1,1)
added_sine=added_amplitude*np.cos(2*np.pi*added_frequency*time)
added_label=sidebar_col1.text_input(label="Wave Name",value=str(st.session_state.default_wave_name) ,max_chars=50)
add_wave_button=sidebar_col1.button("Add Wave")
if(add_wave_button):
    st.session_state.default_wave_name+=1


#call the add_signal function when button is clicked
if(add_wave_button):
    add_signal(added_label,time,added_sine)
    st.session_state.frequencies_list.append(added_frequency)
    

#loop over each item in added_signals and plot them all on the same plot   
added_signals_list=st.session_state.added_signals
remove_options=[]


for dict in added_signals_list:
    remove_options.append(dict['name'])


if(len(st.session_state.added_signals)>1):
    remove_wave_selectbox=sidebar_col1.selectbox('Remove Wave',remove_options)
    remove_wave_button=sidebar_col1.button('Remove')
    if(remove_wave_button):
        remove_signal(remove_wave_selectbox)
plt.xlabel('Time'+ r'$\rightarrow$',fontsize=10)
plt.ylabel('cos(time) '+ r'$\rightarrow$',fontsize=10)
plt.grid(True)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.axhline(y=0, color='k')
plt.axvline(x=0, color='k')

sum_amplitude=[]
if(uploaded_file is None):
    if(len(st.session_state.added_signals)>0):
        y0=(st.session_state.added_signals[0])['y']
        for index in range(len(y0)):
            sum=0
            for dict in st.session_state.added_signals:
                if noise_checkbox:
                    sum+=dict['y'][index]+noise[index]
                else:
                    sum+=dict['y'][index]
            sum_amplitude.append(sum)
        #execute sampling function if sampling checkbox is true

    if (len(st.session_state.added_signals)>=0):
        signal_label="Sampling Points"
        max_frequency=max(st.session_state.frequencies_list)
        if(uploaded_file is None):
            added_samp_frequency=sidebar_col2.slider("Sampling Frequency (Hz)", min_value=float(0.5*max_frequency), max_value=float(5*max_frequency), step=float(1), value=float(2*max_frequency))
            with sidebar_col2:
                st.write("Fs= ",str(added_samp_frequency/max_frequency)+"Fmax")
            composer_sampling(added_samp_frequency, time, sum_amplitude)

        if sampling_checkbox:
            
            plt.xlabel('Time'+ r'$\rightarrow$',fontsize=10)
            #Setting y axis label for the plot
            plt.ylabel('cos(time) '+ r'$\rightarrow$',fontsize=10)
                # Showing grid
            plt.grid(True)
            plt.xticks(fontsize=10)
            plt.yticks(fontsize=10)
            # Highlighting axis at x=0 and y=0
            plt.axhline(y=0, color='k')
            plt.axvline(x=0, color='k')
            ax.stem(samp_time, samp_amp,'k',label="Sampling points",linefmt='k',basefmt=" ")
            ax.legend(loc="upper right",fontsize=8.5, bbox_to_anchor=(1.1, 1.05))
                
            T=1/added_samp_frequency
            n=np.arange(0,3/T)
            nT=n*T
            nT_array=np.array(nT)
            if reconstruct_checkbox:
                composer_interpolation(samp_amp,samp_time,time)
            
            if noise_checkbox:
                sine_with_noise=added_amplitude* np.cos(2 * np.pi * max_frequency * nT)
                noise=np.random.normal(mean_noise,np.sqrt(noise_watts),len(sine_with_noise))
                sampled_amplitude=noise+sine_with_noise
                

            else:
                sampled_amplitude=added_amplitude*np.cos(2 * np.pi * max_frequency * nT )
                


    sum_amplitude_array=np.array(sum_amplitude)
    if(len(st.session_state.added_signals)):
        ax.plot(time,sum_amplitude,color='royalblue',label="Summation Signal")
        plt.xlim([0, 1])
     
    else:
        fig, ax= plt.subplots(figsize=(8,4))
        # ax.legend(fontsize=8.5, bbox_to_anchor=(1.1, 1.05))
        plt.grid(True)
        plt.xlabel("Time")
        plt.ylabel("amplitude")
        plt.xlim([0, 1])
        plt.ylim([-1, 1])
    if(len(st.session_state.added_signals)>0):
        ax.legend(loc="upper right",fontsize=8.5, bbox_to_anchor=(1.1, 1.05))


        for i in range (0,len(st.session_state.added_signals)):
            ax.plot(st.session_state.added_signals[i]['x'], st.session_state.added_signals[i]['y'],
            label=st.session_state.added_signals[i]['name'])
            ax.legend(loc="upper right",fontsize=8.5, bbox_to_anchor=(1.1, 1.05))

    if uploaded_file is None:
        st.pyplot(fig)
