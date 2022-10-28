from distutils.command.upload import upload
import streamlit as st 
import pandas as pd
from streamlit_option_menu import option_menu
import matplotlib.pyplot as plt 
import numpy as np


def read_csv():
    uploaded_file = st.file_uploader(label="", type=['csv', 'xlsx'])
    if uploaded_file is not None:
        
        options=st.sidebar.multiselect(label='select csv optins ',options=['sampling','noise','reconstruct'])
        snr_db=st.sidebar.slider("SNR level",value=15,min_value=0,max_value=120,step=5)
        sampling_freq=st.sidebar.slider(label="Sampling frequency",min_value=1,max_value=10,value=5)
        
        try:
            df = pd.read_csv(uploaded_file)

        except Exception as e:
            df = pd.read_excel(uploaded_file)

                

    # st.markdown("<style> ul {display: none;} </style>", unsafe_allow_html=True)

    def interactive_plot(df):
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

        