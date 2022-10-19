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

# st.set_option('deprecation.showfileUploaderEncoding', False)
# st.title("data streaming app")
# st.sidebar.subheader("visualisation settings")
# uploaded_file=st.sidebar.file_uploader(label="upload your csv or excel file here", type=['csv', 'xlsx'])

# # global df
# if uploaded_file is not None:
#     print("hello")
#     try:
#         df=pd.read_csv(uploaded_file)
#     except Exception as e:
#         print(e)
#         df=pd.read_csv() 
#         print(df.isnull().sum())
#     else:
#      print("ebn el wskha msh shaghal")

# try:
#     st.write(df)
# except Exception as e:
#     print(e)

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
snr_db=0
if checked:
     snr_db=st.sidebar.number_input("SNR level",value=0,min_value=0,max_value=150,step=5)

amp_noise=np.random.normal(0,0.1 ,len(amplitude)).tolist()
final_withnoise=[]
for i in range (0,len(amplitude),1):
    final_withnoise.append(amp_noise[i]+amplitude[i])

print(len(final_withnoise))
# print(final_withnoise[1])
# print(amplitude[1])
# print(amp_noise[1])
plt.subplot(212)
plt.plot(time, final_withnoise, label='ECG with noise')
plt.xlabel('time', fontsize=15)
plt.ylabel('noise amp', fontsize=15)
plt.legend(fontsize=10, loc='upper right')
fig_html = mpld3.fig_to_html(fig)

plt.tight_layout()

components.html(fig_html, height=600)
