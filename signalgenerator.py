from select import select
import streamlit as st
import time
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
frequency = st.slider('frequency',1, 10, 1, 1)  # freq (Hz)
amplitude=st.slider('amplitude',1,10,1,1)
x=np.linspace(0, 3, 1000)

sampling_checkbox=st.checkbox('Sampling')
sampling_frequency=2*frequency;
sampled_x=np.linspace(0,3,1000)
sampled_y=[]
y=[]
n=np.arange(np.ceil(0*sampling_frequency), np.floor(3*sampling_frequency))
T=1/sampling_frequency
nT=[]
if(sampling_checkbox):
    for i in range(0,len(n),1):
        nT.append(n[i]*T)
        sampled_y.append(amplitude*np.sin(2*np.pi*sampling_frequency*n[i]))
else:
    for i in range(0,len(x),1):    
        y.append(amplitude*np.sin(2 * np.pi * frequency*x[i]))
print("x: ", len(x))
print("y: ", len(y))
print("sampled x: ", len(nT))
print("sampled y: ", len(sampled_y))

original_dataframe={'time': x,
'amplitude':y}
sampled_dataframe={'time': nT,
'amplitude': sampled_y}
if(not sampling_checkbox):
    df=pd.DataFrame(original_dataframe)
else:
    df=pd.DataFrame(sampled_dataframe)


#dynamic
timeaxis= df['time'].tolist()
amplitude = df['amplitude'].tolist()
#defining containers
header = st.container()
select_param = st.container()
plot_spot = st.empty()
#select parmeter drop down
with select_param:
    param_lst = list(df.columns)
    param_lst.remove('time')
    select_param = st.selectbox('Select an Amplitude Parameter',   param_lst)
#function to make chart


def make_chart(df, y_col, ymin, ymax, mode):
    fig = go.Figure(layout_yaxis_range=[ymin, ymax])
    fig.add_trace(go.Scatter(x=df['time'], y=df[y_col],           mode=mode))
    fig.update_layout(width=900, height=570, xaxis_title='time',
    yaxis_title=y_col)
    st.write(fig)
    #func call
n = len(df)
ymax = max(df['amplitude'])+0.4
ymin = min(df['amplitude'])-0.4
for i in range(0, n-1, 1):
    df_tmp = df.iloc[i:i+400, :]
    with plot_spot:
        if(sampling_checkbox):
            make_chart(df_tmp,select_param,ymin,ymax,'markers')
        else:
            make_chart(df_tmp, select_param,ymin, ymax,'lines')

    time.sleep(0.03)  