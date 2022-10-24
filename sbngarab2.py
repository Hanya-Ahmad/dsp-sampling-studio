import matplotlib.pyplot as plt
import streamlit as st
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
from streamlit_option_menu import option_menu
from sbngarab import *
st. set_page_config(layout="wide")
st.header('Generation signal')

## plotting figure ##
mainSignal = px.line(height=450,width=1000)
noisedSignal= px.line(height = 450 ,width=1000)
changedSignal= px.line(height = 450 ,width=1000)
######## Side bar components #########
with st.sidebar:
    selected= option_menu(
        menu_title="Main Menu",
        options=["Home","Composer"],
        icons=["house","plus-circle-dotted"],
        menu_icon="cast",
        orientation="horizontal",
        
    )

    deleteList = []
    if len(Functions.ADDED_FREQUENCES) == 0:
        st.write('')
    else:
        for i in range(len(Functions.ADDED_FREQUENCES)):
            deleteList.append(f"amp = {Functions.ADDED_AMPLITUDES[i]} frequency ={Functions.ADDED_FREQUENCES[i]}")
        delList = st.multiselect("Signals to delete", deleteList)
        if st.button("Delete"):
            for i in range(len(delList)):
                if delList[i]:
                    DELETE_SIGNAL(i)
                    mainSignal.add_trace(go.Scatter(x=time,y=Functions.Current_amplitude))





    
if selected =="Home":
    st.header(f" Welcome to our website") 
    st.write("Our website give you a lot of features to do on signal")
    st.write("Our website is designed for generation signal , plotting , sampling , adding nosie , reconstraction , add or remove signal ")
    st.write("The signal can come from loaded file or composer")



if selected =="Composer":
    dataSet = st.selectbox("Which Type of signal you want?",("Loaded Signal","Uploaded Signal"))
    
    if dataSet == "Loaded Signal":
        frequency = st.number_input('Frequency',1,90)
        amplitude = st.number_input('Amplitude', 1,20)
        addSignal = st.button("Add Signal")
        sample = st.checkbox("Sampling")
        if addSignal:
            ADD_SIGNAL(frequency, amplitude)
        mainSignal.add_trace(go.Scatter(x=time,y=Functions.Current_amplitude))
        if sample:
            fsample = st.slider('Fs', 1,20)
            sampleTime,sampleAmp = sampling(fsample, time, Functions.Current_amplitude)
            changedSignal.add_trace(go.Scatter(x=sampleTime,y=sampleAmp, mode="markers"))
            interpolatedSignal,sampleTimePoints,sampleAmpPoints = sinc_interp(fsample,time,Functions.Current_amplitude)
            changedSignal.add_trace(go.Scatter(x=time,y=interpolatedSignal))

    noiseCheckBox = st.checkbox("Add Noise")
    if noiseCheckBox:
        snrRatio = st.number_input('SNR',1,260,1)
        noiseSignal=noise(Functions.Current_amplitude,snrRatio)
        noisedSignal.add_trace(go.Scatter(x=time,y=noiseSignal),row=1,col=1)

    saveCheckBox = st.checkbox("Save File",key = "test_function", value = False)
    if saveCheckBox:
        save(Functions.Current_amplitude)

    if dataSet == "Uploaded Signal":
        uploaded_file = st.file_uploader("Upload your file")
        sample = st.checkbox("Sampling")

        if uploaded_file is not None:
            x,y = readCsv(uploaded_file)
            mainSignal.add_trace(go.Scatter(x=x,y=y))

        if sample:
            fsample = st.slider('Fs', 1,20)
            sampleTime,sampleAmp = sampling(fsample, time, Functions.Current_amplitude)
            changedSignal.add_trace(go.Scatter(x=sampleTime,y=sampleAmp, mode="markers"))
            interpolatedSignal,sampleTimePoints,sampleAmpPoints = sinc_interp(fsample,time,Functions.Current_amplitude)
            changedSignal.add_trace(go.Scatter(x=time,y=interpolatedSignal))

        
        
    st.write(mainSignal)
    st.write(changedSignal)
    st.write(noisedSignal)