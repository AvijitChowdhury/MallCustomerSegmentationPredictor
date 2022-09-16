import pickle

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

st.title('Mall Customers Segmentation Prediction')

annualIncome=st.text_input('Annual Income')

spendingScore = st.text_input('Spending Score')


pickled_model = pickle.load(open('model.pkl', 'rb'))


if st.button('Predict'):
    result=pickled_model.predict([[float(annualIncome),float(spendingScore)]])
    st.write("This Customer belongs to cluster no: ", result[0])
     
    if result[0] == 0:
        st.write("Customers with medium annual income and medium annual spend")
    elif result[0]==1:
        st.write("Customers with high annual income but low annual spend")
    elif result[0]==2:
        st.write("Customers with low annual income and low annual spend")
    elif result[0]==3:
        st.write("Customers low annual income but high annual spend")
    elif result[0]==4:
        st.write("Customers with high annual income and high annual spend")
# else:
#     st.write('Error in giving data...please give the correct data!')


image = Image.open('me.jpeg')

st.sidebar.image(image, caption='Sunrise by the mountains')
st.sidebar.write('Project Author: Avijit Chowdhury(CUET)')
