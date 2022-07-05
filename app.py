import tensorflow as tf
import numpy as np
import streamlit as st
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import urllib

################ PAGE CONFIG ##################
image = Image.open('goodstyle.png')
st.set_page_config(
    layout="centered", 
    page_title="GoodStyle",
    page_icon=image)

@st.cache(allow_output_mutation=True)
def load_data():
    fashion = pd.read_csv("fashion.csv")
    return fashion
fashion = load_data()

@st.cache(allow_output_mutation=True)
def load_data2():
    ratings = pd.read_csv("pake.csv")
    return ratings
ratings = load_data2()

num_fashion = 50

# Load model
model = tf.keras.models.load_model("best_model.hdf5")

#Function Prediction Recommend System
def rec_sys():
    fashion_list = fashion[(fashion['SubCategory']==prediction)].ProductId.sample(num_fashion).values
    user = np.array([user_id for k in range(len(fashion_list))])
    pred = model.predict([user, fashion_list]).reshape(num_fashion)
    top_5_ids = (-pred).argsort()[:5]
    top_5_product_id = fashion_list[top_5_ids]
    top_5_product_rating = pred[top_5_ids]*(ratings.rating.max() - ratings.rating.min()) + ratings.rating.min()

    fig2,ax2=plt.subplots(ncols=5,figsize=(8,5))
    for i,id in enumerate(top_5_product_id):
        a=Image.open(urllib.request.urlopen(fashion['ImageURL'].loc[id]))
        ax2[i].imshow(a)
        ax2[i].set_title('{}\nrating: {:.0f}'.format(fashion['SubCategory'].loc[id],top_5_product_rating[i]))
        ax2[i].axis('off')
    st.pyplot(fig2)

# Homepage
st.title("GoodStyle")
st.subheader('This apps will help you choose your best fashion')

#Header
st.image(image, use_column_width = True, caption='GoodStyle Fashion Recommendation')

# recommend system section
st.subheader('Choose Your Fashion SubCategory')
prediction = st.selectbox("Sub Category", ["Topwear", 'Bottomwear', "Dress", "Innerwear", "Socks", "Apparel Set", "Shoes", "Flip Flops", "Sandal"])

st.subheader('Input Your Number ID')
user_id = st.number_input('Insert User ID')
if prediction is not None: 
    #st.image(prediction, use_column_width='auto')
    btn = st.button('Predict')
    #if btn:
    pred = rec_sys()
        #if pred :
            #st.success('Product Recommendation For You')
        #else:
            #st.error('Error')
else:
    st.write(" ")