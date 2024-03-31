import keras
from PIL import Image
import numpy as np
from matplotlib.pyplot import imshow
import streamlit as st


def names(number):
    if number==0:
        return 'Its a Tumor'
    else:
        return 'No, Its not a tumor'


st.title("Brain Tumor Detection")
st.write("Upload a brain MRI Image for tumor detection")
uploaded_file = st.file_uploader("Choose a brain MRI Image ...", type="jpg")
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded MRI.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    x = np.array(img.resize((128,128)))
    x = x.reshape(1,128,128,3)
    model=keras.models.load_model('loc.keras')
    res = model.predict_on_batch(x)
    classification = np.where(res == np.amax(res))[1][0]
    st.write(str(res[0][classification]*100) + '% Confidence This Is A ' + names(classification))
