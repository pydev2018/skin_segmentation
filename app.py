import numpy as np
#import matplotlib as mpl
#import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import pandas as pd
import seaborn as sns
from skimage.io import imread
from skimage.color import rgb2ycbcr, gray2rgb
import PIL 
Image = PIL.Image
import streamlit as st
st.set_option('deprecation.showPyplotGlobalUse', False)



st.set_page_config(
    page_title=" Skin Segmentation",
    page_icon=":shark",  # EP: how did they find a symbol?
    
    
)

@st.cache(allow_output_mutation=True)
def read_df():

    df = pd.read_csv('skin_segmentation_data.csv', header=None, delim_whitespace=True)
    df.columns = ['B', 'G', 'R', 'skin']
    
    return df 
    

df = read_df()

st.title('Human skin segmentation with GMM EM Algorithm')

st.markdown(
    
    "The skin dataset is collected by randomly sampling B,G,R values from face images of various age groups (young, middle, and old)"
    "race groups (white, black, and asian), and genders obtained from FERET database and PAL database. Total learning sample size is 245057; out of which 50859 is \
    the skin samples and 194198 is non-skin samples. Color FERET Image Database"
    "the dataset is constructed over B, G, R color space. Skin and Nonskin dataset is generated \
    using skin textures from face images of diversity of age, gender, and race people."
    
    "The dataset is collect from UCI repository  "
    "[ Link to data ](https://archive.ics.uci.edu/ml/datasets/skin+segmentation)"
)


st.image('./Images/uci.jpg')

st.markdown('### 5 samples taken randomly from the data')
st.table(df.sample(5))

st.markdown(
    '### Methodology '
)

st.markdown('## Factor plot of RGB Channel data')
st.image('./Images/factor_plot.JPG')

st.markdown('''
            YCbCr is one of two primary color spaces used to represent digital media, the other is RGB. 
            The difference between YCbCr and RGB is that YCbCr represents color as brightness and two color 
            difference signals, while RGB represents color as red, green and blue. In YCbCr, the Y is the 
            brightness (luma), Cb is blue minus luma (B-Y) and Cr is red minus luma (R-Y).
            
            --
            
            We will extract the Cb and Cr channel values from the RGB values please press the button to extract YCbCr data
            ''')

@st.cache
def extract_YCbCr(df):
    df['Cb'] = np.round(128 -.168736*df.R -.331364*df.G + .5*df.B).astype(int)
    df['Cr'] = np.round(128 +.5*df.R - .418688*df.G - .081312*df.B).astype(int)
    df.drop(['B','G','R'], axis=1, inplace=True)
    return df 
    

if st.button("Extract YCbCr Data"):
    df_data = extract_YCbCr(df)
    st.markdown("### 5 samples taken randomly from the data")
    st.table(df_data.sample(5))
    

st.markdown('## Factor plot of YCbCr Channel data')
st.image('./Images/cbcr_distribution.JPG')

st.markdown('''
            After that we will split the skin and the non-skin training examples and fit two Gaussian mixture
            models, one on the skin examples and another on the nonskin examples, each using 4 Gaussian components.
            ''')


@st.cache
def fit_GMM_Model(df):
    skin_data = df[df.skin==1].drop(['skin'], axis=1).to_numpy()
    not_skin_data = df[df.skin==2].drop(['skin'], axis=1).to_numpy()
    skin_gmm = GaussianMixture(n_components=4, covariance_type='full').fit(skin_data)
    not_skin_gmm = GaussianMixture(n_components=4, covariance_type='full').fit(not_skin_data)
    return skin_gmm, not_skin_gmm
    
    


if st.button("Fit GMM Model"):
    global skin_gmm
    global not_skin_gmm
    skin_gmm, not_skin_gmm = fit_GMM_Model(df)
    
    
    #print(skin_gmm, not_skin_gmm)
    st.markdown("### Model fitted")
    

st.image('./Images/gmm_fitted.JPG')



st.markdown('###  We can use our trained GMM model to perform skin segmentation on any picture with skin exposure \
            the model has now learned about the difference between pixels with skin and without skin' )



st.markdown('## Upload any image on which you wish to perform skin segmentation')
img_file_buffer = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if img_file_buffer is not None:
    image = (img_file_buffer)
else:
    image = ('./Images/demo.jpg')
    
st.image(
    image, caption=f"Input Image", use_column_width=True,
)


@st.cache
def segment_skin_from_image(image, skin_gmm, not_skin_gmm):
    image = imread(image)[...,:3]
    #print(image.shape)
    proc_image = np.reshape(rgb2ycbcr(image), (-1, 3))
    #print(proc_image.shape)
    skin_score = skin_gmm.score_samples(proc_image[...,1:])
    not_skin_score = not_skin_gmm.score_samples(proc_image[...,1:])
    result = skin_score > not_skin_score
    result = result.reshape(image.shape[0], image.shape[1])
    result = np.bitwise_and(gray2rgb(255*result.astype(np.uint8)), image)
    return result 
    



if st.button("Segment skin from image"):
    skin_gmm, not_skin_gmm = fit_GMM_Model(df)
    result = segment_skin_from_image(image, skin_gmm, not_skin_gmm)
    st.markdown('## Image after skin segmentation')
    st.image(result, caption=f"Segmented Image", use_column_width=True) 
    





    

