import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import pandas as pd
import seaborn as sns
from skimage.io import imread
from skimage.color import rgb2ycbcr, gray2rgb
from PIL import Image
import streamlit as st
st.set_option('deprecation.showPyplotGlobalUse', False)

df = pd.read_csv('skin_segmentation_data.csv', header=None, delim_whitespace=True)
df.columns = ['B', 'G', 'R', 'skin']


st.title('Human skin segmentation with GMM EM Algorithm')

st.markdown(
    " The Skin Segmentation dataset is constructed over B, G, R color space. Skin and Nonskin dataset is generated"
    "using skin textures from face images of diversity of age, gender, and race people."
    
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
st.image('./Images/factor_plot.jpg')

st.markdown('''
            YCbCr is one of two primary color spaces used to represent digital media, the other is RGB. 
            The difference between YCbCr and RGB is that YCbCr represents color as brightness and two color 
            difference signals, while RGB represents color as red, green and blue. In YCbCr, the Y is the 
            brightness (luma), Cb is blue minus luma (B-Y) and Cr is red minus luma (R-Y).
            
            --
            
            We will extract the Cb and Cr channel values from the RGB values
            ''')


st.markdown('## Factor plot of YCbCr Channel data')
st.image('./Images/cbcr_distribution.jpg')

st.markdown('''
            After that we will split the skin and the non-skin training examples and fit two Gaussian mixture
            models, one on the skin examples and another on the nonskin examples, each using 4 Gaussian components.
            ''')
st.image('./Images/gmm_fitted.jpg')




df['Cb'] = np.round(128 -.168736*df.R -.331364*df.G + .5*df.B).astype(int)
df['Cr'] = np.round(128 +.5*df.R - .418688*df.G - .081312*df.B).astype(int)
df.drop(['B','G','R'], axis=1, inplace=True)



skin_data = df[df.skin==1].drop(['skin'], axis=1).to_numpy()
not_skin_data = df[df.skin==2].drop(['skin'], axis=1).to_numpy()

skin_gmm = GaussianMixture(n_components=4, covariance_type='full').fit(skin_data)
not_skin_gmm = GaussianMixture(n_components=4, covariance_type='full').fit(not_skin_data)

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
def skin_segmentation_using_trained_GMM(image):
    image = imread(image)[...,:3]
    #print(image.shape)
    proc_image = np.reshape(rgb2ycbcr(image), (-1, 3))
    skin_score = skin_gmm.score_samples(proc_image[...,1:])
    not_skin_score = not_skin_gmm.score_samples(proc_image[...,1:])
    result = skin_score > not_skin_score
    result = result.reshape(image.shape[0], image.shape[1])
    result = np.bitwise_and(gray2rgb(255*result.astype(np.uint8)), image)
    return result 

st.markdown('## Image after skin segmentation')

st.image(
    skin_segmentation_using_trained_GMM(image), caption=f"Segmented Image", use_column_width=True)
    

