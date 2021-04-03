import numpy as np
from sklearn.mixture import GaussianMixture
from skimage.io import imread
from skimage.color import rgb2ycbcr, gray2rgb
import pandas as pd
import joblib

'''
def read_df():
    
    df = pd.read_csv('skin_segmentation_data.csv', header=None, delim_whitespace=True)
    df.columns = ['B', 'G', 'R', 'skin']
    
    return df 
    

df = read_df()

def extract_YCbCr(df):
    df['Cb'] = np.round(128 -.168736*df.R -.331364*df.G + .5*df.B).astype(int)
    df['Cr'] = np.round(128 +.5*df.R - .418688*df.G - .081312*df.B).astype(int)
    df.drop(['B','G','R'], axis=1, inplace=True)
    return df 

df_data = extract_YCbCr(df)

def fit_GMM_Model(df):
    skin_data = df[df.skin==1].drop(['skin'], axis=1).to_numpy()
    not_skin_data = df[df.skin==2].drop(['skin'], axis=1).to_numpy()
    skin_gmm = GaussianMixture(n_components=4, covariance_type='full').fit(skin_data)
    not_skin_gmm = GaussianMixture(n_components=4, covariance_type='full').fit(not_skin_data)
    return skin_gmm, not_skin_gmm

skin_gmm, not_skin_gmm = fit_GMM_Model(df)


filename = 'nskin_model.joblib.pkl'
_ = joblib.dump(skin_gmm, filename, compress=9)

'''

filename = 'non_skin_model.joblib.pkl'
clf2 = joblib.load(filename)
print(clf2)


