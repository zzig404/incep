
import streamlit as st
import cv2
import tensorflow as tf 
import numpy as np
from keras.models import load_model
from PIL import Image
import PIL
from keras.applications.vgg16 import VGG16

#model
#model= load_model('classify.hdf5',compile=(False))
model = VGG16(weights='imagenet')

PAGE_CONFIG = {"page_title":"StColab.io", "page_icon":":smiley:","layout":"centered"}
st.set_page_config("Main Menu")

st.title("Object Detection with InceptionV3")
st.text("Choose UPLOAD on the dropdown menu to upload your own video and search for objects")

# Play video whilist the predictions are loading
video_file = open('./uploads/mosquito_net.mp4', 'rb')
video_bytes = video_file.read()
st.video(video_bytes)



#body and functions of the app
def creating_frames(name):
    vidcap = cv2.VideoCapture(name)
    success,frame = vidcap.read()
    count = 0
    frame_skip =1
    while success:
        success, frame = vidcap.read() # get next frame
        cv2.imwrite(r"./frame%d.jpg" % count, frame) 
        if count % frame_skip == 0: # analyse frames
            print('frame: {}'.format(count)) 
            pil_img = Image.fromarray(frame) # convert frames
            
        if count > 20 :
            break
        count += 1
    converting_vidd()

def converting_vidd():
    idx = tf.io.read_file('frame2.jpg')
    idx = tf.io.decode_image(idx,channels=3) 
    idx = tf.image.resize(idx,[224,224])
    idx = tf.expand_dims(idx, axis=0)
    idx = tf.keras.applications.inception_v3.preprocess_input(idx)
    return idx
    
def predict(idx):
    mod_pred = tf.keras.applications.inception_v3.decode_predictions(model.predict(idx), top=1)
    return mod_pred
    
def main():
    st.sidebar.title("OBJECT Detection.")
    st.title("Object Detection Using Inception V3")
    vid = None

    Search = st.sidebar.text_input("Search for an object",)
    uploaded_video = st.sidebar.file_uploader("Choose Video",type=(['avi','mp4','mov','mkv']))
    if uploaded_video is not None: 
        vid = uploaded_video.name
        with open(vid, mode='wb') as f:
            f.write(uploaded_video.read()) 

        st.sidebar.markdown(f"""
        ### Files
        - {vid}
        """,
        unsafe_allow_html=True) # display file name

        vidcap = cv2.VideoCapture(vid) # load video from disk
        cur_frame = 0
        success = True
        
    if st.sidebar.button("Detect Objects"):
        convert_frames = creating_frames(vid)
        output = converting_vidd()
        results_final = predict(output)
    
        #st.success('The Output is {}'.format(output))
        st.success(results_final)

        
if __name__=='__main__':
    main()
