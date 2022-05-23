import cv2 as cv
import streamlit as st
import numpy as np
from PIL import Image



def canny(img):
    canny = cv.Canny(img ,  100, 175)
    return canny

def hsv(img):
    hsv = cv.cvtColor(img , cv.COLOR_BGR2HSV)
    return hsv

def brighten(image, amount):
    img_bright = cv.convertScaleAbs(image, beta=amount)
    return img_bright


def blur(image, amount):
    blur_img = cv.GaussianBlur(image, (7, 7), amount)
    return blur_img


def enhance(img):
    hdr = cv.detailEnhance(img, sigma_s=14, sigma_r=0.10)
    return hdr

def grey(img):
    grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    return grey

def obj_detect_img(img):
    proc = img
    grey_img = cv.cvtColor(proc, cv.COLOR_BGR2GRAY)

    face_smile_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
    smiles = face_smile_cascade.detectMultiScale(grey_img , 1.12 , 3)

    for (a,b,c,d) in smiles:
        cv.rectangle(proc , (a,b) , (a+c, b+d) , (0,255,0) ,2)

    return proc



# make india flag overlay function for filters


##################

def main_loop():
    st.title("OpenCV x StreamLit")

    st.sidebar.title("Select Activity")
    choice  = st.sidebar.selectbox("MODE",("Filters","Object Detection(Image)","About"))
   
    if choice == "Filters" :
        apply_enhancement_filter = st.sidebar.checkbox('Enhance Details')
        apply_grey_filter = st.sidebar.checkbox('See in greyscale')
        apply_canny_filter = st.sidebar.checkbox('Canny edges')
        apply_hsv_filter = st.sidebar.checkbox('HSV')
        
        blur_rate = st.sidebar.slider("Blurring", min_value=0.5, max_value=3.5)
        brightness_amount = st.sidebar.slider("Brightness", min_value=-50, max_value=50, value=0)
        
        image_file = st.file_uploader("Upload Your Image", type=['jpg', 'png', 'jpeg'])
        if not image_file:
            return None

        original_image = Image.open(image_file)
        original_image = np.array(original_image)

        processed_image = blur(original_image, blur_rate)
        processed_image = brighten(processed_image, brightness_amount)

        if apply_enhancement_filter:
            processed_image = enhance(processed_image)
        if apply_grey_filter:
            processed_image = grey(processed_image)
        if apply_canny_filter:
            processed_image = canny(processed_image)
        if apply_hsv_filter:
            processed_image = hsv(processed_image)

        st.text("Original vs Processed")
        st.image([original_image, processed_image])

    elif choice == "Object Detection(Image)":
        #st.subheader("Object Detection")
        image_file = st.file_uploader("Upload Your Image", type=['jpg', 'png', 'jpeg'])
        if not image_file:
            return None

        original_image = Image.open(image_file)
        original_image = np.array(original_image)

 
        processed_image = obj_detect_img(original_image)


        st.text("Processed")
        st.image(processed_image)
            
    elif choice == "About":
        st.subheader('This is a project illustrating the use of OpenCV to perform alterations to images and detect objects')


    # image_file = st.file_uploader("Upload Your Image", type=['jpg', 'png', 'jpeg'])
    # if not image_file:
    #     return None

    # original_image = Image.open(image_file)
    # original_image = np.array(original_image)


    # if apply_enhancement_filter:
    #     processed_image = enhance(processed_image)
    # if apply_grey_filter:
    #     processed_image = grey(processed_image)


    # processed_image = blur(original_image, blur_rate)
    # processed_image = brighten(processed_image, brightness_amount)


    # st.text("Original vs Processed")
    # st.image([original_image, processed_image])


if __name__ == '__main__':
    main_loop()