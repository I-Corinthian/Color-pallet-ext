import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms as T
from torchvision import models
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


model = models.detection.maskrcnn_resnet50_fpn_v2(weights='DEFAULT').eval()
transform = T.ToTensor()
kmeans = KMeans(n_clusters=5, random_state=0)


def draw_polyline(input_img):
    img_tensor = transform(input_img)

    with torch.no_grad():
        output = model([img_tensor])

    mask = output[0]['masks']
    mask = mask[0,0] > 0.5

    img_mask = mask.numpy().astype('uint8') * 255 
    output_img = cv2.bitwise_and(input_img,input_img,mask=img_mask)

    contours, _ = cv2.findContours(img_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.polylines(input_img, contours, isClosed=True, color=(0, 255, 0), thickness=2)

    st.image(input_img,caption="contour img")

    return output_img

def ext_colorpallet(output_img):
    reshaped_image = output_img.reshape(-1, 3)
    reshaped_image = reshaped_image[~np.all(reshaped_image == 0, axis=1)]

    kmeans.fit(reshaped_image)
    colors = kmeans.cluster_centers_.astype(int)

    return colors




st.title("Color Pallet Extractor")

img_path = st.file_uploader("choose a image",type=['png','jpg','jpeg'])

if img_path is not None:
    img = np.array(Image.open(img_path))
    st.image(img, caption='Image')
    output_img = draw_polyline(img)
    colors = ext_colorpallet(output_img)

    palette = np.zeros((50, 300, 3), dtype=np.uint8)
    step = 300 // len(colors)
    for i, color in enumerate(colors):
        palette[:, i * step:(i + 1) * step] = color

    fig, ax = plt.subplots(figsize=(8, 2))
    ax.imshow(palette, aspect="auto")
    ax.axis('off')  

    st.pyplot(fig)


