import streamlit as st
from streamlit_drawable_canvas import st_canvas
import torch
import torch.nn as nn
import cv2
import numpy as np

# 1. Wahi same Model Class jo humne Jupyter mein banayi thi
class MeraModel(nn.Module):
    def __init__(self):
        super(MeraModel, self).__init__()
        self.flatten = nn.Flatten() 
        self.hidden_layer = nn.Linear(28 * 28, 128) 
        self.relu = nn.ReLU() 
        self.output_layer = nn.Linear(128, 10) 

    def forward(self, x):
        x = self.flatten(x)
        x = self.hidden_layer(x)
        x = self.relu(x)
        x = self.output_layer(x)
        return x

# 2. Model ko load karna aur Test mode mein dalna
model = MeraModel()
model.load_state_dict(torch.load('mera_digit_model.pth'))
model.eval()

# 3. Dashboard ka Design (Streamlit)
st.title("Pehchano Kaun? 🔢")
st.write("Niche box mein mouse se koi number (0-9) draw karo aur dekho AI kaise pehchanta hai!")

# Ek drawing canvas banana (Black background, white pen - kyunki hamara model aise hi seekha hai)
canvas_result = st_canvas(
    fill_color="black",
    stroke_width=25,
    stroke_color="white",
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

# 4. Predict Button
if st.button('Number Pehchano!'):
    if canvas_result.image_data is not None:
        # Photo ko 28x28 size mein chota karna aur grayscale karna
        img = cv2.cvtColor(canvas_result.image_data, cv2.COLOR_RGBA2GRAY)
        img = cv2.resize(img, (28, 28))
        
        # Image ko PyTorch Tensor mein badalna
        img_tensor = torch.tensor(img, dtype=torch.float32) / 255.0
        img_tensor = img_tensor.unsqueeze(0).unsqueeze(0) 
        
        # Model se prediction lena
        with torch.no_grad():
            output = model(img_tensor)
            _, predicted = torch.max(output.data, 1)
            
        st.success(f"Mujhe lagta hai ye number hai: **{predicted.item()}**")