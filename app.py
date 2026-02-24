import streamlit as st
from streamlit_drawable_canvas import st_canvas
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import pandas as pd

# 1. Page Setting (Wider layout ke liye)
st.set_page_config(page_title="AI Digit Recognizer", page_icon="✨", layout="wide")

# 2. Custom HTML/CSS Styling
st.markdown("""
    <style>
    .main-title { font-size: 45px; color: #FF4B4B; font-weight: bold; text-align: center; margin-bottom: 0px;}
    .sub-title { font-size: 18px; color: #B0B0B0; text-align: center; margin-bottom: 30px;}
    .result-text { font-size: 25px; color: #4CAF50; font-weight: bold;}
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">Identify Digits? ✨</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">  Draw any digit using your mouse and let the AI predict it!    </p>', unsafe_allow_html=True)

# 3. Naya CNN Model aur Loading (Same purana wala)
class MeraModel(nn.Module):
    def __init__(self):
        super(MeraModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16 * 14 * 14, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = MeraModel()
model.load_state_dict(torch.load('mera_digit_model.pth'))
model.eval()

# 4. Page ko 2 Columns mein batna
col1, col2 = st.columns(2)

with col1:
    st.markdown("### ✍️  Draw your number here -- 0 to 9 ")
    canvas_result = st_canvas(
        fill_color="black",
        stroke_width=20,
        stroke_color="white",
        background_color="black",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )
    
    predict_btn = st.button('🚀  See Result', use_container_width=True)

with col2:
    st.markdown("### 🤖 Brain of AI")
    
    if predict_btn and canvas_result.image_data is not None:
        # Image Preprocessing (Wahi border add karne wali trick)
        img = cv2.cvtColor(canvas_result.image_data, cv2.COLOR_RGBA2GRAY)
        img = cv2.copyMakeBorder(img, 40, 40, 40, 40, cv2.BORDER_CONSTANT, value=0)
        img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
        
        img_tensor = torch.tensor(img, dtype=torch.float32) / 255.0
        img_tensor = img_tensor.unsqueeze(0).unsqueeze(0) 
        
        # Model Prediction aur Probabilities nikalna
        with torch.no_grad():
            output = model(img_tensor)
            # Softmax use karke raw score ko percentage (0-100) mein badalna
            probabilities = F.softmax(output[0], dim=0).numpy() * 100
            _, predicted = torch.max(output.data, 1)
            
        st.markdown(f'<p class="result-text">Answer is : {predicted.item()} 🎉</p>', unsafe_allow_html=True)
        
        # Bar Chart banana
        st.write("**Model Confidence (Percentage %):**")
        chart_data = pd.DataFrame(
            probabilities,
            index=[str(i) for i in range(10)],
            columns=['Confidence']
        )
        st.bar_chart(chart_data)
    else:
        st.info("Bahar draw karne ke baad 'Result Btao!' button par click karo.")