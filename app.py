import streamlit as st
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

st.set_page_config(page_title="Gen AI Image Captioning", page_icon="🖼️")

@st.cache_resource
def load_model():
    name = "Salesforce/blip-image-captioning-base"
    processor = BlipProcessor.from_pretrained(name)
    model = BlipForConditionalGeneration.from_pretrained(
        name, low_cpu_mem_usage=True
    )
    model.eval()
    return processor, model

st.title("🖼️ Gen AI Powered Image Captioning")
st.write("Upload an image and let AI generate a meaningful caption.")

uploaded = st.file_uploader("Upload Your Image", type=["jpg", "jpeg", "png"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, use_container_width=True)

    if st.button("Generate Caption"):
        with st.spinner("Loading model & generating caption..."):
            processor, model = load_model()
            inputs = processor(image, return_tensors="pt")
            with torch.no_grad():
                output = model.generate(**inputs, max_new_tokens=50)
            caption = processor.decode(output[0], skip_special_tokens=True)
        st.success(caption)
