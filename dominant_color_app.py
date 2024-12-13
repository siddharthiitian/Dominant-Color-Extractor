%pip install -r requirements.txt

import streamlit as st
import cv2
import numpy as np
from sklearn.cluster import KMeans
from io import BytesIO
from PIL import Image

# Function to calculate dominant colors
def get_dominant_colors(image, clusters=3):
    # Convert the image to RGB and resize for efficiency
    image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (100, 100))  # Reduce size for faster processing
    
    # Flatten the image
    image_flatten = image.reshape(-1, 3)
    
    # Apply k-means clustering
    kmeans = KMeans(n_clusters=clusters, random_state=42)
    kmeans.fit(image_flatten)
    
    # Get cluster centers as dominant colors
    dominant_colors = kmeans.cluster_centers_.astype(int)
    return dominant_colors

# Streamlit app
def main():
    st.title("Dominant Color Extractor")
    st.write("Upload an image to find its dominant colors!")

    # File uploader
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(BytesIO(uploaded_file.read()))
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Button to process the image
        if st.button("Go"):
            with st.spinner("Processing..."):
                # Get dominant colors
                dominant_colors = get_dominant_colors(image)

                # Display the dominant colors
                st.write("### Dominant Colors:")
                for idx, color in enumerate(dominant_colors):
                    st.write(f"Color {idx + 1}: RGB {tuple(color)}")
                    st.markdown(
                        f'<div style="width:100px; height:50px; background-color:rgb({color[0]},{color[1]},{color[2]});"></div>',
                        unsafe_allow_html=True,
                    )

if __name__ == "__main__":
    main()
