#importing the libraries
import streamlit as st
from PIL import Image
import requests
import os

# Load the CSS file
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("Frontend/styles.css")

# Initialize session state for controlling visibility
if "verification_result" not in st.session_state:
    st.session_state.verification_result = None  # Initially, no result is available

## SECTION- 1
# Title of the webpage
st.header('Document Verification System')

# Create a container for the file uploader and image preview
with st.container():
    # File uploader
    uploaded_file = st.file_uploader("Choose a file", type=["jpg", "jpeg", "png" , "tif"])
    
    # Display the uploaded image
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', width=300)

        uploaded_file.seek(0)  # Reset file pointer to the start
        img_bytes = uploaded_file.read()
        uploaded_file.seek(0)
        
        # Define the API endpoint
        api_endpoint = "http://127.0.0.1:8000/verify-document"
        
        # Function to send the file to the backend
        def upload_file():
            response = requests.post(api_endpoint, files={"file": (uploaded_file.name, img_bytes, uploaded_file.type)})
            if response.status_code == 200:
                st.success("File successfully verified!")
                st.session_state.verification_result = response.json()  # Save the result to session state
            else:
                st.error("File verification failed.")
                st.session_state.verification_result = None
        
        # Upload Button
        if st.button('Verify', key='verify-button'):
            upload_file()

## SECTION 2
# Display Section 2 only if verification_result is available
if st.session_state.verification_result:
    data = st.session_state.verification_result  # Retrieve the stored result

    # Create two columns for better alignment
    col1, col2 = st.columns(2)

    # Display ACCOUNT DETAILS IN THE CONTAINER
    with col1:
        st.markdown('<div class="info-block">', unsafe_allow_html=True)
        st.header("Account Details")
        st.write(data.get('account_details', 'N/A'))  # Display account details
        st.markdown('</div>', unsafe_allow_html=True)

    # Display SIGNATURE IN THIS BLOCK
    with col2:
        st.markdown('<div class="signature-block">', unsafe_allow_html=True)
        st.header("Extracted Signature")
        if 'signature' in data:  # Check if signature is available
            try:
                # Read the image from the path
                signature_image = Image.open(data['signature'])
                st.image(signature_image, caption='Signature')
            except Exception as e:
                st.write(f"Error loading signature: {e}")
        else:
            st.write("No signature available.")
        st.markdown('</div>', unsafe_allow_html=True)

    # Display verification details and distance in the results section
    with st.container():
        st.markdown('<div class="results-block">', unsafe_allow_html=True)
        st.subheader('Results')
        if data.get('Verification details', 'Not Available') == 'Matched':
            st.markdown(f'<span class="matched">Verification Details: {data["Verification details"]}</span>', unsafe_allow_html=True)
        else:
            st.markdown(f'<span class="not-matched">Verification Details: {data["Verification details"]}</span>', unsafe_allow_html=True)
        st.markdown(f'<span class="distance">Distance: {data.get("Distance", "N/A")}</span>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
