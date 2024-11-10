import streamlit as st
import tensorflow as tf
import numpy as np
import time as t

#Tensorflow model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model('trained_plant_disease_model.keras')
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to a batch.
    predictions = model.predict(input_arr)
    result_index = np.argmax(predictions)  # Return index of max element
    return result_index

# Initialize session state for page tracking
if 'page' not in st.session_state:
    st.session_state['page'] = 'Home'

# Sidebar with individual buttons
st.sidebar.title("Dashboard")

# Set session state based on button clicks
if st.sidebar.button('Home'):
    st.session_state['page'] = 'Home'
if st.sidebar.button('About'):
    st.session_state['page'] = 'About'
if st.sidebar.button('Disease Recognition'):
    st.session_state['page'] = 'Disease Recognition'

# Home Page
if st.session_state['page'] == 'Home':
    st.header("Plant Disease Recognition System")
    image_path = "home.jpg"
    st.image(image_path, use_column_width=True)
    st.markdown("""
    Welcome to the Plant Disease Recognition System! 🌿🔍
    
    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!

    ### About Us
    Learn more about the project, our team, and our goals on the **About** page.
    """)

# About Page
elif st.session_state['page'] == 'About':
    st.header("About")
    st.markdown("""
    #### About Dataset
    This dataset is recreated using offline augmentation from the original dataset. The original dataset can be found on this GitHub repo.
    This dataset consists of about 87K RGB images of healthy and diseased crop leaves, categorized into 38 different classes. The total dataset is divided into an 80/20 ratio of training and validation sets, preserving the directory structure. A new directory containing 33 test images is created later for prediction purposes.
    
    #### Content
    1. train (70,295 images)
    2. test (33 images)
    3. validation (17,572 images)
    """)

# Disease Recognition Page
elif st.session_state['page'] == 'Disease Recognition':
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an Image:")

    if test_image is not None:
        st.image(test_image, use_column_width=True)
    
    # Predict Button
    if st.button("Predict") and test_image is not None:
        with st.spinner("Predicting......"):
            t.sleep(1.5)
            st.write("Our Prediction")
            result_index = model_prediction(test_image)  # Replace this with your model's prediction function
            
            # Define class names
            class_names = [
                'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
                'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
                'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
                'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
                'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
                'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
                'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
                'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
                'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
                'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
                'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
            ]
            # Predefined knowledge base for disease details and solutions
            disease_info = {
                'Apple___Apple_scab': {
                    'details': "Apple scab is a fungal disease caused by Venturia inaequalis, affecting apple leaves, fruit, and shoots. It appears as dark, velvety lesions on the leaves and fruit, leading to premature leaf drop and reduced fruit quality. The disease thrives in cool, wet weather, particularly in spring. If left untreated, apple scab can weaken trees and make them more vulnerable to other diseases and pests, significantly reducing yields.",
                    'solution': [
                        "Apply fungicide sprays starting early in the growing season.",
                        "Prune trees to increase airflow and reduce moisture retention.",
                        "Remove and destroy fallen leaves and infected fruit.",
                        "Plant resistant apple varieties to minimize susceptibility.",
                        "Practice crop rotation and avoid planting apples in the same soil year after year."
                    ]
                },
                'Apple___Black_rot': {
                    'details': "Black rot in apples is caused by the fungus Botryosphaeria obtusa, which affects both fruit and tree structure. The disease is most common in warm, humid climates and causes dark, sunken lesions on fruit, bark cankers, and leaf spots. Infected fruit often shrivels, and the disease can remain dormant in branches, reappearing in later seasons if not treated.",
                    'solution': [
                        "Prune and remove infected branches to prevent further spread.",
                        "Apply appropriate fungicides during the growing season.",
                        "Maintain orchard sanitation by removing fallen fruit and debris.",
                        "Improve airflow by thinning dense tree canopies.",
                        "Ensure trees are healthy through proper watering and fertilization."
                    ]
                },
                'Apple___Cedar_apple_rust': {
                    'details': "Cedar apple rust is a disease caused by the fungus Gymnosporangium juniperi-virginianae. The fungus requires both apple and juniper (cedar) trees to complete its life cycle. On apple trees, the disease manifests as bright orange-yellow spots on leaves and fruit. Infected leaves may drop prematurely, and severe infections can reduce the tree's vigor. The disease thrives in moist conditions and spreads through windborne spores.",
                    'solution': [
                        "Remove nearby juniper trees if feasible to break the fungus life cycle.",
                        "Apply fungicides to apple trees during the early stages of growth.",
                        "Monitor for early signs of infection and take preventive action.",
                        "Prune infected branches and ensure proper orchard ventilation."
                    ]
                },
                'Apple___healthy': {
                    'details': "The apple tree is healthy, showing no signs of disease or pest infestations. Healthy apple trees typically exhibit vibrant green leaves, clean and blemish-free fruit, and strong, undamaged branches. Regular monitoring and proper care are essential to maintain tree health and prevent future disease outbreaks.",
                    'solution': [
                        "Continue regular watering and fertilization to maintain tree health.",
                        "Monitor for pests or diseases to catch early signs of infection.",
                        "Prune the tree annually to promote air circulation and sunlight exposure.",
                        "Mulch around the tree base to retain moisture and prevent weeds."
                    ]
                },
                'Blueberry___healthy': {
                    'details': "This blueberry plant is in optimal health, with strong stems, vibrant green leaves, and well-formed berries. Healthy plants are resistant to most common diseases and pests, especially when proper growing conditions and care practices are followed. Good soil, watering, and pruning habits will help maintain plant vitality.",
                    'solution': [
                        "Maintain consistent soil moisture and pH balance (4.5-5.5).",
                        "Regularly prune to encourage air circulation.",
                        "Watch for early signs of diseases or pests.",
                        "Use organic mulch to retain soil moisture and regulate temperature."
                    ]
                },
                'Cherry_(including_sour)___Powdery_mildew': {
                    'details': "Powdery mildew on cherry trees is caused by the fungus Podosphaera clandestina. It appears as a white powdery substance on leaves, stems, and fruit. The disease thrives in warm, dry conditions, but it requires humidity to infect the plant. Left untreated, powdery mildew can stunt growth, reduce fruit yield, and cause premature leaf drop.",
                    'solution': [
                        "Apply sulfur-based fungicides at the first sign of mildew.",
                        "Prune to improve air circulation and sunlight penetration.",
                        "Remove and destroy infected leaves and fruit.",
                        "Water plants at the base to avoid wetting leaves.",
                        "Plant resistant varieties if available."
                    ]
                },
                'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': {
                    'details': "Gray leaf spot is a common fungal disease in maize caused by Cercospora zeae-maydis. The disease appears as grayish or tan lesions on the leaves, leading to tissue death and reduced photosynthesis. Severe infections can reduce yield, especially in regions with high humidity and warm temperatures. It can overwinter in crop residue, making management crucial.",
                    'solution': [
                        "Rotate crops to reduce disease buildup in the soil.",
                        "Use resistant maize varieties to minimize infection.",
                        "Apply fungicides during critical growth stages.",
                        "Plow under infected crop residue after harvest.",
                        "Monitor and scout fields regularly for early signs of the disease."
                    ]
                },
                'Corn_(maize)___Common_rust_': {
                    'details': "Common rust of corn is caused by the fungus Puccinia sorghi, which produces reddish-brown pustules on both sides of the leaves. Infected plants may show reduced vigor, and in severe cases, rust can reduce photosynthetic capacity, leading to lower yields. The disease is more common in cooler, humid climates and can spread rapidly under favorable conditions.",
                    'solution': [
                        "Use rust-resistant corn hybrids.",
                        "Apply fungicides if the infection becomes severe.",
                        "Scout fields early to detect the onset of rust.",
                        "Rotate crops to prevent the buildup of the rust pathogen.",
                        "Maintain optimal plant spacing to improve air circulation."
                    ]
                },
                'Tomato___Tomato_mosaic_virus': {
                    'details': "Tomato mosaic virus (ToMV) is a highly contagious virus that causes mottling and mosaic patterns on tomato leaves, leading to reduced growth and distorted fruit. The virus spreads through contact with infected plants, tools, or soil and can persist in plant debris for long periods. Infected plants show reduced yields and poor fruit quality, making early detection and management crucial.",
                    'solution': [
                        "Use virus-free seeds and seedlings.",
                        "Disinfect tools and equipment regularly to prevent the spread.",
                        "Remove and destroy infected plants immediately.",
                        "Practice crop rotation with non-host plants.",
                        "Avoid handling plants when they are wet to minimize virus spread."
                    ]
                },
                'Tomato___healthy': {
                    'details': "The tomato plant is in excellent health, with no visible signs of disease or pest infestations. Healthy tomato plants have strong, green leaves and produce robust, well-formed fruit. Regular care, such as proper watering, pruning, and monitoring, helps maintain this healthy state and prevent potential issues.",
                    'solution': [
                        "Continue regular watering and fertilization for optimal growth.",
                        "Monitor for pests or early signs of disease.",
                        "Prune the plant to promote airflow and sunlight penetration.",
                        "Support plants with stakes or cages to prevent fruit from touching the ground."
                    ]
                }
            }
            predicted_class = class_names[result_index]
            st.success(f"Model is predicting: {predicted_class}")

            # Assume predicted_class is already set and exists in the dictionary
            if predicted_class in disease_info:
                st.write(f"**Details about {predicted_class}:**")
                st.write(disease_info[predicted_class]['details'])

                st.write("### Solution:")
                for solution in disease_info[predicted_class]['solution']:
                    st.write(f"- {solution}")
            else:
                st.write(f"Sorry, no information available for {predicted_class}.")


