import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import requests
from datetime import datetime
from streamlit_lottie import st_lottie
import matplotlib.pyplot as plt

# Page Setup
st.set_page_config(page_title="🌾 Rice Leaf Disease Detector", layout="centered")

# Load Lottie animation
def load_lottie_url(url):
    r = requests.get(url)
    return r.json() if r.status_code == 200 else None

lottie_rice = load_lottie_url("https://assets2.lottiefiles.com/packages/lf20_3vbOcw.json")

# Load models
@st.cache_resource
def load_models():
    try:
        return load_model("best_mobilenetv2_model (2).h5"), load_model("densenet_model.h5")
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

mobilenet_model, densenet_model = load_models()
class_names = ["Bacterialblight", "Blast", "Brownspot", "Tungro"]
CONFIDENCE_THRESHOLD = 0.85
GREEN_PIXEL_THRESHOLD = 10000
ENSEMBLE_WEIGHTS = {"mobilenet": 0.6, "densenet": 0.4}

# Initialize session state
if "page" not in st.session_state:
    st.session_state.page = "home"
if "language" not in st.session_state:
    st.session_state.language = "English"
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None
if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None
if "prediction_probs" not in st.session_state:
    st.session_state.prediction_probs = None

# Translation helper
def t(en, kh):
    return kh if st.session_state.language == "Khmer" else en

# Sidebar
st.sidebar.title("🌾 Rice Leaf Disease Detector")
st.session_state.language = st.sidebar.radio("🌐 Language", ["English", "Khmer"])
page = st.sidebar.selectbox(t("Navigation", "ការរុករក"), [t("Home", "ទំព័រដើម"), t("Predict", "វិភាគ"), t("About", "អំពីកម្មវិធី"), t("Help", "ជំនួយ")])
page_key = {t("Home", "ទំព័រដើម"): "home", t("Predict", "វិភាគ"): "predict", t("About", "អំពីកម្មវិធី"): "about", t("Help", "ជំនួយ"): "help"}[page]
st.session_state.page = page_key

# Home Page
if page_key == "home":
    st.markdown(f"<h1 style='text-align:center;color:#2E8B57;font-weight:bold;'>🌿 {t('Welcome to the Rice Leaf Disease Detector App!', 'សូមស្វាគមន៍មកកាន់កម្មវិធីចាប់អារម្មណ៍ជំងឺស្លឹកស្រូវ!')}</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align:center;font-size:1.2rem;color:#555;'>{t('Upload a rice leaf image or use the camera to detect diseases instantly.', 'បង្ហោះរូបភាពស្លឹកស្រូវ ឬប្រើកាមេរ៉ាដើម្បីរកជំងឺភ្លាមៗ។')}</p><hr>", unsafe_allow_html=True)
    if lottie_rice:
        st_lottie(lottie_rice, height=250)
    st.markdown(f"### 🎯 {t('Problem Statement', 'បញ្ហា')}")
    st.write(t("Diseases like Bacterial Blight, Blast, Brown Spot, and Tungro threaten rice production.", "ជំងឺដូចជា Bacterial Blight, Blast, Brown Spot និង Tungro គឺជាអាសន្នសម្រាប់ផលិតភាពស្រូវ។"))
    st.markdown(f"### 💡 {t('Purpose of This App', 'គោលបំណងរបស់កម្មវិធីនេះ')}")
    st.write(t("This app uses AI to detect rice leaf diseases from images.", "កម្មវិធីនេះប្រើបច្ចេកវិទ្យា AI ដើម្បីរកជំងឺស្លឹកស្រូវពីរូបភាព។"))

# Predict Page
elif page_key == "predict":
    if not mobilenet_model or not densenet_model:
        st.error(t("Models failed to load. Please check the model files.", "ម៉ូដែលមិនអាចផ្ទុកបាន។ សូមពិនិត្យឯកសារម៉ូដែល។"))
    else:
        st.markdown(f"### 📤 {t('Upload or Capture an Image', 'ផ្ទុកឬថតរូបភាព')}")
        model_choice = st.selectbox(t("Select Model", "ជ្រើសរើសម៉ូដែល"), ["MobileNetV2", "densNet121", "Ensemble"])
        input_mode = st.radio(t("Choose input method:", "ជ្រើសរើសវិធីបញ្ចូល:"), [t("Upload", "ផ្ទុក"), t("Camera", "កាមេរ៉ា")])
        uploaded_file = st.file_uploader(t("Upload an image", "ផ្ទុករូបភាព"), type=["jpg", "jpeg", "png"], key="file_uploader") if input_mode == t("Upload", "ផ្ទុក") else st.camera_input(t("Capture image", "ថតរូបភាព"), key="camera_input")
        if uploaded_file:
            st.session_state.uploaded_file = uploaded_file
        if st.session_state.uploaded_file:
            try:
                img = Image.open(st.session_state.uploaded_file).convert("RGB")
                st.image(img, caption=t("Uploaded Image", "រូបភាពដែលបានបញ្ចូល"), use_column_width=True)
                img_resized = img.resize((224, 224))
                img_array = image.img_to_array(img_resized) / 255.0
                img_exp = np.expand_dims(img_array, axis=0)
                green_pixels = np.sum((img_array[:, :, 1] > 0.4) & (img_array[:, :, 1] > img_array[:, :, 0]) & (img_array[:, :, 1] > img_array[:, :, 2]))
                if green_pixels < GREEN_PIXEL_THRESHOLD:
                    st.warning(t("🚫 This doesn't look like a rice leaf.", "🚫 វាមិនមែនជាស្លឹកស្រូវទេ។"))
                    st.session_state.prediction_result = st.session_state.prediction_probs = None
                else:
                    with st.spinner(t("🧠 Analyzing...", "🧠 កំពុងវិភាគ...")):
                        preds = mobilenet_model.predict(img_exp)[0] if model_choice == "MobileNetV2" else densenet_model.predict(img_exp)[0] if model_choice == "densenet_model" else (ENSEMBLE_WEIGHTS["mobilenet"] * mobilenet_model.predict(img_exp)[0] + ENSEMBLE_WEIGHTS["densenet"] * densenet_model.predict(img_exp)[0])
                        max_index = np.argmax(preds)
                        max_conf = preds[max_index]
                        if max_conf < CONFIDENCE_THRESHOLD:
                            st.error(t("🤔 Confidence too low. Try another image.", "🤔 ការពិចារណាទាបពេក។ សូមសាកល្បងរូបភាពផ្សេង។"))
                            st.session_state.prediction_result = st.session_state.prediction_probs = None
                        else:
                            disease = class_names[max_index]
                            st.session_state.prediction_result = {"datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "disease": disease, "confidence": round(float(max_conf), 4)}
                            st.session_state.prediction_probs = preds
                    if st.session_state.prediction_result:
                        res = st.session_state.prediction_result
                        st.success(f"✅ {t('Disease', 'ជំងឺ')} : {res['disease']}")
                        st.info(f"🔬 {t('Confidence', 'ភាពជឿជាក់')} : {res['confidence']*100:.2f}%")
                        if st.session_state.prediction_probs is not None:
                            st.markdown(f"### 📊 {t('Prediction Probabilities', 'ប្រូបាប៊ីលីតេនៃការព្យាករណ៍')}")
                            fig, ax = plt.subplots()
                            ax.bar(class_names, st.session_state.prediction_probs, color='green', alpha=0.7)
                            ax.set_title(t("Prediction Probabilities for Each Disease", "ប្រូបាប៊ីលីតេនៃការព្យាករណ៍សម្រាប់ជំងឺនីមួយៗ"))
                            ax.set_xlabel(t("Disease", "ជំងឺ"))
                            ax.set_ylabel(t("Probability", "ប្រូបាប៊ីលីតេ"))
                            ax.set_ylim(0, 1)
                            plt.xticks(rotation=45, ha='right')
                            plt.tight_layout()
                            st.pyplot(fig)
                col1, _ = st.columns(2)
                with col1:
                    if st.button(t("Clear Photo & Results", "លុបរូបភាព និងលទ្ធផល"), key="clear_button"):
                        st.session_state.uploaded_file = st.session_state.prediction_result = st.session_state.prediction_probs = None
                        st.success(t("Photo and results cleared.", "រូបភាព និងលទ្ធផលត្រូវបានលុប។"))
            except Exception as e:
                st.error(f"❌ {t('Error processing image:', 'កំហុសក្នុងការបំលែងរូបភាព')} {e}")
        else:
            st.info(t("👈 Please upload or capture an image.", "👈 សូមផ្ទុកឬថតរូបភាព។"))

# About Page
elif page_key == "about":
    st.markdown(f"# {t('About This App', 'អំពីកម្មវិធីនេះ')}")
    st.write(t(
        """
        The Rice Leaf Disease Detector app is designed to help farmers and researchers
        quickly identify diseases in rice plants using AI and image recognition.
        
        This app uses MobileNetV2, ResNet50, or an ensemble of both deep learning models
        trained on rice leaf images to improve prediction accuracy. It supports multiple
        diseases including Bacterial Blight, Blast, Brown Spot, and Tungro.
        
        Developed by a passionate team to support sustainable agriculture.
        """,
        """
        កម្មវិធីចាប់អារម្មណ៍ជំងឺស្លឹកស្រូវត្រូវបានរចនាឡើងដើម្បីជួយកសិករ និងអ្នកស្រាវជ្រាវ
        ក្នុងការស្គាល់ជំងឺនៅលើរុក្ខជាតិស្រូវយ៉ាងលឿនដោយប្រើបច្ចេកវិទ្យា AI និងការទទួលស្គាល់រូបភាព។
        
        កម្មវិធីនេះប្រើ MobileNetV2, ResNet50 ឬការរួមបញ្ចូលគ្នានៃម៉ូដែលទាំងពីរ
        ដែលបានបណ្តុះបណ្តាលលើរូបភាពស្លឹកស្រូវដើម្បីបង្កើនភាពត្រឹមត្រូវនៃការព្យាករណ៍។
        វាគាំទ្រជំងឺចម្បងៗដូចជា Bacterial Blight, Blast, Brown Spot និង Tungro។
        
        បានបង្កើតឡើងដោយក្រុមអ្នកបង្កើតដែលមានចំណង់ចំណូលចិត្តដើម្បីគាំទ្រឧស្សាហកម្មកសិកម្មចីរភាព។
        """
    ))
    st.markdown(f"### 📈 {t('Sample Confidence Trend', 'លំនាំភាពជឿជាក់')}")
    times = ["2025-05-20 10:00", "2025-05-21 12:00", "2025-05-22 14:00", "2025-05-23 16:00"]
    confidences = [0.7, 0.85, 0.9, 0.95]
    fig, ax = plt.subplots()
    ax.plot(times, confidences, marker='o', linestyle='-', color='green')
    ax.set_title(t("Confidence Over Time", "ភាពជឿជាក់តាមពេលវេលា"))
    ax.set_xlabel(t("Time", "ពេលវេលា"))
    ax.set_ylabel(t("Confidence", "ភាពជឿជាក់"))
    ax.set_ylim(0, 1)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig)

# Help Page
elif page_key == "help":
    st.markdown(f"# {t('Help & Instructions', 'ជំនួយ និងការណែនាំ')}")
    st.write(t(
        """
        Use the Predict page to upload or capture an image of a rice leaf.
        The app will analyze and predict if the leaf is infected by one of the diseases.
        Clear the image and results to upload a new one.
        Use the About page to learn more about the app.
        For further help, contact the development team.
        """,
        """
        ប្រើទំព័រវិភាគដើម្បីផ្ទុកឬថតរូបភាពស្លឹកស្រូវ។
        កម្មវិធីនឹងវិភាគ និងព្យាករណ៍ថាតើស្លឹកមានជំងឺណាមួយទេ។
        លុបរូបភាព និងលទ្ធផលដើម្បីផ្ទុករូបភាពថ្មី។
        ប្រើទំព័រអំពីកម្មវិធីដើម្បីស្វែងយល់បន្ថែម។
        សម្រាប់ជំនួយបន្ថែម សូមទាក់ទងក្រុមបង្កើតកម្មវិធី។
        """
    ))
    st.markdown(f"### 📞 {t('Contact Us', 'ទាក់ទងមកយើងខ្ញុំ')}")
    st.write(t("For any issues or suggestions, please reach out to us at:", "សម្រាប់បញ្ហាឬយោបល់ណាមួយ សូមទាក់ទងមកយើងខ្ញុំតាមអាសយដ្ឋាន៖"))