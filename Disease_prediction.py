import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Set Background Image (Ambulance Theme)
def set_bg_image():
    st.markdown(
        """
        <style>
        .stApp {
            background-image: url("https://images.unsplash.com/photo-1588776814546-7b1cf79f2b68?ixlib=rb-4.0.3&auto=format&fit=crop&w=1470&q=80");
            background-size: cover;
            background-position: center;
        }
        .stApp::before {
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(255, 255, 255, 0.6); /* overlay for readability */
            z-index: -1;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


# App Title
st.write("## 👨‍⚕️ Doctor Information")

col1, col2 = st.columns([1,3])

with col1:
    st.image("https://cdn-icons-png.flaticon.com/512/387/387561.png", width=120)  # doctor icon

with col2:
    st.markdown("""
    **Dr. John Doe**  
    *General Physician*  
    📧 johndoe@example.com  
    📞 +1-555-123-4567  
    🏥 HealthCare Saraswati, New York, Allahabad
    """)

# Doctor Information Section
# Doctor Information Section
st.sidebar.title("👨‍⚕️ Doctor Information")

# Multiple doctors with availability
doctors = [
    {
        "name": "Dr. John Doe",
        "specialization": "General Physician",
        "email": "johndoe@example.com",
        "phone": "+1-555-123-4567",
        "clinic": "HealthCare Saraswati, New York, Allahabad",
        "availability": "Monday - Saturday, 9 AM - 5 PM"
    },
    {
        "name": "Dr. Priya Sharma",
        "specialization": "Cardiologist",
        "email": "priyasharma@example.com",
        "phone": "+91-98765-43210",
        "clinic": "HeartCare Center, Allahabad",
        "availability": "Sunday - Friday, 10 AM - 4 PM"
    },
    {
        "name": "Dr. Rajesh Kumar",
        "specialization": "Pediatrician",
        "email": "rajeshkumar@example.com",
        "phone": "+91-91234-56789",
        "clinic": "ChildCare Hospital, Allahabad",
        "availability": "Monday - Saturday, 11 AM - 6 PM"
    }
]

# Display doctors in sidebar
for doc in doctors:
    st.sidebar.markdown(f"""
    **{doc['name']}**  
    *{doc['specialization']}*  
    📧 {doc['email']}  
    📞 {doc['phone']}  
    🏥 {doc['clinic']}  
    ⏰ Availability: {doc['availability']}
    ---
    """)




# Load Dataset
@st.cache_data
def load_data():
    df = pd.read_csv("disease_prediction.csv")
    return df

df = load_data()

st.write("### Dataset Preview")
st.dataframe(df.head())
st.write("Shape of dataset:", df.shape)

# Features & Target
X = df.drop("Disease", axis=1)
y = df["Disease"]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

st.write("## Enter Symptoms")

# Create input fields dynamically based on dataset columns
user_input = []

for column in X.columns:
    value = st.selectbox(f"{column}", [0, 1])
    user_input.append(value)

# Prediction
if st.button("Predict Disease"):
    input_data = np.array([user_input])
    prediction = model.predict(input_data)
    st.success(f"Predicted Disease: {prediction[0]}")
