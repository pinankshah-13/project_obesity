import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression # Needed for type hinting/loading

# --- Configuration ---
st.set_page_config(page_title="Obesity Level Prediction App", layout="wide")

# --- Load Preprocessing Objects and Model ---
@st.cache_resource
def load_artifacts():
    """Loads the trained model and necessary preprocessing objects."""
    try:
        # Load all saved artifacts
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('le_y.pkl', 'rb') as f:
            le_y = pickle.load(f)
        with open('feature_columns.pkl', 'rb') as f:
            feature_cols = pickle.load(f)
        return model, scaler, le_y, feature_cols
    except FileNotFoundError as e:
        st.error(f"Error: Missing essential file for deployment. Please ensure '{e.filename}' is in the directory.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred during loading artifacts: {e}")
        st.stop()

model, scaler, le_y, feature_cols = load_artifacts()

# --- Feature Mappings (based on original dataset and notebook logic) ---
# Note: These map to the LabelEncoder output (0 or 1)
BINARY_MAP = {'yes': 1, 'no': 0}
GENDER_MAP = {'Male': 1, 'Female': 0}
# Options for One-Hot Encoded features (Used for Streamlit widgets)
CAEC_OPTIONS = ['Always', 'Frequently', 'Sometimes', 'no']
CALC_OPTIONS = ['Always', 'Frequently', 'Sometimes', 'no']
MTRANS_OPTIONS = ['Automobile', 'Bike', 'Motorbike', 'Public_Transportation', 'Walking']
NUMERICAL_COLS = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']

# --- Helper Function for Prediction ---
def predict_obesity_level(data):
    """Takes raw user data, preprocesses it, and returns the prediction."""
    # 1. Create a DataFrame from user input
    input_df = pd.DataFrame([data], columns=data.keys())

    # 2. Apply Label Encoding (Binary/Ordinal)
    input_df['Gender'] = input_df['Gender'].map(GENDER_MAP)
    input_df['family_history_with_overweight'] = input_df['family_history_with_overweight'].map(BINARY_MAP)
    input_df['FAVC'] = input_df['FAVC'].map(BINARY_MAP)
    input_df['SMOKE'] = input_df['SMOKE'].map(BINARY_MAP)
    input_df['SCC'] = input_df['SCC'].map(BINARY_MAP)

    # 3. Apply One-Hot Encoding and match feature column list
    
    # Initialize a new DataFrame with all expected features set to 0
    final_df = pd.DataFrame(0, index=input_df.index, columns=feature_cols)

    # Copy the numerical/label encoded columns
    for col in input_df.columns:
        if col in final_df.columns:
            final_df[col] = input_df[col]

    # Handle One-Hot Encoding for 'CAEC', 'CALC', 'MTRANS' (drop_first=True was used)
    # The 'no' category for CAEC/CALC and 'Automobile' for MTRANS is the baseline (all 0)
    
    # CAEC
    caec_col = f'CAEC_{input_df["CAEC"].iloc[0]}'
    if caec_col in final_df.columns and input_df["CAEC"].iloc[0] != 'no':
        final_df[caec_col] = 1

    # CALC
    calc_col = f'CALC_{input_df["CALC"].iloc[0]}'
    if calc_col in final_df.columns and input_df["CALC"].iloc[0] != 'no':
        final_df[calc_col] = 1

    # MTRANS
    mtrans_col = f'MTRANS_{input_df["MTRANS"].iloc[0]}'
    if mtrans_col in final_df.columns and input_df["MTRANS"].iloc[0] != 'Automobile':
        final_df[mtrans_col] = 1
        
    # Drop original categorical columns
    final_df = final_df.drop(columns=['CAEC', 'CALC', 'MTRANS'], errors='ignore')
    
    # Ensure final columns are in the exact order the model expects
    final_df = final_df[feature_cols]

    # 4. Apply Standard Scaling to numerical features
    # .values is important to prevent pandas from aligning columns by name during transformation
    final_df[NUMERICAL_COLS] = scaler.transform(final_df[NUMERICAL_COLS].values)

    # 5. Make Prediction and Decode
    prediction = model.predict(final_df)
    obesity_level = le_y.inverse_transform(prediction)
    
    return obesity_level[0]

# --- Streamlit UI ---
st.title("Obesity Level Classification Deployment")
st.markdown("---")
st.markdown("Use this form to predict an individual's obesity level based on personal and lifestyle factors.")

# Input Form
with st.form("obesity_form"):
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Personal Metrics")
        gender = st.selectbox('Gender', options=list(GENDER_MAP.keys()))
        age = st.slider('Age', min_value=14.0, max_value=61.0, value=25.0, step=0.1)
        height = st.slider('Height (m)', min_value=1.45, max_value=2.0, value=1.70, step=0.01)
        weight = st.slider('Weight (kg)', min_value=39.0, max_value=173.0, value=80.0, step=0.1)
        
    with col2:
        st.subheader("Lifestyle Factors")
        family_history_with_overweight = st.selectbox('Family history with overweight?', options=list(BINARY_MAP.keys()))
        favc = st.selectbox('Frequent consumption of high caloric food (FAVC)', options=list(BINARY_MAP.keys()))
        smoke = st.selectbox('Smokes (SMOKE)', options=list(BINARY_MAP.keys()))
        scc = st.selectbox('Calorie Consumption monitoring (SCC)', options=list(BINARY_MAP.keys()))

    st.markdown("---")
    st.subheader("Diet and Activity Habits")
    col3, col4, col5 = st.columns(3)

    with col3:
        fcvc = st.slider('Consumption of vegetables (FCVC)', min_value=1.0, max_value=3.0, value=2.0, step=0.1)
        ncp = st.slider('Number of main meals (NCP)', min_value=1.0, max_value=4.0, value=3.0, step=0.1)
        ch2o = st.slider('Consumption of water (CH2O)', min_value=1.0, max_value=3.0, value=2.0, step=0.1)

    with col4:
        faf = st.slider('Physical activity frequency (FAF)', min_value=0.0, max_value=3.0, value=1.0, step=0.1)
        tue = st.slider('Time using technology (TUE)', min_value=0.0, max_value=2.0, value=1.0, step=0.1)
        caec = st.selectbox('Consumption of food between meals (CAEC)', options=CAEC_OPTIONS)

    with col5:
        calc = st.selectbox('Consumption of alcohol (CALC)', options=CALC_OPTIONS)
        mtrans = st.selectbox('Mode of Transportation (MTRANS)', options=MTRANS_OPTIONS)

    submitted = st.form_submit_button("Predict Obesity Level", type="primary")

# --- Prediction and Output ---
if submitted:
    
    user_data = {
        'Gender': gender,
        'Age': age,
        'Height': height,
        'Weight': weight,
        'family_history_with_overweight': family_history_with_overweight,
        'FAVC': favc,
        'FCVC': fcvc,
        'NCP': ncp,
        'CAEC': caec,
        'SMOKE': smoke,
        'CH2O': ch2o,
        'SCC': scc,
        'FAF': faf,
        'TUE': tue,
        'CALC': calc,
        'MTRANS': mtrans
    }
    
    try:
        # Calculate BMI for display
        bmi = user_data['Weight'] / (user_data['Height'] ** 2)

        with st.spinner('Running Logistic Regression Model...'):
            prediction = predict_obesity_level(user_data)
            
            st.success('Prediction Complete! 🎉')
            
            st.markdown("---")
            st.subheader("Prediction Results")
            
            col_res1, col_res2 = st.columns(2)
            
            with col_res1:
                st.metric("Body Mass Index (BMI)", f"{bmi:.2f}")

            with col_res2:
                st.metric("Predicted Obesity Level", f"**{prediction}**")
                
            st.balloons()
            
    except Exception as e:
        st.error(f"An unexpected error occurred during prediction: {e}")