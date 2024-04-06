import pandas as pd
import streamlit as st
import pickle


@st.cache(allow_output_mutation=True)
def load_models():
    with open('flat_resales_model', 'rb') as file:
        loaded_model = pickle.load(file)
    with open('binary_encoder', 'rb') as file:
        binary_encoder = pickle.load(file)
    return loaded_model, binary_encoder


loaded_model, binary_encoder = load_models()

# Define unique values for select boxes
flat_model_options = ['IMPROVED', 'NEW GENERATION', 'MODEL A', 'STANDARD', 'SIMPLIFIED',
                      'MODEL A-MAISONETTE', 'APARTMENT', 'MAISONETTE', 'TERRACE', '2-ROOM',
                      'IMPROVED-MAISONETTE', 'MULTI GENERATION', 'PREMIUM APARTMENT',
                      'ADJOINED FLAT', 'PREMIUM MAISONETTE', 'MODEL A2', 'DBSS', 'TYPE S1',
                      'TYPE S2', 'PREMIUM APARTMENT LOFT', '3GEN']
flat_type_options = ['1 ROOM', '2 ROOM', '3 ROOM', '4 ROOM', '5 ROOM', 'EXECUTIVE', 'MULTI GENERATION']
town_options = ['ANG MO KIO', 'BEDOK', 'BISHAN', 'BUKIT BATOK', 'BUKIT MERAH', 'BUKIT TIMAH',
                'CENTRAL AREA', 'CHOA CHU KANG', 'CLEMENTI', 'GEYLANG', 'HOUGANG',
                'JURONG EAST', 'JURONG WEST', 'KALLANG/WHAMPOA', 'MARINE PARADE',
                'QUEENSTOWN', 'SENGKANG', 'SERANGOON', 'TAMPINES', 'TOA PAYOH', 'WOODLANDS',
                'YISHUN', 'LIM CHU KANG', 'SEMBAWANG', 'BUKIT PANJANG', 'PASIR RIS', 'PUNGGOL']
storey_range_options = ['10 TO 12', '04 TO 06', '07 TO 09', '01 TO 03', '13 TO 15', '19 TO 21',
                        '16 TO 18', '25 TO 27', '22 TO 24', '28 TO 30', '31 TO 33', '40 TO 42',
                        '37 TO 39', '34 TO 36', '46 TO 48', '43 TO 45', '49 TO 51']


# Load the saved model


# Streamlit app title
st.title("Resale Price Prediction App")

# Create a Streamlit sidebar with input fields
st.sidebar.title("Flat Details")
town = st.sidebar.selectbox("Town", options=town_options)
flat_type = st.sidebar.selectbox("Flat Type", options=flat_type_options)
ordinal_mapping = {'1 ROOM': 1,
                   '2 ROOM': 2,
                   '3 ROOM': 3,
                   '4 ROOM': 4,
                   '5 ROOM': 5,
                   'EXECUTIVE': 6,
                   'MULTI GENERATION': 7,
                   }
flat_type = ordinal_mapping.get(flat_type)
floor_area_sqm = st.sidebar.number_input("Floor Area (sqm)", min_value=0.0, max_value=500.0, value=100.0)
flat_model = st.sidebar.selectbox("Flat Model", options=flat_model_options)
storey_range = st.sidebar.selectbox("Storey Range", options=storey_range_options)
storey_range = storey_range.split("TO")[0].strip()
Resale_date = st.sidebar.text_input("Type Resale date in YYYY-MM", "2023-04")
lease_commence_year = st.sidebar.text_input("Type lease_commence_year in YYYY", "1990")

# Create a button to trigger the prediction
if st.sidebar.button("Predict Resale Price"):
    # Prepare input data for prediction
    df = pd.DataFrame({
        'town': [town],
        'flat_type': [flat_type],
        'floor_area_sqm': [floor_area_sqm],
        'flat_model': [flat_model],
        'storey_range_lower_limit': [storey_range],
        'Resale_date': [Resale_date],
        "lease_commence_year": [lease_commence_year],
    })
    df["Resale_date"] = pd.to_datetime(df["Resale_date"], format='%Y-%m')
    df["lease_commence_year"] = pd.to_datetime(df["lease_commence_year"], format='%Y')
    f = df['Resale_date'] - df['lease_commence_year']
    df["year_old"] = f / pd.Timedelta(days=365.25)
    df["resale_year"] = Resale_date.split("-")[0].strip()
    df["resale_month"] = Resale_date.split("-")[1].strip()
    df.drop(columns=["Resale_date", "lease_commence_year"], inplace=True)
    df = binary_encoder.transform(df)

    # Make a prediction using the model
    prediction = loaded_model.predict(df)

    # Display the prediction
    st.write("Predicted Resale Price:", prediction)
