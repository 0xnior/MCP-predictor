import streamlit as st
import pickle
import pandas as pd

st.markdown("<h1 style='text-align: center; font-size: 40px;'>MCP Predictor (Rs/MWh)</h1>", unsafe_allow_html=True)


# Load the model and scaler
pipe = pickle.load(open("IEX_model.pickle", 'rb'))
scaler = pickle.load(open("scaler.pickle", 'rb'))

# Define the form
with st.form("my_form"):
    # Input fields
    d = {
        'Purchase': [st.number_input("Purchase")],
        'sbTotal': [st.number_input("sbTotal")],
        'sbSolar': [st.number_input("sbSolar")],
        'sbNonSolar': [st.number_input("sbNonSolar")],
        'sbHydro': [st.number_input("sbHydro")],
        'mcvTotal': [st.number_input("mcvTotal")],
        'mcvNonsolar': [st.number_input("mcvNonsolar")],
        'mcvHydro': [st.number_input("mcvHydro")],
        'fsvTotal': [st.number_input("fsvTotal")]
    }

    # Submit button inside the form
    submitted = st.form_submit_button(label='Predict')
    # Perform prediction after form submission
    if submitted:
        # Convert the dictionary to a DataFrame
        df = pd.DataFrame(d)

        # Scale the input data and make a prediction
        prediction = pipe.predict(scaler.transform(df))

        # Display the result
        st.markdown(f"<h1 style='text-align: center; font-size: 40px;'>Prediction: {prediction[0]}</h1>",
                    unsafe_allow_html=True)
