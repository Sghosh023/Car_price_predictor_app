import pandas as pd
import numpy as np
import pickle
import sklearn
#from sklearn.preprocessing import StandardScaler
import streamlit as st

# load the random-forest model from the pickle file
rf_model = pickle.load(open('random_forest.pickle', 'rb'))
# load the StandardScaler from the pickle file
scaler = pickle.load(open('standardScaler.pickle', 'rb'))


def predict_selling_price(year, fuel_type, present_price, kms_driven, seller_type, transmission, owner):
    years = 2020 - year
    # Conversion according to fuel_type
    if fuel_type == 'Petrol':
        fuel_type_petrol = 1
        fuel_type_diesel = 0
    elif fuel_type == 'Diesel':
        fuel_type_petrol = 0
        fuel_type_diesel = 1
    else:
        fuel_type_petrol = 0
        fuel_type_diesel = 0
    # Conversion according to seller_type
    if seller_type == 'Dealer':
        seller_type = 0
    else:
        seller_type = 1
    # converting according to transmission
    if transmission == 'Manual':
        transmission = 0
    else:
        transmission = 1

    # Creating a dictionary, so that we can create a Dataframe out of it and later use it for scaling
    dict_pred = {'Present_price': present_price, 'Kms_Driven': kms_driven, 'Seller_Type': seller_type,
                 'Transmission': transmission, 'Owner': owner, 'Years': years, 'Fuel_Type_Diesel': fuel_type_diesel,
                 'Fuel_Type_Petrol': fuel_type_petrol
                 }
    df = pd.DataFrame(dict_pred, index=[0, ])
    #print(df)
    #scaler = StandardScaler()

    df_scaled = scaler.transform(df)  # applying the StandardScaler on the data
    #print(df_scaled)

    prediction = rf_model.predict(df_scaled)
    print(prediction[0])  # printing it console
    output = np.round(prediction[0], 2)  # rounding the output to 2 decimal places
    return output


def main():
    st.title("Car price predictor app")  # title of the streamlit app
    # heading of the streamlit app, the text-color will be white and background color will be blue
    html_temp = """
    <div style = "background-color:blue;padding: 10px">
    <h2 style = "color: white;text-align: center;">Streamlit car predictor ML app </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    selling_price = 0  # this is the prediction we are going to make, creating this variable & assigning 0.
    year = st.number_input('Year', 1999, 2019, step=1)
    present_price = st.number_input('Present price of the Car (in Lakhs)', 1.00, 100.00, step=0.10)
    kms_driven = st.number_input(label='Kms_driven')
    fuel_type = st.selectbox("Fuel_type", ('Petrol', 'Diesel', 'CNG'))
    seller_type = st.selectbox("Seller_type", ('Dealer', 'Individual'))
    transmission = st.selectbox("Transmission", ('Manual', 'Automatic'))
    owner = int(st.selectbox("Owner", ('0', '1', '3')))

    if st.button("Predict"):
        selling_price = predict_selling_price(year, fuel_type, present_price, kms_driven, seller_type, transmission,
                                              owner)

    if selling_price >= 0:
        st.success("The selling price of the car is {} lakhs".format(selling_price))
    #st.stop()


if __name__ == '__main__':
    main()
