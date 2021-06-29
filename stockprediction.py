import streamlit as st
from datetime import date

import yfinance as yf
from prophet import Prophet 
import plotly.graph_objs as go
from prophet.plot import plot_plotly
#from fbprophet import graph_objs as go

#daily_seasonality=True

START = "2010-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title("Stock Prediction App")

form = st.form(key='my-form')
name = form.text_input('Geben Sie ihren gewünschten Aktienkürzel ein und drücken Sie auf Vorhersage')
submit = form.form_submit_button('Vorhersage')

st.subheader("oder")

stocks = ("AAPL", "GOOG", "MSFT", "GME", "AMZN", "GEO", "EURUSD=X")

selected_stock = st.selectbox("wählen Sie einen aus unserer Datenbank", stocks)

if submit:
    selected_stock = name


n_years = st.slider("Years of prediction",1 ,10)
period = n_years * 365

@st.cache 
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text("Load data...")
data = load_data(selected_stock)
data_load_state.text("Loading data...done!")

st.subheader('Aktuelle Daten von: ')
st.subheader(selected_stock)
st.write(data.tail())

def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='stock_open'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='stock_close'))
    fig.layout.update(title_text='Time Series Data', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()

#Vorhersage
df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

st.subheader('Forecast data')
st.write(forecast.tail())


st.write('Forecast data')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write('Forecast components')
fig2 = m.plot_components(forecast)
st.write(fig2)

st.markdown("<h3 style='text-align: right;'>&copy; Daniel Sartison</h3>", unsafe_allow_html=True)