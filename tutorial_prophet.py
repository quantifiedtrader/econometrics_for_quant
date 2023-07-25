import prophet
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import streamlit as st
from datetime import date


start='2015-01-01'
end=date.today().strftime("%Y-%m-%d")
st.markdown("<h1 style='text-align: center; color: black;'>Stock Predicition App</h1>", unsafe_allow_html=True)
stocks= [
    "ADANIPORTS",
    "ASIANPAINT",
    "AXISBANK",
    "BAJAJ-AUTO",
    "BAJFINANCE",
    "BAJAJFINSV",
    "BPCL",
    "BRITANNIA",
    "CIPLA",
    "COALINDIA",
    "DRREDDY",
    "EICHERMOT",
    "GAIL",
    "GRASIM",
    "HCLTECH",
    "HDFCBANK",
    "HEROMOTOCO",
    "HINDALCO",
    "HINDUNILVR",
    "HDFC",
    "ICICIBANK",
    "IOC",
    "INDUSINDBK",
    "INFY",
    "ITC",
    "JSWSTEEL",
    "KOTAKBANK",
    "LT",
    "M&M",
    "MARUTI",
    "NESTLEIND",
    "NTPC",
    "ONGC",
    "POWERGRID",
    "RELIANCE",
    "SBIN",
    "SBILIFE",
    "SHREECEM",
    "SUNPHARMA",
    "TCS",
    "TATAMOTORS",
    "TATASTEEL",
    "TATACONSUM",
    "TECHM",
    "TITAN",
    "ULTRACEMCO",
    "UPL",
    "WIPRO",
]
selected_stock=st.selectbox("Selct the stock for prediction",stocks)
n_years=st.slider("Years of Prediction",1,4)
period=n_years*365

@st.cache_data
def load_data(ticker):
    data=yf.download(tickers=str(ticker)+".NS",start=start,end=end)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text("Load Data ....")
data=load_data(selected_stock)
data_load_state.text("Loading Data .... Done")

st.subheader("**Raw Data**")
st.write(data.tail())

def plot_raw_data():
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=data.Date,y=data.Close,name='Stock Closing Prices'))
    fig.layout.update(title_text="Time Series Data",xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()

#Forecasting
df_train = data[["Date","Close"]]
df_train = df_train.rename(columns={"Date":"ds","Close":"y"})

model=Prophet()
model.fit(df_train)

prediction = model.make_future_dataframe(periods=period)
forecast=model.predict(prediction)

st.subheader("**Forecast Data**")
st.write(forecast.tail())

st.write("**Forecast Plot**")
fig_1=plot_plotly(model,forecast)
st.plotly_chart(fig_1)

st.write("**Forecast Components**")
fig_2=model.plot_components(forecast)
st.write(fig_2)













