# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 23:14:03 2024

@author: umars
"""

import numpy as np
import pickle
import streamlit as st
import yfinance as yf
import pandas as pd
import praw
import plotly.graph_objs as go

st.set_page_config(
    page_title="STOCK PRICE PREDICTION WEB APP",
    page_icon="C:/Users/umars/Downloads/stock-exchange.png"
    )

page_bg_img = """
<style>
[data-testid="stAppViewContainer"]{
background-image: "C:/Users/umars/Downloads/abstract-financial-chart-trend-line-260nw-766689100.webp";
background-size: cover;
}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

st.image("C:/Users/umars/Downloads/nvidia.jpeg", width=180)


# Display historical data
st.write("## NVIDIA Historical Data (2022-2024)")

# Fetch historical data from Yahoo Finance
nvidia_data = yf.download('NVDA', start='2022-01-01', end='2024-06-28')

if not nvidia_data.empty:
    st.write(nvidia_data)
else:
    st.write("No data found for the specified period.")
   
# Reddit API setup
reddit = praw.Reddit(
        client_id='p8fHIpWa4uoINH_b-5EvDg',
        client_secret='Ly_z9P63eWbyX_VwfkIFJYGcW71C_A',
        user_agent='stockpriceprediction'
)

# Adding a selectbox for filtering Reddit posts
st.write("## Reddit Posts")
subreddit_name = st.text_input("Enter the subreddit name", value="wallstreetbets")
filter_type = st.selectbox("Select post filter", ["hot", "new", "rising", "top"])

# Fetch and display Reddit posts based on the selected filter
def fetch_reddit_posts(subreddit, filter_type):
    if filter_type == "hot":
        return reddit.subreddit(subreddit).hot(limit=10)
    elif filter_type == "new":
        return reddit.subreddit(subreddit).new(limit=10)
    elif filter_type == "rising":
        return reddit.subreddit(subreddit).rising(limit=10)
    elif filter_type == "top":
        return reddit.subreddit(subreddit).top(limit=10, time_filter='day')
    
if subreddit_name:
    try:
        posts = fetch_reddit_posts(subreddit_name, filter_type)
        for post in posts:
            st.write(f"### {post.title}")
            if post.is_self:
                st.write(post.selftext)
            elif hasattr(post, 'preview') and 'images' in post.preview:
                image_url = post.preview['images'][0]['source']['url']
                st.image(image_url)
            else:
                st.write(f"[Link to post]({post.url})")
            st.markdown(f"<p style='font-size:20px;'>Score: {post.score} | Comments: {post.num_comments}</p>", unsafe_allow_html=True)
            st.write("---")
    except Exception as e:
        st.error(f"Error fetching posts: {e}")


# Display Historical Prices
st.write("## NVIDIA CORPORATION STOCK")
def plot_nvidia_closing_prices():
    # Fetch historical data from Yahoo Finance
    dataset = pd.read_excel(r"C:\Users\umars\Downloads\hellofinalnvidiadfmerged2.xlsx")
    dataset.reset_index(inplace=True)
    
    # Create the Plotly figure
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=dataset['Date'],
        y=dataset['Close'],
        mode='lines',
        name='NVIDIA Closing Prices',
        line=dict(color='royalblue', width=2)  
    ))

    # Update layout with enhanced aesthetics
    fig.update_layout(
        title=dict(text='NVIDIA Closing Prices Over Time', font=dict(size=24)), 
        xaxis_title='Date',
        yaxis_title='Closing Price ($)',
        legend=dict(title='Legend', font=dict(size=14)), 
        template='plotly_white',  
        xaxis=dict(tickangle=45, showgrid=True),  
        yaxis=dict(showgrid=True),
        margin=dict(l=50, r=50, t=80, b=50),  
    )
    
    # Update x-axis to rotate the tick labels
    fig.update_xaxes(tickangle=45)
    
    return fig

fig = plot_nvidia_closing_prices()
st.plotly_chart(fig)


   
# Display PREDICTED PRICE V/S ACTUAL PRICE
st.write("## PREDICTED PRICE V/S ACTUAL PRICE")
#st.image("C:/Users/umars/Pictures/Screenshots/Screenshot 2024-06-30 212650.png", caption="THE MODEL'S PREDICTED PRICE V/S ACTUAL PRICE NVIDIA", use_column_width=True)
# Data
predicted_prices = [132.32678548, 136.89516422, 133.00205525, 127.97370234, 119.64416125, 127.37224039, 127.70624829, 125.17880508]
actual_prices = [135.58, 130.78, 126.57, 118.11, 126.09, 126.40, 123.99, 123.54]
dates = ["2024-06-18", "2024-06-20", "2024-06-21", "2024-06-24", "2024-06-25", "2024-06-26", "2024-06-27", "2024-06-28"]

# Create a DataFrame
predicted_data = {
    'Date': dates,
    'Predicted Price': predicted_prices,
    'Actual Price': actual_prices
}
predicted_df = pd.DataFrame(predicted_data)

# Plotly figure
fig = go.Figure()

# Add traces for predicted and actual prices
fig.add_trace(go.Scatter(x=predicted_df['Date'], y=predicted_df['Predicted Price'],
                         mode='lines+markers', name='Predicted Price', line=dict(color='blue', width=2)))

fig.add_trace(go.Scatter(x=predicted_df['Date'], y=predicted_df['Actual Price'],
                         mode='lines+markers', name='Actual Price', line=dict(color='green', width=2)))

# Update layout
fig.update_layout(
    title='Predicted vs Actual Prices',
    xaxis_title='Date',
    yaxis_title='Price',
    xaxis=dict(tickangle=45),
    template='plotly_white'  
)

# Show plotly figure
st.plotly_chart(fig)


# Load the model
loaded_model = pickle.load(open("C:/LORD_AI_INTERNSHIP/NVIDIA_STOCK_PREDICTION_USING_SENTIMENT_ANALYSIS_WALLSTREETBETS/nvidia_stock_market_prediction.sav", 'rb'))

# Creating a function for prediction
def stock_prediction(input_data):
    input_data_as_numpy_array = np.asarray(input_data, dtype=float)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    stock_prediction = loaded_model.predict(input_data_reshaped)
    return stock_prediction[0][0]

def main():
    # Giving a title
    st.markdown("<p style='font-size:50px; color:royalblue;'>NVIDIA Stock Price Prediction</p>", unsafe_allow_html=True)


    # Getting input from the user
    Open = st.text_input(f'Enter the Opening value of the stock for the day')
    High = st.text_input('Enter the Highest value of the stock for the day')
    Low = st.text_input('Enter the Lowest value of the stock for the day')
    Close = st.text_input('Enter the Closing value of the stock for the day')
    Volume = st.text_input('Enter the Volume traded of the stock for the day')
    Sentiment = st.text_input('Enter the Sentiment value of the stock for the day')
    
    try:
        Open = float(Open)
        High = float(High)
        Low = float(Low)
        Close = float(Close)
        Volume = float(Volume)
        Sentiment = float(Sentiment)
    except ValueError:
        st.error("All inputs must be numeric values (integer or float).")
        return

    if st.button('STOCK PREDICTION FOR THE NEXT DAY'):
        values = [Open, High, Low, Close, Volume, Sentiment]
        
        # Prediction
        prediction = stock_prediction(values)
        st.write(f"<p style='font-size:25px; color:green;'>The predicted stock value is {prediction}</p>", unsafe_allow_html=True)

        

if __name__ == '__main__':
    main()