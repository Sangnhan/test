import streamlit as st
import pandas as pd 
import numpy as np 
import yfinance as yf 
st.write("""
         #Simple Stock price APP
         Shown are the stock clossing price and volume of Google!
         """)
tickerSymbol = 'GOOGL'
tickerData = yf.Ticker(tickerSymbol)
tickerDf = tickerData.history(period='1d', start='2010-5-31', end='2020-5-31')
st.line_chart(tickerDf.Close)
st.line_chart(tickerDf.Volume)