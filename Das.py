import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import os
from datetime import datetime
from datetime import timedelta
from PIL import Image
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
st.title("Sentiment Analysis on USA Airlines Dataset")

st.sidebar.title("Analysis of Tweets 🇺🇸️")

st.markdown("Streamlit Dashboard to analyze USA airlines Dataset")

DATA_URL = ("/home/madhuri/Tweets.csv")

@st.cache(persist=True)
def load_data():
    data = pd.read_csv(DATA_URL)
    data['tweet_created'] = pd.to_datetime(data['tweet_created'])
  
    return data

data = load_data()



st.sidebar.subheader("Random Tweets Tweeted")
random_tweet = st.sidebar.radio('Sentiment', ('positive', 'neutral', 'negative'))
st.sidebar.markdown(data.query('airline_sentiment == @random_tweet')[["text"]].sample(n=5).iat[0,0])


st.sidebar.markdown('## No. of Tweets through Sentiment')
select = st.sidebar.selectbox('Visualization type', ['Histogram', 'Pie Chart'], key='1')

sentiment_count = data['airline_sentiment'].value_counts()
sentiment_count = pd.DataFrame({'Sentiment':sentiment_count.index, 'Tweets':sentiment_count.values})


if not st.sidebar.checkbox("Hide", True):
     st.markdown("## No. of Tweets by Sentiment")
     if select == "Histogram":
        fig = px.bar(sentiment_count,x='Sentiment', y='Tweets', color='Tweets', height=450)
        st.plotly_chart(fig)
     else:
         fig = px.pie(sentiment_count, values='Tweets', names='Sentiment')
         st.plotly_chart(fig)


st.sidebar.subheader("Places From where and at which time users are tweeting?")
hour = st.sidebar.slider("Hour of the day", min_value=1,max_value=23)
modified_data = data[data['tweet_created'].dt.hour == hour]
if not st.sidebar.checkbox("CLOSE", True, key='1'):
    st.markdown("## Location Based Tweets in the time of Day")
  
    st.markdown("%i Tweets between %i:00 and %i:00" % (len(modified_data), hour, (hour+1) %24))
   
    if st.sidebar.checkbox("Show Raw Data", False):
        st.write(modified_data)

st.sidebar.subheader("Airline Tweets BreakDown by Analysis")
choice = st.sidebar.multiselect('Pick airlines', ('US Airways', 'United', 'American', 'Southwest', 'Delta', 'Virgin America'), key='0')

if len(choice) > 0:
   choice_data = data[data.airline.isin(choice)]
   fig_choice = px.histogram(choice_data, x='airline', y='airline_sentiment', histfunc='count', color='airline_sentiment',
   facet_col='airline_sentiment', labels={'airline_sentiment':'tweets'}, height=600, width=800)
   st.plotly_chart(fig_choice)


st.sidebar.header("Word Cloud")
word_sentiment = st.sidebar.radio('Displaying word cloud for What Kind Of Sentiment(Emotion)?', ('positive', 'neutral', 'negative'))
from wordcloud import WordCloud
if not st.sidebar.checkbox("Close", True, key='3'):
   st.header('Word cloud for %s sentiment' % (word_sentiment)) 
   df = data[data['airline_sentiment']==word_sentiment]
   words = ' '.join(df['text'])
   processed_words = ' '.join([word for word in words.split() if 'http' not in word and not word.startswith('@') and word != 'RT'])
   wordcloud = WordCloud(stopwords=STOPWORDS, background_color='white', height=640, width=800).generate(processed_words)
   plt.imshow(wordcloud)
   plt.xticks([])
   plt.yticks([])
   st.pyplot()


