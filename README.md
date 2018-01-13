# Social_Media_HW

Media Mood Homework

The New York Times is currently viewed very positively. All the other news networks stayed close to neutral but CBS and BBC are the only ones viewed negatively. The scatter plot shows that many of the tweets are neutral but also that there are people with a varying opinion on all of the networks.


```python
# Import and Initialize Sentiment Analyzer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Dependencies
import tweepy
import json
import numpy as np
import pandas as pd

# Import and Initialize Sentiment Analyzer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

# Twitter API Keys
consumer_key = "TzLrhwUVMPqB0AeQoDc3lpVOk"
consumer_secret = "4ZkmmV4ASo4gJzPtMJLlUAiKBU5vWZekAzFXainTHIbJpn03Go"
access_token = "942094742950518785-7b7scF6iKbikgCaBnMGlH60NIzbwOr1"
access_token_secret = "KorsLpCIGzsoeCKI52SMkBglukDq6C855sf1BuxiG9LOh"

# Setup Tweepy API Authentication
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, parser=tweepy.parsers.JSONParser())

# Target User Account
users = ['@cnn', '@bbc','@cbs','@FoxNews','@nytimes']

# Variables for holding sentiments
compound_list = []
positive_list = []
negative_list = []
neutral_list = []
users_list = []
tweets_ago_list =[]

ago = 0

for user in users:
    ago = 0
    # Loop through 10 pages of tweets (total 200 tweets)
    for x in range(5):
    
        # Get all tweets from home feed
        public_tweets = api.user_timeline(user, page=x)
        public_tweets[0]['text']
    
        # Loop through all tweets
        for tweet in public_tweets:
            # print(tweet)
            # Run Vader Analysis on each tweet
            # print(analyzer.polarity_scores(tweet["text"]))
            compound = analyzer.polarity_scores(tweet["text"])["compound"]
            pos = analyzer.polarity_scores(tweet["text"])["pos"]
            neu = analyzer.polarity_scores(tweet["text"])["neu"]
            neg = analyzer.polarity_scores(tweet["text"])["neg"]
            ago = ago + 1
    
            # Add each value to the appropriate list
            users_list.append(user)
            compound_list.append(compound)
            positive_list.append(pos)
            negative_list.append(neg)
            neutral_list.append(neu)
            tweets_ago_list.append(ago)
            
tweet_df = pd.DataFrame({
                        'User_List':users_list,
                        'Compound':compound_list,
                        'Positive':positive_list,
                        'Negative':negative_list,
                        'Neutral':neutral_list,
                        'Tweets_Ago':tweets_ago_list})
```


```python
tweet_df.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Compound</th>
      <th>Negative</th>
      <th>Neutral</th>
      <th>Positive</th>
      <th>Tweets_Ago</th>
      <th>User_List</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.6705</td>
      <td>0.333</td>
      <td>0.667</td>
      <td>0.000</td>
      <td>1</td>
      <td>@cnn</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0772</td>
      <td>0.000</td>
      <td>0.939</td>
      <td>0.061</td>
      <td>2</td>
      <td>@cnn</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.6597</td>
      <td>0.241</td>
      <td>0.759</td>
      <td>0.000</td>
      <td>3</td>
      <td>@cnn</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.6124</td>
      <td>0.222</td>
      <td>0.778</td>
      <td>0.000</td>
      <td>4</td>
      <td>@cnn</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>5</td>
      <td>@cnn</td>
    </tr>
  </tbody>
</table>
</div>




```python
#cnn_df = tweet_df.filter(like='@nytimes', axis=5)
nytimes_df = tweet_df[(tweet_df.User_List == '@nytimes')]
cnn_df = tweet_df[(tweet_df.User_List == '@cnn')]
foxnews_df = tweet_df[(tweet_df.User_List == '@FoxNews')]
bbc_df = tweet_df[(tweet_df.User_List == '@bbc')]
cbs_df = tweet_df[(tweet_df.User_List == '@cbs')]

bbc_df.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Compound</th>
      <th>Negative</th>
      <th>Neutral</th>
      <th>Positive</th>
      <th>Tweets_Ago</th>
      <th>User_List</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>100</th>
      <td>-0.7845</td>
      <td>0.535</td>
      <td>0.465</td>
      <td>0.000</td>
      <td>1</td>
      <td>@bbc</td>
    </tr>
    <tr>
      <th>101</th>
      <td>-0.7506</td>
      <td>0.348</td>
      <td>0.652</td>
      <td>0.000</td>
      <td>2</td>
      <td>@bbc</td>
    </tr>
    <tr>
      <th>102</th>
      <td>-0.4404</td>
      <td>0.182</td>
      <td>0.818</td>
      <td>0.000</td>
      <td>3</td>
      <td>@bbc</td>
    </tr>
    <tr>
      <th>103</th>
      <td>0.4404</td>
      <td>0.000</td>
      <td>0.854</td>
      <td>0.146</td>
      <td>4</td>
      <td>@bbc</td>
    </tr>
    <tr>
      <th>104</th>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>5</td>
      <td>@bbc</td>
    </tr>
  </tbody>
</table>
</div>




```python
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 8))
plt.scatter(nytimes_df['Tweets_Ago'].tolist(), nytimes_df['Compound'].tolist(), s=20, c='b', marker="o", label='NYTimes')
plt.scatter(cnn_df['Tweets_Ago'].tolist(), cnn_df['Compound'].tolist(), s=20, c='g', marker="o", label='CNN')
plt.scatter(foxnews_df['Tweets_Ago'].tolist(), foxnews_df['Compound'].tolist(), s=20, c='r', marker="o", label='Fox_News')
plt.scatter(bbc_df['Tweets_Ago'].tolist(), bbc_df['Compound'].tolist(), s=20, c='c', marker="o", label='BBC')
plt.scatter(cbs_df['Tweets_Ago'].tolist(), cbs_df['Compound'].tolist(), s=20, c='m', marker="o", label='NYTimes')

plt.title('Sentiment Analysis for Media (1/12/17)')

plt.legend(loc='upper center', bbox_to_anchor=(1.2, 0.8), shadow=False, ncol=1)
plt.xlabel('Tweets Ago')
plt.ylabel('Tweet Polarity')
plt.grid()
plt.savefig('Media_Sentiment.png')
plt.show()
```


![png](output_4_0.png)



```python
avg_list = []
avg_list.append(cnn_df['Compound'].mean())
avg_list.append(bbc_df['Compound'].mean())
avg_list.append(cbs_df['Compound'].mean())
avg_list.append(foxnews_df['Compound'].mean())
avg_list.append(nytimes_df['Compound'].mean())

avg_list
```




    [0.04350900000000001,
     -0.07566499999999998,
     -0.047193000000000006,
     0.08438999999999998,
     0.3387340000000001]




```python
x_axis =np.arange(len(users))
avg_plot = plt.bar(x_axis, avg_list, color="b", align="edge",width=1)
avg_plot[0].set_color('c')
avg_plot[1].set_color('g')
avg_plot[2].set_color('r')
avg_plot[3].set_color('b')
avg_plot[4].set_color('y')
tick_locations = [value+0.4 for value in x_axis]
plt.xticks(tick_locations, users)
plt.title("Overall Media Sentiment based on Tweeter (1/12/18)")
plt.xlabel("Media")
plt.ylabel("Sentiment")
plt.savefig('Media_Mood.png')
plt.show()
```


![png](output_6_0.png)



```python
tweet_df.to_csv('media_mood.csv')
```
