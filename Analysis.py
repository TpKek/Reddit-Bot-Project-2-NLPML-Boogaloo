import praw
from praw.models import MoreComments
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords, twitter_samples
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
import pandas as pd

#use to filter out deduntant letters/words
sw = set(stopwords.words("english"))
punct = {".",",","!","?"}
unwanted = set.union(sw,punct)

#lexicon normalization (lemmatization reduces words to their base word, which is linguistically correct lemmas.)
lem = WordNetLemmatizer()

#tweet data to use for testing Reddit posts
positive = twitter_samples.strings('positive_tweets.json')
negative = twitter_samples.strings('negative_tweets.json')

#create a dataframe from positive tweets
df = pd.DataFrame(positive, columns=["Tweet"])

#add a column to dataframe for positive sentiment value 1
df["Sentiment"] = 1

#create a dataframe for negative tweets
df2 = pd.DataFrame(negative, columns=["Tweet"])

#add a column to dataframe for negative sentiment value 0
df2["Sentiment"] = 0

#combine positive and negative tweets in one single dataframe
df = df.append(df2, ignore_index=True)
df = df.sample(frac = 1) 
df.reset_index(drop=True, inplace=True)

#displaying shape of dataset
print("Dataset size:",df.shape)
df.groupby("Sentiment").count()

#converting all tweets to lowercase
def convert_to_lowercase(tweet):
 return tweet.lower()
 
df["Tweet"] = df["Tweet"].apply(lambda x: convert_to_lowercase(x))

#lexicon normalization (lemmatization reduces words to their base word, which is linguistically correct lemmas.)
lem = WordNetLemmatizer()

#create a sentiment instance for testing.
sia = SentimentIntensityAnalyzer()

#creating a reddit instance (if you want to read public posts, there is no need to login)
reddit = praw.Reddit(
    client_id="",
    client_secret="",
    user_agent="",
    username="RBotProject",
    password="")

#grab a subreddit from reddit using reddit instance
subreddit = reddit.subreddit("RBotProject")

#Check to see if it finds the designated subreddit post for the test
for submission in subreddit.search("Testing Neural Network Sentiment Analysis"):
    print("**********")
    #print title of posts
    print(submission.title)
    #print post score(upvotes - downvotes)
    print("Score: ", submission.score)

#read comments on subreddit
for comment in submission.comments:
    #making sure all the words in the comment is lowercase
    comment_lower = comment.body.lower()
    #This is to ignore the "Load more comments" object on Reddit
    if isinstance(comment, MoreComments):
        continue
    #tokenize comments (split a sentence into words or "tokens")
    word_tokens = word_tokenize(comment_lower)
    #lemmatize every word in word token list
    leml = [lem.lemmatize(word) for word in word_tokens]
    #print(leml)
    filtered_list = [w for w in leml if not w in unwanted]
    #print(filtered_list)

    filtered_sentence = " ".join(filtered_list)
    #print(filtered_sentence)

    polscore = sia.polarity_scores(filtered_sentence)
    print("*******************************")
    print(comment.body)
    print("sentence was rated as ", polscore['neg']*100, "% Negative")
    print("sentence was rated as ", polscore['neu']*100, "% Neutral")
    print("sentence was rated as ", polscore['pos']*100, "% Positive")
    print("*******************************")

    #TODO:Start with a training model to make sentiment more accurate