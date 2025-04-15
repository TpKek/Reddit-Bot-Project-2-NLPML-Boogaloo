import praw
from praw.models import MoreComments
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
import time

#use to filter out deduntant letters/words
sw = set(stopwords.words("english"))
punct = set(string.punctuation)
unwanted = set.union(sw, punct)

#create a sentiment instance for testing.
sia = SentimentIntensityAnalyzer()
#lexicon normalization 
#(lemmatization reduces words to their base word, which is linguistically correct lemmas.)
lem = WordNetLemmatizer()

def process_comment(comment):
    """
    Process a Reddit comment by tokenizing, lemmatizing, and filtering unwanted words.

    Args:
        comment (str): The comment text.

    Returns:
        str: The processed comment.
    """
    comment_lower = comment.lower()
    word_tokens = word_tokenize(comment_lower)
    leml = [lem.lemmatize(word) for word in word_tokens]
    filtered_list = [w for w in leml if not w in unwanted]
    return " ".join(filtered_list)

#creating a reddit instance (if you want to read public posts, there is no need to login)
reddit = praw.Reddit(
    client_id="",
    client_secret="",
    user_agent="",
    username="RBotProject",
    password="")

#grab a subreddit from reddit using reddit instance
try:
    subreddit = reddit.subreddit("RBotProject")
except Exception as e:
    print(f"Error accessing subreddit: {e}")
    exit()

#Check to see if it finds the designated subreddit post for the test
for submission in subreddit.search("Testing Neural Network Sentiment Analysis"):
    print("**********")
    #print title of posts
    print(submission.title)
    #print post score(upvotes - downvotes)
    print("Score: ", submission.score)

#read comments on subreddit
for comment in submission.comments:
    #This is to ignore the "Load more comments" object on Reddit
    if isinstance(comment, MoreComments):
        continue

    filtered_sentence = process_comment(comment.body)
    
    polscore = sia.polarity_scores(filtered_sentence)

    #init var for later use
    #NOTE: I made an error here since 100% neutral comments come out as "Overall Negative", but I explained it in my README
    #won't change since it isn't nessasary.
    polRating = " "
    if polscore["compound"] < -0.05:
        polRating = "Negative"
    elif -0.05 <= polscore["compound"] <= 0.05:
        polRating = "Neutral"
    else:
        polRating = "Positive"

    #console test
    polscore = sia.polarity_scores(filtered_sentence)
    print("*******************************")
    print(comment.body)
    print("sentence was rated as ", polscore['neg']*100, "% Negative")
    print("sentence was rated as ", polscore['neu']*100, "% Neutral")
    print("sentence was rated as ", polscore['pos']*100, "% Positive")
    print("*******************************")

    #using this to reply multiple lines in one comment/reply
    #weird way of doing it, but PRAW doesn't allow you to reply with mutliple lines in an easier way.
    rep =""
    rep += "sentence was rated as " + '%.2f' %(polscore['neg']*100) + "% Negative.\n\n"
    rep += "sentence was rated as " + '%.2f' %(polscore['neu']*100) + "% Neutral.\n\n"
    rep += "sentence was rated as " + '%.2f' %(polscore['pos']*100) + "% Positive.\n\n"
    rep += "overall sentence rating conducted by VADER: "+ polRating + ".\n\n"

    #post sentiment of comment as a reply
    comment.reply(rep)
    time.sleep(2)  # Wait 2 seconds between replies

    #NOTE: I realised I tried doing what VADER already does.
    #NOTE to self: Read the documentation properly next time, dummy.
