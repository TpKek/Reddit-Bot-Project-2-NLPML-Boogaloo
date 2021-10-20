import praw
from praw.models import MoreComments
from nltk.corpus import twitter_samples
from nltk.tokenize import sent_tokenize

#creating a reddit instance (if you want to read public posts, there is no need to login)
reddit = praw.Reddit(
    client_id="",
    client_secret="",
    user_agent="",
    username="RBotProject",
    password="")

#grab a subreddit from reddit using reddit instance
subreddit = reddit.subreddit("RBotProject")

#loop through first 10 hot submissions in subreddit and post the title and score
for submission in subreddit.hot(limit=10):
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
    #test text
    #text= """Hello Mr. Smith, how are you doing today? The weather is great, and city is awesome.
             #The sky is pinkish-blue. You shouldn't eat cardboard"""
    text = comment_lower
    #tokenize text (breaking down a text paragraph into smaller chunks such as words or sentence is called Tokenization)
    tokenized_text = sent_tokenize(text)
    print(text)