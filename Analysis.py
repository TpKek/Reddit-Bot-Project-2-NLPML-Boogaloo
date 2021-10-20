import praw
from praw.models import MoreComments
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

#use to filter out deduntant letters/words
sw = set(stopwords.words("english"))
punct = {".",",","!","?"}
unwanted = set.union(sw,punct)

#create a sentiment instance for testing.
sia = SentimentIntensityAnalyzer()
#lexicon normalization 
# #(lemmatization reduces words to their base word, which is linguistically correct lemmas.)
lem = WordNetLemmatizer()

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

    #init var for later use
    polRating = " "
    if polscore["compound"] <= 0:
        polRating = "Negative"
    elif polscore["compound"] <= 0.5 and polscore["compound"] > 0:
        polRating = "Neutral"
    else :
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
    rep =""
    rep += "sentence was rated as " + '%.2f' %(polscore['neg']*100) + "% Negative.\n\n"
    rep += "sentence was rated as " + '%.2f' %(polscore['neu']*100) + "% Neutral.\n\n"
    rep += "sentence was rated as " + '%.2f' %(polscore['pos']*100) + "% Positive.\n\n"
    rep += "overall sentence rating conducted by VADER: "+ polRating + ".\n\n"

    #post sentiment of comment as a reply
    comment.reply(rep)

    #NOTE: I Realised I tried doing what VADER already does.
    #NOTE to self: Read the documentation properly next time, dummy.