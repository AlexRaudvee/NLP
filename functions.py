### imports 
import nltk
import string

from nltk.corpus import stopwords
from urllib.parse import urlparse

nltk.download('stopwords')

### functions for preprocessing the text

def remove_url(text: str) -> str:
    """
    This function removes urls from the text that were input\n
    Parameters:\n
    text: string from which we want to remove the urls\n
    Returns:\n
    Text that have been cleaned from urls
    """

    parsed = urlparse(text)
    text = text.replace(parsed.scheme + "://" + parsed.netloc, "")

    return text 


def remove_stop_words(text: str) -> str:
    """
    This function removes stop words from the text that were input\n
    Parameters:\n
    text: the text which we are going to process\n
    Returns:\n
    The text which is cleaned from stop words
    """
    # Tokenize the text into words
    words = nltk.word_tokenize(text)

    # Get English stopwords from NLTK
    stop_words = set(stopwords.words('english'))

    filtered_text = [word for word in words if word.lower() not in stop_words]

    filtered_text = ' '.join(filtered_text)

    return filtered_text


def remove_punctuation(text: str) -> str:
    """
    This function takes in text from which it is going to remove any punctuation marks\n
    Parameters:\n
    text: string the we want to preprocess\n
    Returns:\n
    Cleaned from punctuation marks string
    """

    clean_text = text.translate(str.maketrans('', '', string.punctuation))

    return clean_text

def remove_upercase(text: str) -> str:
    """
    This function replaces any upper case letters from the text by lower case\n
    Parameters: \n
    text: string in which we want to replace uppercase by lower case\n
    Returns:\n
    Filtered out of upper case letters string 
    """

    lowercase_text = text.lower()

    return lowercase_text

def remove_extra_whitespace(text: str) -> str:
    """
    This function removes any extra spaces from the text\n
    Parameters:\n
    text: string from which we want to remove extra spaces\n
    Returns:\n
    String(text) which has no extra space between words 
    """
    clean_text = ' '.join(text.split())

    return clean_text


