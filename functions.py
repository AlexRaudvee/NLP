### imports 
import nltk
import string
import re
import contractions
import emoji

from nltk.corpus import stopwords
from urllib.parse import urlparse
from autocorrect import Speller
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.stem import PorterStemmer, SnowballStemmer, LancasterStemmer


# Download WordNet data (needed for lemmatization) and punkt for normal functioning of tokinizer and averaged perceptron tagger for POS tagging
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Download stopwords data for removing of stopwords
nltk.download('stopwords')
check = Speller(lang='en')

def create_antonym_dictionary():
    antonyms = {}
    for synset in list(wordnet.all_synsets()):
        for lemma in synset.lemmas():
            for antonym in lemma.antonyms():
                antonyms[lemma.name()] = antonym.name()
    return antonyms

antonyms = create_antonym_dictionary()
########################## FUNCTIONS FOR PREPROCESSING OF THE TEXT ##########################

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

def remove_numbers(text: str) -> str:
    """
    This function removes any numbers from the text\n
    Parameters:\n
    text: string from which we want to remove numbers\n
    Returns:\n
    String (text) which has no numbers in it
    """
    text = re.sub(r'\d+', '', text)
    return text

def remove_hashtags(text: str) -> str:
    """
    This function removes any hashtags from the text\n
    Parameters:\n
    text: string from which we want to remove hashtags\n
    Returns:\n
    String (text) which has no hashtags in it 
    """
    text = re.sub(r'#\w+\b', '', text)
    return text

def remove_usernames(text: str) -> str:
    """
    This function removes any usernames from the text\n
    Parameters:\n
    text: string from which we want to remove usernames\n
    Returns:\n
    String (text) which has no usernames in it 
    """
    text = re.sub(r'@\w+\b', '', text)
    return text

def correct_spelling(text: str) -> str:
    """
    This function corrects the spelling of the text\n
    Parameters:\n
    text: string in which we want to correct the spelling of the words\n
    Returns:\n
    String (text) which has correct spessing of the words
    """

    text = check(text)

    return text

def lemmatizer(text):
    """
    This function provides the lemmatization of the text\n
    Parameters:\n
    text: string in which we want to perform lemmatization of the words\n
    Returns:\n
    String (text) which has all words being lemmatized
    """
    if type(text) == str:
        lemmatizer = WordNetLemmatizer()
        tokens = nltk.word_tokenize(text)
        
        # Lemmatize each word
        lemmatized_words = [lemmatizer.lemmatize(word, pos=wordnet.VERB) for word in tokens]
        
        return ' '.join(lemmatized_words)
    
    elif type(text) == list:
        lemmatizer = WordNetLemmatizer()
        lemmatized_words_list = []
        for word in text:
            word_lemmatized = lemmatizer.lemmatize(word[0], pos=wordnet.VERB)
            word_tag = word[1]
            lemmatized_words_list.append((word_lemmatized, word_tag))

        return lemmatized_words_list


def stemmer(text, int_: int = 1) -> list:
    """
    This function applies 1 out of 3 stemming algorithms on the text\n
    Parameters:\n
    text: string in which we want to do stemming of the words\n
    int_: 1: PorterStemmer (commonly used and default for this function), 2: SnowballStemmer, 3: LancasterStemmer
    Returns:\n
    String (text) which has all words being stemmed
    """

    # Apply stemming using different algorithms
    if type(text) == str:
        if int_ == 1:
            porter = PorterStemmer()
            stemmed_words_porter = [porter.stem(word) for word in text.split()]
            return stemmed_words_porter
        elif int_ == 2:
            snowball = SnowballStemmer('english')
            stemmed_words_snowball = [snowball.stem(word) for word in text.split()]
            return stemmed_words_snowball
        elif int_ == 3:
            lancaster = LancasterStemmer()
            stemmed_words_lancaster = [lancaster.stem(word) for word in text.split()]
            return stemmed_words_lancaster
        else:
            print("Check if the 'int_' parameter is correct there is only 3 Stemmers")
            raise NameError
    elif type(text) == list:
        if int_ == 1:
            porter = PorterStemmer()
            stemmed_words_porter = []
            for word in text:
                stemmed_word_porter = porter.stem(word[0])
                stemmed_words_porter.append((stemmed_word_porter, word[1]))
            return stemmed_words_porter
        elif int_ == 2:
            snowball = SnowballStemmer('english')
            stemmed_words_snowball = []
            for word in text:
                stemmed_word_snowball = porter.stem(word[0])
                stemmed_words_snowball.append((stemmed_word_snowball, word[1]))
            return stemmed_words_snowball
        elif int_ == 3:
            lancaster = LancasterStemmer()
            stemmed_words_lancaster = []
            for word in text:
                stemmed_word_lancaster = porter.stem(word[0])
                stemmed_words_lancaster.append((stemmed_word_lancaster, word[1]))
            return stemmed_words_lancaster
        else:
            print("Check if the 'int_' parameter is correct there is only 3 Stemmers")
            raise NameError
    else:
        print(f'Check the text type input, your was: {text}')
        raise TypeError

def word_expansion(text: str) -> str:
    """
    This function expands english words and slang\n
    Parameters:\n
    text: string in which we want to expand the words or slang\n
    Returns:\n
    String (text) which has all words and slang being expanded
    """
    return contractions.fix(text)

def handle_negation(text: str, antonym_dict: dict=antonyms) -> str:
    """
    This function replaces the negation by antonyms for example "not good" by "bad"\n
    Parameters:\n
    text: string in which we want to replace the negation and word by antonyms\n
    Returns:\n
    String (text) which has negations replaces with antonyms where it's possible
    """
    tokens = nltk.word_tokenize(text)
    negation_words = ['not', 'no', 'n\'t']  # Add more negation words as needed
    
    # Identify negation words and replace with antonyms
    i = 0
    while i < len(tokens):
        if tokens[i].lower() in negation_words and i+1 < len(tokens):
            negated_word = tokens[i+1].lower()
            if negated_word in antonym_dict:
                tokens[i] = antonym_dict[negated_word]
                del tokens[i+1]
            i += 1
        i += 1
    
    updated_text = ' '.join(tokens)

    return updated_text

def POS_tagging(text: str) -> list:
    """
    This function does POS tagging for the text that you put in\n
    Parameters:\n
    text: string in which we want to perform a POS tagging\n
    Returns:\n
    List of Tuples where in the tuple we have 1st values is the word and the 2nd value is the tag
    """
    tokens = nltk.word_tokenize(text)
    pos_tags = nltk.pos_tag(tokens)

    return pos_tags

def remove_emoji(text: str) -> str:
    """
    This function removes any emojis from the text\n
    Parameters:\n
    text: string in which we want to remove any emojis\n
    Returns:\n
    Original text (string), but without emojis 
    """
    
    emoji_pattern = re.compile(emoji.get_emoji_regexp())
    text_without_emojis = emoji_pattern.sub(r'', text)

    return text_without_emojis

########################## FUNCTIONS FOR VECTORIZATION OF THE TOKENS ##########################