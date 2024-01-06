### imports 
import nltk
import string
import re
import contractions
import emoji
import spacy

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

# Load the small English model from spacy for merging of noun tokens
nlp = spacy.load("en_core_web_sm")

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

    return text.replace(parsed.scheme + "://" + parsed.netloc, "")

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

    return ' '.join(filtered_text)

def remove_emoji(text: str) -> str:
    """
    This function removes any emojis from the text\n
    Parameters:\n
    text: string in which we want to remove any emojis\n
    Returns:\n
    Original text (string), but without emojis 
    """
    
    emoji_pattern = re.compile(emoji.get_emoji_regexp())

    return emoji_pattern.sub(r'', text)

def remove_punctuation(text: str) -> str:
    """
    This function takes in text from which it is going to remove any punctuation marks\n
    Parameters:\n
    text: string the we want to preprocess\n
    Returns:\n
    Cleaned from punctuation marks string
    """

    return text.translate(str.maketrans('', '', string.punctuation))

def remove_upercase(text: str) -> str:
    """
    This function replaces any upper case letters from the text by lower case\n
    Parameters: \n
    text: string in which we want to replace uppercase by lower case\n
    Returns:\n
    Filtered out of upper case letters string 
    """

    return text.lower()

def remove_extra_whitespace(text: str) -> str:
    """
    This function removes any extra spaces from the text\n
    Parameters:\n
    text: string from which we want to remove extra spaces\n
    Returns:\n
    String(text) which has no extra space between words 
    """
    return ' '.join(text.split())

def remove_numbers(text: str) -> str:
    """
    This function removes any numbers from the text\n
    Parameters:\n
    text: string from which we want to remove numbers\n
    Returns:\n
    String (text) which has no numbers in it
    """
    return re.sub(r'\d+', '', text)

def remove_hashtags(text: str) -> str:
    """
    This function removes any hashtags from the text\n
    Parameters:\n
    text: string from which we want to remove hashtags\n
    Returns:\n
    String (text) which has no hashtags in it 
    """
    return re.sub(r'#\w+\b', '', text)

def remove_usernames(text: str) -> str:
    """
    This function removes any usernames from the text\n
    Parameters:\n
    text: string from which we want to remove usernames\n
    Returns:\n
    String (text) which has no usernames in it 
    """
    return re.sub(r'@\w+\b', '', text)

def correct_spelling(text: str) -> str:
    """
    This function corrects the spelling of the text\n
    Parameters:\n
    text: string in which we want to correct the spelling of the words\n
    Returns:\n
    String (text) which has correct spessing of the words
    """

    return check(text)

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
        tokens = nltk.word_tokenize(text)

        return [lemmatizer.lemmatize(word, pos=wordnet.VERB) for word in tokens]

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
    if type(text) is str:
        if int_ == 1:
            porter = PorterStemmer()
            return [porter.stem(word) for word in text.split()]
        elif int_ == 2:
            snowball = SnowballStemmer('english')
            return [snowball.stem(word) for word in text.split()]
        elif int_ == 3:
            lancaster = LancasterStemmer()
            return [lancaster.stem(word) for word in text.split()]
        else:
            print("Check if the 'int_' parameter is correct there is only 3 Stemmers")
            raise NameError    
    elif type(text) is list:
        if int_ == 1:
            porter = PorterStemmer()
            return [porter.stem(word) for word in text] 
        elif int_ == 2:
            snowball = SnowballStemmer('english')
            return [snowball.stem(word) for word in text] 
        elif int_ == 3:
            lancaster = LancasterStemmer()
            return [lancaster.stem(word) for word in text] 
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
    
    return ' '.join(tokens)

def multi_word_grouping(text: str) -> list:
    """
    This function is grouping the tokens if neighbor tokens are both Nouns\n
    Parameters:\n
    text: text in which we want to group any neighbour tokens\n
    Returns:\n
    List (string), returns list with grouped tokens
    """

    doc = nlp(text)

    # Merge consecutive noun phrases
    with doc.retokenize() as retokenizer:
        for np in list(doc.noun_chunks):
            retokenizer.merge(np)

    return [token.text for token in doc]

def tokenizer(text: str) -> list:
    """
    This function Tokenizes the text that we input\n
    Parameters:\n
    text: string which we want to tokenize\n
    Returns:\n
    list of tokens after tokenization 
    """
    return nltk.word_tokenize(text)

def flow_preprocessing_1(text) -> str:
    return remove_emoji(remove_usernames(remove_hashtags(remove_url(text))))

def flow_preprocessing_2(text) -> str:
    return remove_extra_whitespace(remove_numbers(remove_punctuation(remove_upercase(text))))

def flow_preprocessing_3(text) -> str:
    return handle_negation(correct_spelling(remove_upercase(remove_stop_words(text))))

def flow_preprocessing_4(text: str) -> list[str]:
    return tokenizer(lemmatizer(text))

def flow_preprocessing_5(text: str) -> list[str]:
    return tokenizer(stemmer(text))

def flow_preprocessing_6(text: str) -> list[str]:
    return tokenizer(word_expansion(text))

def flow_preprocessing_7(text: str) -> list[str]:
    return tokenizer(handle_negation(text))

def flow_preprocessing_8(text: str) -> list[str]:
    return multi_word_grouping(text)

def flow_preprocessing_9(text: str) -> list[str]:
    return tokenizer(lemmatizer(flow_preprocessing_1(text)))

def flow_preprocessing_10(text: str) -> list[str]:
    return tokenizer(lemmatizer(flow_preprocessing_2(text)))

def flow_preprocessing_11(text: str) -> list[str]:
    return tokenizer(lemmatizer(flow_preprocessing_3(text)))

def flow_preprocessing_12(text: str) -> list[str]:
    return tokenizer(stemmer(flow_preprocessing_1(text)))

def flow_preprocessing_13(text: str) -> list[str]:
    return tokenizer(stemmer(flow_preprocessing_2(text)))

def flow_preprocessing_14(text: str) -> list[str]:
    return tokenizer(stemmer(flow_preprocessing_3(text)))

def flow_preprocessing_15(text: str) -> list[str]:
    return tokenizer(word_expansion(flow_preprocessing_1(text)))

def flow_preprocessing_16(text: str) -> list[str]:
    return tokenizer(word_expansion(flow_preprocessing_2(text)))

def flow_preprocessing_17(text: str) -> list[str]:
    return tokenizer(word_expansion(flow_preprocessing_3(text)))

def flow_preprocessing_18(text: str) -> list[str]:
    return multi_word_grouping(flow_preprocessing_1(text))

def flow_preprocessing_19(text: str) -> list[str]:
    return multi_word_grouping(flow_preprocessing_2(text))

def flow_preprocessing_20(text: str) -> list[str]:
    return multi_word_grouping(flow_preprocessing_3(text))

########################## FUNCTIONS FOR VECTORIZATION OF THE TOKENS ##########################
