import unicodedata
from src.pre_processing import Preprocessing

preprocessing = Preprocessing()


def strip_accents(text):
    try:
        text = unicode(text, 'utf-8')
    except NameError:  # unicode is a default on python 3
        pass

    text = unicodedata.normalize('NFD', text) \
        .encode('ascii', 'ignore') \
        .decode("utf-8")

    return str(text)


# cleans a single string of Ingredient
def cleanSingleIngredient(ingredient):
    # removes accent from the string
    ingredient = preprocessing.strip_accents(ingredient)
    # tokenize the string and returns a list
    # e.g: token_list = ['black', 'olives']
    token_list = preprocessing.tokenizer(ingredient+" ")
    # returns a dictionary with lemmas as keys and tf's as their values
    # e.g: lemma_set = {black: 1, olive: 1, ...}
    lemma_set = preprocessing.lemmatizer(token_list)
    # joins all the keys in the dict with spaces to form a string
    cleaned_ingredient = " ".join(lemma_set.keys())
    # returns a cleaned ingredient
    return cleaned_ingredient


s = cleanSingleIngredient("écheveaux-olives")


