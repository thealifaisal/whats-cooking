# ******************************************
#
#   AUTHOR: ALI FAISAL
#   Github: https://github.com/thealifaisal/
#
# ******************************************

import os
import json
from random import shuffle
from src.pre_processing import Preprocessing
import operator


class Serialization:

    # constructor
    def __init__(self):
        self.preprocessing = Preprocessing()

    # reads raw text files and serialize them to JSON format
    # returns a list
    def readRawData(self, main_data_path):
        jsonlist = []
        folders = os.listdir(main_data_path)
        # main_data_path = bbcnews
        # folders = folders in the main directory as list
        for folder in folders:
            # files = files in the sub-folder as list
            files = os.listdir(main_data_path + folder + "/")

            for file in files:
                # for each file, a JSON object is created
                json_str = self.createJSONString(main_data_path + folder + "/" + file)
                # JSON object is added in the list
                jsonlist.append(json_str)
        return jsonlist

    def createJSONString(self, file_path):

        file = open(file_path, "r")

        tokens = self.preprocessing.tokenizer(file.read() + " ")
        lemmas = self.preprocessing.lemmatizer(tokens)

        # file_path = "../resources/bbcsport/tennis/034.txt"
        ID = file_path.split("/")[4].split(".")[0]
        label = file_path.split("/")[3]
        json_string = {
            "id": ID,
            "label": label,
            "features": lemmas
        }

        file.close()
        return json_string

    # shuffles a JSON list so that
    # there is no bias in splitting the data set into train/test
    @staticmethod
    def shuffleJSONObjects(jsonlist):
        shuffle(jsonlist)
        return

    # writes the data(list or dict) in to JSON file
    @staticmethod
    def writeToJSONFile(data, jsonfile_path):

        # opens a JSON file for writing
        file = open(jsonfile_path, "w")

        # checks the ype of data
        if type(data) is list:
            # if list, then iterates for every object in list
            # e.g: [{id, label, features}, {...}, ...]
            for obj in data:
                # obj = {id, label, features}
                json_object = json.dumps(obj, indent=4)
                file.write(json_object + "\n")

        elif type(data) is dict:
            # if dictionary
            # e.g: {class-1: {term-1: tf, term-2: tf}, class-2: {...}, ...}
            json_object = json.dumps(data, indent=4)
            file.write(json_object + "\n")

        file.close()
        return

    @staticmethod
    def sortClassTerms(class_terms):
        # source: https://www.w3resource.com/python-exercises/dictionary/python-data-type-dictionary-exercise-1.php
        # noinspection PyTypeChecker
        for ct in class_terms:
            class_terms[ct] = dict(sorted(class_terms[ct].items(), key=operator.itemgetter(1),
                                          reverse=True))
        return

    @staticmethod
    def classTermFrequency(data):
        # e.g: data = [{id: id1, label: lb1, features: {term: tf}}, {}, {}, ...]

        # class_terms = {class-1: {term-1: tf}, class-2: {}, ...}
        class_terms = {}

        # iterates for every file in train-set
        for obj in data:

            label = obj["cuisine"]
            features_dict = obj["ingredients"]

            # checks if label/class is in the dict
            if label not in class_terms.keys():
                # if not, inserts a label with a dict => obj["features]
                # feat = tuple(term, tf); feat[0] => term, feat[1] => tf
                # using dict comprehension
                class_terms[label] = {feat[0]: feat[1] for feat in features_dict.items()}
            else:
                # fetches dictionary => {term, tf, ...} against label
                dictionary: dict = class_terms[label]
                # iterates for every for term
                # dict.items() returns a list of tuples
                for feat in features_dict.items():
                    # feat = tuple(term, tf); feat[0] => term, feat[1] => tf
                    term = feat[0]
                    tf = feat[1]
                    if term not in dictionary.keys():
                        dictionary[term] = tf
                    else:
                        # if exists, fetches the tf for the term (feat[0]) in class => dictionary.get(feat[0])
                        # fetches the tf for the term (feat[0]) in the current doc => feat[1]
                        # sums the two tf, and assigns this new tf against the term (feat[0]) in class
                        dictionary[term] = dictionary.get(term) + tf
                        # updates the dictionary
                        class_terms[label] = dictionary
        return class_terms

    @staticmethod
    def importStopList(path):
        # a stop-word file is opened and parsed to be saved as a list
        # when path is not empty
        if path != "":
            try:
                # try opening a file
                stop_file = open(path, "r")
                # when opens, parse the file and save stop-words to list
                stoplist = stop_file.read().split("\n")
                # and close the file
                stop_file.close()
            except FileNotFoundError:
                # if file not opens, an empty list is returned
                stoplist = []
        else:
            # if path is empty, an empty list is returned
            stoplist = []

        # returning a list
        return stoplist

    # cleans all strings of Ingredient in all the JSON objects
    def cleanIngredients(self, json_list):
        # iterating over each JSON object in json_list
        for obj in json_list:
            # e.g: cleaned_ing_dict = {ing-1: tf, ing-2: tf, ...}
            cleaned_ing_dict = {}
            # iterating for each ingredient in the list: obj["ingredients"]
            for ing in obj["ingredients"]:
                # returns a clean ingredient
                cleaned_ing = self.cleanSingleIngredient(ing)
                # if cleaned_ing is not in dict
                if cleaned_ing not in cleaned_ing_dict.keys():
                    cleaned_ing_dict[cleaned_ing] = 1  # initially tf is 1
                else:
                    count = cleaned_ing_dict.get(cleaned_ing)  # fetch tf of lemma
                    count += 1  # inc tf
                    cleaned_ing_dict[cleaned_ing] = count  # update tf
            # assigning the ingredients dict with tf at index: obj["ingredients"]
            obj["ingredients"] = cleaned_ing_dict

        # returns a json_list after cleaning ingredients from each JSON object
        # e.g: json_list = [{id: id1, cuisine: cs1, ingredients: {ing-1, tf, ing-2: tf, ...}}, ...]
        return json_list

    # cleans a single string of Ingredient
    def cleanSingleIngredient(self, ingredient):
        # e.g: ingredient = "black olives"
        # removes accent from the string
        ingredient = self.preprocessing.strip_accents(ingredient)
        # tokenize the string and returns a list
        # e.g: token_list = ['black', 'olives']
        token_list = self.preprocessing.tokenizer(ingredient+" ")
        # returns a dictionary with lemmas as keys and tf's as their values
        # e.g: lemma_set = {black: 1, olive: 1, ...}
        lemma_set = self.preprocessing.lemmatizer(token_list)
        # joins all the keys in the dict with spaces to form a string
        cleaned_ingredient = " ".join(lemma_set.keys())
        # returns a cleaned ingredient
        # e.g: black olive
        return cleaned_ingredient
