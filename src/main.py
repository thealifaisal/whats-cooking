# ******************************************
#
#   AUTHOR: ALI FAISAL
#   Github: https://github.com/thealifaisal/
#
# ******************************************

from datetime import datetime
from src.ml_vsm import MachineLearning
from src.serialization import Serialization
import json

if __name__ == "__main__":

    print(datetime.now().strftime("%H:%M:%S") + ": setting paths...")
    # input paths
    data_folder_path = "../resource/train.json/"
    stop_file_path = "../out/stop-words.txt"

    # output paths
    train_file_path = "../out/train-set.json"
    test_file_path = "../out/test-set.json"
    json_file_path = "../out/json_out.json"
    # vocab_file_path = "../out/vocab.txt"
    # class_file_path = "../out/class-tf.json"

    # print(datetime.now().strftime("%H:%M:%S") + ": serializing raw data...")
    ser = Serialization()
    # imports stoplist
    stop_list = ser.importStopList(stop_file_path)
    ser.preprocessing.stop_word = stop_list

    # returns a serialized data from raw text files
    # e.g: json_list = [{"id": id1, "cuisine": lb, "ingredients": {"term": tf, ...}}, {...}, {...}, ...]
    # json_list = ser.readRawData(data_folder_path)

    json_file = open("../resource/train.json", "r")
    # e.g: json_list = [{id: id1, cuisine: cs1, ingredients: [ing-1, ing-2, ...] }, ...]
    json_list = json.load(json_file)

    # using ML class to use functions
    ml = MachineLearning()

    # returns a dict with cuisines as keys and number of docs as keys
    # e.g: labels = {filipino: 200, southern_us: 3000, ...}
    labels = ml.findLabels(json_list)

    # *******************************************************************************

    # since the data-set is too large to work on, top N JSON objects will be selected
    # print(datetime.now().strftime("%H:%M:%S") + ": selecting top N data objects...")
    # json_list = ml.selectTopNObjects(json_list, 1000)

    # closing the json file
    json_file.close()

    # returns a json_list after cleaning ingredients from each JSON object
    # e.g: json_list = [{id: id1, cuisine: cs1, ingredients: {ing-1, tf, ing-2: tf, ...}}, ...]
    print(datetime.now().strftime("%H:%M:%S") + ": cleaning data...")
    json_list = ser.cleanIngredients(json_list)

    # randomize all the files for fair splitting
    print(datetime.now().strftime("%H:%M:%S") + ": shuffling data...")
    ser.shuffleJSONObjects(json_list)
    # ser.writeToJSONFile(json_list, json_file_path)

    print(datetime.now().strftime("%H:%M:%S") + ": splitting data...")
    # splits data into 70/30 ratio
    train_set, test_set = ml.dataSplit(json_list)
    trainset_len = len(train_set)
    testset_len = len(test_set)

    print(datetime.now().strftime("%H:%M:%S") + ": lengths => train-data: " + str(trainset_len)
          + ", test-data: " + str(testset_len))

    # class_terms = ser.classTermFrequency(train_set)
    # ser.sortClassTerms(class_terms)
    # ser.writeToJSONFile(class_terms, class_file_path)

    # insert feature selection here and uncomment class_terms
    # input: train_set, class_terms
    # output: train_set (with relevant features)

    print(datetime.now().strftime("%H:%M:%S") + ": creating vocabulary...")
    vocabulary = ml.createVocabulary(train_set)
    vocabulary_len = len(vocabulary)
    print(datetime.now().strftime("%H:%M:%S") + ": vocabulary size = " + str(vocabulary_len))

    print(datetime.now().strftime("%H:%M:%S") + ": creating training vectors...")
    # e.g: train_vectors = {"doc-id": [tf-idf, ...], ...}
    train_vectors, idf_vector = ml.createTrainVectors(vocabulary, train_set)

    print(datetime.now().strftime("%H:%M:%S") + ": creating centroids...")
    # e.g: centroids = {southern_us: [tf-idf, ...], ...}
    centroids = ml.createCentroids(train_vectors, vocabulary_len, labels)

    print(datetime.now().strftime("%H:%M:%S") + ": creating testing vectors...")
    test_vectors = ml.createTestVectors(vocabulary, idf_vector, test_set)

    print(datetime.now().strftime("%H:%M:%S") + ": ************** Select Classifier **************")
    print(datetime.now().strftime("%H:%M:%S") + ": 1 -> KNN")
    print(datetime.now().strftime("%H:%M:%S") + ": 2 -> Rochhio")
    option = int(input(datetime.now().strftime("%H:%M:%S") + ": Enter Option: "))

    print(datetime.now().strftime("%H:%M:%S") + ": started testing...")

    if option == 1:
        # will run the KNN classifier
        ml.testKNN(testset_len, test_set, train_vectors, test_vectors)
    elif option == 2:
        # will run the Rochhio classifier
        train_vectors.clear()
        ml.testRochhio(testset_len, test_set, centroids, test_vectors)

    # ----------- the end -----------
