# ******************************************
#
#   AUTHOR: ALI FAISAL
#   Github: https://github.com/thealifaisal/
#
# ******************************************

import math
import operator
import numpy as np
from datetime import datetime

# ignores divide by zero warning because it is already handled
np.seterr(divide='ignore', invalid='ignore')


class MachineLearning:

    @staticmethod
    def createTrainVectors(vocabulary, train_set):

        # e.g: train_vectors = {"id-1": [tf-1, tf-2, tf-3, ...], "id-2": [tf-1, tf-2, ...], ...}
        train_vectors = {}
        trainset_len = len(train_set)
        vocabulary_len = len(vocabulary)
        idf_vector = [0] * vocabulary_len

        # iterates for every vocabulary
        i = 0
        for voc in vocabulary:
            # wb_sheet.cell(i + 2, 1).value = voc

            df = 0
            for doc in range(0, trainset_len):

                doc_id = train_set[doc]["id"]
                label = train_set[doc]["cuisine"]
                # creating a composite key
                key = str(doc_id) + "-" + label

                try:
                    if key not in train_vectors.keys():
                        train_vectors[key] = [0] * vocabulary_len

                    # assigns tf of term in doc
                    train_vectors[key][i] = train_set[doc]["ingredients"][voc]
                    df += 1

                except KeyError:
                    # when, term from vocab not in doc
                    train_vectors[key][i] = 0

            idf_vector[i] = (float(format(math.log10(trainset_len / df), ".5f")))

            for doc in range(0, trainset_len):
                doc_id = train_set[doc]["id"]
                label = train_set[doc]["cuisine"]
                key = str(doc_id) + "-" + label

                tf = float(format(train_vectors[key][i], ".5f"))
                train_vectors[key][i] = float(format(idf_vector[i] * tf, ".5f"))

            i += 1

        return train_vectors, idf_vector

    @staticmethod
    def createTestVectors(vocabulary, idf_vector, test_set):
        # train_vectors = {"id-1": [tf-1, tf-2, tf-3, ...], "id-2": [tf-1, tf-2, ...], ...}
        test_vectors = {}
        test_set_len = len(test_set)
        vocabulary_len = len(vocabulary)

        # iterates for every vocabulary
        for i, voc in enumerate(vocabulary):

            for doc in range(0, test_set_len):

                doc_id = test_set[doc]["id"]
                label = test_set[doc]["cuisine"]
                key = str(doc_id) + "-" + label

                try:
                    if key not in test_vectors.keys():
                        test_vectors[key] = [0] * vocabulary_len

                    # assigns tf-idf of term in test-doc
                    tf = test_set[doc]["ingredients"][voc]
                    test_vectors[key][i] = float(format(idf_vector[i] * tf, ".5f"))

                except KeyError:
                    # when, term from vocab not in doc
                    test_vectors[key][i] = 0

        return test_vectors

    @staticmethod
    def prepareSheet(wb_sheet, train_set):
        # train_set = [{id, label, features: {term: tf}}, {}, ...]
        trainset_len = len(train_set)
        wb_sheet.cell(1, 1).value = "bag"
        for _i in range(2, trainset_len + 1):
            # e.g: doc-id = 037a, 110t, ...
            wb_sheet.cell(1, _i).value = train_set[_i - 2]["id"] + train_set[_i - 2]["cuisine"][0]
        wb_sheet.cell(1, trainset_len + 2).value = "idf"
        return

    # splits the data-set into train and test sets
    # returns trains and test sets
    @staticmethod
    def dataSplit(jsonlist):
        # jsonlist = [{id: id1, cuisine: lb1, ingredients: {term: tf}}, {}, {}, ...]

        docs_len = len(jsonlist)  # 737
        train_len = round(70 / 100 * docs_len)  # 516
        test_len = docs_len - train_len  # 221
        trainset = []
        testset = []

        for f in range(0, train_len):
            trainset.append(jsonlist[f].copy())

        for f in range(train_len, docs_len):
            testset.append(jsonlist[f].copy())

        jsonlist.clear()

        return trainset, testset

    @staticmethod
    def createVocabulary(trainset):
        vocab = []
        for file in trainset:
            for feature in file["ingredients"].keys():
                if feature not in vocab:
                    vocab.append(feature)
        return vocab

    @staticmethod
    def cosineSimilarity(train_vectors, test_vectors, test_key):
        result_set = {}
        # test_vector => list
        test_vector = test_vectors[test_key]
        np_test = np.array(test_vector)

        for train_key in train_vectors.keys():

            # train_vector => list
            train_vector = train_vectors[train_key]
            np_train = np.array(train_vector)

            dot_product = np.sum(np.multiply(np_train, np_test))

            train_modulus = np.sqrt(np.sum(np.square(np_train)))
            test_modulus = np.sqrt(np.sum(np.square(np_test)))

            try:
                cosine = dot_product / (train_modulus * test_modulus)
            except ZeroDivisionError:
                cosine = -1

            if cosine != -1:
                result_set[train_key] = float(format(cosine, ".5f"))

        result_set = dict(sorted(result_set.items(), key=operator.itemgetter(1), reverse=True))

        # cosines_file = open("../out/cosines.txt", "w")
        # cosines_file.write(str(result_set))
        # cosines_file.close()

        return result_set

    @staticmethod
    def classifyRochhio(result_set, test_composite_key):
        predicted_class = list(result_set.keys())[0]
        actual_class = test_composite_key.split("-")[1]

        if predicted_class == actual_class:
            correct = 1
        else:
            correct = 0

        return correct

    @staticmethod
    def classifyKNN(result_set, k, test_composite_key):
        # KNN:
        #   key: doc-class
        #   value: number of docs that belong to that class from top K result_set
        # e.g: KNN = {"filipino": 2, "southern_us": 1}
        # e.g: KNN = {"filipino": 1`, "italian": 1, "southern_us": 1}
        KNN = {}

        # e.g: KNN = {'063_italian': 0.13047, '141_filipino': 0.12382, '012_italian': 0.10805}
        # i=0, 1, 2, .. | item=("067_filipino", 0.12458), ...
        for i, item in enumerate(result_set.items()):

            if i >= k:
                # after selecting top k, break
                break

            # item = ("003_filipino", 0.12312)
            # item[0]="003_filipino", item[1]=0.12312
            lab = item[0].split("-")[1]

            # if a class of doc from top K result is not in KNN
            if lab not in KNN.keys():
                # insert class with its initial doc occurrence = 1
                KNN[lab] = 1
            else:
                # else increment doc occurrence from that class
                KNN[lab] += 1

        # sort KNN in descending order,
        # so that class with most docs in top K results is at starting index
        # e.g: KNN = {"filipino": 2, "tennis": 1} -> 2 docs are from class 'filipino', 1 doc is from class 'italian'
        KNN = dict(sorted(KNN.items(), key=operator.itemgetter(1), reverse=True))

        # fetch item with max doc occurrences from the start
        # e.g: max_item = ("filipino": 2)
        max_item = list(KNN.items())[0]

        # check if all values are same in KNN
        #   means that in top K results, each class occurred has same number of docs occurrences
        # e.g: KNN = {"filipino": 1`, "italian": 1, "rugby": 1}
        # all_values_same = {True, False}
        all_values_same = all(x == max_item[1] for x in KNN.values())

        if all_values_same:
            # then select the doc from result_set with max cosine value
            # e.g: doc_key = "003_filipino"
            doc_key = list(result_set.keys())[0]
            predicted_class = doc_key.split("-")[1]
        else:
            # otherwise select the class with most documents in KNN
            # e.g: max_item = ("filipino": 2)
            # max_item[0] = "filipino"
            predicted_class = max_item[0]

        actual_class = test_composite_key.split("-")[1]

        # file = open("../out/prediction.txt", "a+")

        if predicted_class == actual_class:
            # file.write("test_file: " + test_composite_key + " \t\tpredicted class: " + labels[predicted_class] + "\n")
            correct = 1
        else:
            # file.write("test_file: " + test_composite_key + " \t\tpredicted class: " + labels[predicted_class] + "\n")
            correct = 0

        # file.close()

        return correct

    # if data-set is too large for the machine, then selects top N objects from JSON data
    @staticmethod
    def selectTopNObjects(json_data, n):
        json_list = []
        for i, obj in enumerate(json_data):
            if i == n:
                break
            else:
                json_list.append(obj)

        return json_list

    @staticmethod
    def findLabels(json_data):
        labels = {}
        for obj in json_data:
            cuisine = obj["cuisine"]
            if cuisine not in labels.keys():
                labels[cuisine] = 1
            else:
                labels[cuisine] += 1
        # e.g: labels = {filipino: 200, southern_us: 3000, ...}
        return labels

    @staticmethod
    def createCentroids(train_vectors, vocabulary_len, labels):
        centroids = {}
        # e.g: doc_id = 11300-filipino
        for doc_id in train_vectors.keys():
            cuisine = doc_id.split("-")[1]
            train_vec = np.array(train_vectors.get(doc_id))
            if cuisine not in centroids.keys():
                centroids[cuisine] = [0] * vocabulary_len
            centr_vec = np.array(centroids[cuisine])
            sum_cent = np.add(train_vec, centr_vec)
            centroids[cuisine] = sum_cent

        for cuisine in centroids.keys():
            centroids[cuisine] = np.divide(centroids[cuisine], labels[cuisine])

        return centroids

    def testKNN(self, testset_len, test_set, train_vectors, test_vectors):
        # number of correct predictions
        correct_predictions = 0
        # iterating for every test file
        for i in range(0, testset_len):

            doc_id = test_set[i]["id"]
            label = test_set[i]["cuisine"]
            key = str(doc_id) + "-" + label

            # returns a result dict, where doc with highest cosine is the first item
            # result_set = {"096-filipino": 0.51468, ...} for KNN
            result_set = self.cosineSimilarity(train_vectors, test_vectors, key)  # for KNN
            # result_set = ml.cosineSimilarity(centroids, test_vectors, key)

            # k = 3; to select top k neighbors/docs from result_set
            k = 3

            # e.g: test_composite_key = 11300-filipino
            correct_predictions += self.classifyKNN(result_set, k, key)

        # accuracy = no of files correctly predicted / total files tested
        accuracy = format((correct_predictions / testset_len) * 100, ".5f")

        print(datetime.now().strftime("%H:%M:%S") + ": KNN accuracy = " + accuracy)

    def testRochhio(self, testset_len, test_set, centroids, test_vectors):
        # number of correct predictions
        correct_predictions = 0
        # iterating for every test file
        for i in range(0, testset_len):

            doc_id = test_set[i]["id"]
            label = test_set[i]["cuisine"]
            key = str(doc_id) + "-" + label

            # returns a result dict, where doc with highest cosine is the first item
            # result_set = {"filipino": 0.51468, ...} for Rochhio
            result_set = self.cosineSimilarity(centroids, test_vectors, key)

            # e.g: test_composite_key = 11300-filipino
            correct_predictions += self.classifyRochhio(result_set, key)

        # accuracy = no of files correctly predicted / total files tested
        accuracy = format((correct_predictions / testset_len) * 100, ".5f")

        print(datetime.now().strftime("%H:%M:%S") + ": Rochhio accuracy = " + accuracy)
