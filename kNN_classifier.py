import numpy as np
import json

class kNN_classifier:
    def __init__(self, K, distance_metric):
        self.K = K
        self.distance_metric = distance_metric
        self.data = self.load_data()
        
    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))
        
    def manhattan_distance(self, x1, x2):
        return np.sum(np.abs(x1 - x2))
    
    def load_data(self):
        with open('dataset.json') as f:
            self.data = json.load(f)
            self.data = preprocess(self.data)
        return self.data
    
    def distance(self, x1, x2):
        if self.distance_metric == 'euclidean':
            return self.euclidean_distance(x1, x2)
        elif self.distance_metric == 'manhattan':
            return self.manhattan_distance(x1, x2)
        else:
            raise ValueError("Invalid distance metric")
        
    def predict(self, test_instance):
        distances = []
        for instance in self.data:
            features = np.array([instance[key] for key in instance if key not in ['Day', 'PlayTennis']])
            dist = self.distance(features, test_instance)
            distances.append((instance, dist))
            
        # x[1] is the distance, lambda function is used to sort the distances based on the distance
        distances.sort(key=lambda x: x[1])
        # get the K nearest neighbors
        neighbors = distances[:self.K]
        
        # printing neighbors with their distances
        print("Neighbors with distances")
        print("Neighbors with distances", file=log)
        for neighbor in neighbors:
            print(neighbor[0], neighbor[1])
            print(neighbor[0], neighbor[1], file=log)
        
        classes_count = {}
        for neighbor in neighbors:
            # neighbor[0] is the instance, get the class label from the instance
            class_label = neighbor[0].get('PlayTennis')
            if class_label not in classes_count:
                classes_count[class_label] = 1
            else:
                # increment the vote count for the class label
                classes_count[class_label] += 1
        return max(classes_count, key=classes_count.get)
        
    def evaluate(self, test_data):
        correct = 0
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        for instance in test_data:
            # get the features of the instance
            features = np.array([instance[key] for key in instance if key not in ['Day', 'PlayTennis']])
            predicted_class = self.predict(features)
            actual_class = instance.get('PlayTennis')
            print("Predicted class: ", predicted_class)
            print("Actual class: ", actual_class, "\n")
            # log
            print("Predicted class: ", predicted_class, file=log)
            print("Actual class: ", actual_class, "\n", file=log)
            
            # calculate confusion matrix
            if predicted_class == 1 and actual_class == 1:
                tp += 1
            elif predicted_class == 0 and actual_class == 0:
                tn += 1
            elif predicted_class == 1 and actual_class == 0:
                fp += 1
            elif predicted_class == 0 and actual_class == 1:
                fn += 1
                
            if predicted_class == actual_class:
                correct += 1
        accuracy = correct / len(test_data)
        print("\nConfusion Matrix")
        print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
        print("\nConfusion Matrix", file=log)
        print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}", file=log)
        
        return accuracy
                
        
def preprocess(data):
    # convert all feature categories to numerical values
    mapping = {'Outlook': {'Sunny': 0, 'Overcast': 1, 'Rain': 2},
                'Temperature': {'Hot': 0, 'Mild': 1, 'Cool': 2},
                'Humidity': {'High': 0, 'Normal': 1},
                'Wind': {'Weak': 0, 'Strong': 1},
                'PlayTennis': {'No': 0, 'Yes': 1}}
    for instance in data:
        # key is the feature name, value is the feature value
        for key in instance:
            if key in mapping:
                instance[key] = mapping[key][instance[key]]
    return data
 
log = open("output.txt", "w")    
def main():
    # open dataset.json file and load the data
    data = None
    with open('dataset.json') as f:
        data = json.load(f)
    print("Play Tennis Dataset")
    print("Play Tennis Dataset", file=log)
    
    for instance in data:
        print(str(instance) + "\n")
        print(str(instance) + "\n", file=log)
        
    # summarize the dataset
    print("Dataset Summary")
    print("Total instances: ", len(data))
    print("Total instances per class")
    print("Dataset Summary", file=log)
    print("Total instances: ", len(data), file=log)
    print("Total instances per class", file=log)
    
    
    class_counts = {}
    for instance in data:
        # class label is the value of the key 'PlayTennis'
        class_label = instance.get('PlayTennis')
        if class_label is not None:
            if class_label not in class_counts:
                class_counts[class_label] = 1
            else:
                class_counts[class_label] += 1
    for key, value in class_counts.items():
        print(key, value)
        print(key, value, file=log)
        
    # getting K and distance metric from the user
    K = input("Enter the value of K: ")
    distance_metric = input("Enter the distance metric (euclidean or manhattan): ")
    print("K: ", K)
    print("Distance metric: ", distance_metric, "\n")
    print("K: ", K, file=log)
    print("Distance metric: ", distance_metric, "\n", file=log)
    
    # create an instance of kNN_classifier
    classifier = kNN_classifier(int(K), distance_metric)
    # label the data with numerical values
    preprocessed_data = preprocess(data)
    test_data = preprocessed_data
    # test each instance in the test dataset using evaluate method
    accuracy = classifier.evaluate(test_data)
    print("Accuracy: ", accuracy)
    print("Accuracy: ", accuracy, file=log)
    
    
if __name__ == "__main__":
    main()
    