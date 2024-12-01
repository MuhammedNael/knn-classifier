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
	
	# calculate the distance between two instances according to the distance metric
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
			# features are all the values except the class label
			features = np.array([value for key, value in instance.items() if not key.startswith("PlayTennis")])
			dist = self.distance(features, test_instance)
			distances.append((instance, dist))
		
		# sort the distances in ascending order. x[1] is the distance
		distances.sort(key=lambda x: x[1])
		neighbors = distances[:self.K]
		
		print("Neighbors with distances")
		print("Neighbors with distances", file=log)
		for neighbor in neighbors:
			print(neighbor[0], neighbor[1])
			print(neighbor[0], neighbor[1], file=log)
		
		# count classes to determine the majority class
		classes_count = {"PlayTennis_No": 0, "PlayTennis_Yes": 0}
		for neighbor in neighbors:
			# count the number of instances of each class. neighbor[0] is the instance
			for label in ["PlayTennis_No", "PlayTennis_Yes"]:
				classes_count[label] += neighbor[0][label]
		
		# highest count is returned as the predicted class
		return "Yes" if classes_count["PlayTennis_Yes"] > classes_count["PlayTennis_No"] else "No"

		
	def evaluate(self, test_data):
		correct = 0
		tp = tn = fp = fn = 0
		
		for instance in test_data:
			# features for the test instance
			features = np.array([value for key, value in instance.items() if not key.startswith("PlayTennis")])
			predicted_class = self.predict(features)
			actual_class = "Yes" if instance["PlayTennis_Yes"] == 1 else "No"
			
			print("Predicted class: ", predicted_class)
			print("Actual class: ", actual_class, "\n")
			print("Predicted class: ", predicted_class, file=log)
			print("Actual class: ", actual_class, "\n", file=log)
			
			# updating matrix's values
			if predicted_class == "Yes" and actual_class == "Yes":
				tp += 1
			elif predicted_class == "No" and actual_class == "No":
				tn += 1
			elif predicted_class == "Yes" and actual_class == "No":
				fp += 1
			elif predicted_class == "No" and actual_class == "Yes":
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
	# one-hot encode the data to convert categorical features to numerical values
	one_hot_encoded_data = []
	
	unique_values = {
		"Outlook": ["Sunny", "Overcast", "Rain"],
		"Temperature": ["Hot", "Mild", "Cool"],
		"Humidity": ["High", "Normal"],
		"Wind": ["Weak", "Strong"],
		"PlayTennis": ["No", "Yes"]
	}
	
	for instance in data:
		one_hot_instance = {}
		
		# for each unique value of each feature, create a new feature with the value 1 if the instance has that value, 0 otherwise
		for key in unique_values:
			if key in instance:
				for value in unique_values[key]:
					one_hot_instance[f"{key}_{value}"] = 1 if instance[key] == value else 0
		
		one_hot_encoded_data.append(one_hot_instance)
	
	return one_hot_encoded_data

 
 
log = open("log.txt", "w")    
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
	
	# print the class counts of yes and no
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
	# label the data with numerical values using one-hot encoding
	preprocessed_data = preprocess(data)
	test_data = preprocessed_data
	# test each instance in the test dataset using evaluate method
	accuracy = classifier.evaluate(test_data)
	print("Accuracy: ", accuracy)
	print("Accuracy: ", accuracy, file=log)
	
	
if __name__ == "__main__":
	main()
	