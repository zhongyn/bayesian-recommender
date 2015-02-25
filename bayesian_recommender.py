import numpy as np

class Feature(object):
	"""The features of an item."""

	def __init__(self, name, n, m):
		# prior probability
		# n is the num of this feature has been used to decribe an item
		# m is the num of items
		self.name = name
		self.p0 = (n+0.5)/(m+1)
		self.p1 = 1 - self.p0

	def inference(self):
		pass


class Item(object):
	"""An item."""

	def __init__(self, pa):
		self.parents = pa

	def create_weight(self):
		weights = np.zeros((self.parents.size,2,2))
		for i,p in enumerate(self.parents):
			weights[i][]

	def inference(self):
		pass

class User(object):
	"""A user."""

	def __init__(self, pa, ne):
		self.parents = pa
		self.neighbors = ne

	def cb_weight(self):
		pass

	def cf_weight(self):
		pass

	def inference(self):
		pass


class BayesianRecommender(object):
	"""A bayesian network based hybrid recommender system."""

	def __init__(self, rating_file, item_file):
		self.rating_file = rating_file
		self.item_file = item_file

	def read_data(self):
		pass

	def create_features(self):
		pass

	def create_items(self):
		pass

	def create_users(self):
		pass

	def inference(self):
		pass



























