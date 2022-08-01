""" Network Class. Add is_training flag for Dropout."""

class Network():
	def __init__(self):
		self.layerList = []
		self.numLayer = 0
		self._is_training = True

	@property
	def is_training(self):
		return self._is_training

	@is_training.setter
	def is_training(self, val):
		for layer in self.layerList:
			layer.is_training = val
		self._is_training = val

	def add(self, layer):
		self.numLayer += 1
		self.layerList.append(layer)

	def forward(self, x):
		# forward layer by layer
		for i in range(self.numLayer):
			x = self.layerList[i].forward(x)
		return x

	def backward(self, delta):
		# backward layer by layer
		for i in reversed(range(self.numLayer)): # reversed
			delta = self.layerList[i].backward(delta)
