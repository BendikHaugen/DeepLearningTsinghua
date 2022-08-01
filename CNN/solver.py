""" 训练和测试 """

import numpy as np


def train(model, criterion, optimizer, train_set, val_set, max_epoch, batch_size, disp_freq):
	avg_train_loss, avg_train_acc = [], []
	avg_val_loss, avg_val_acc = [], []

	# Training process
	for epoch in range(max_epoch):
		batch_train_loss, batch_train_acc = train_one_epoch(model, criterion, optimizer, train_set,
													max_epoch, batch_size, disp_freq, epoch)
		batch_val_loss, batch_val_acc = validate(model, criterion, val_set, batch_size)

		avg_train_acc.append(np.mean(batch_train_acc))
		avg_train_loss.append(np.mean(batch_train_loss))
		avg_val_acc.append(np.mean(batch_val_acc))
		avg_val_loss.append(np.mean(batch_val_loss))

		print()
		print('Epoch [{}]\t Average training loss {:.4f}\t Average training accuracy {:.4f}'.format(
			epoch, avg_train_loss[-1], avg_train_acc[-1]))

		print('Epoch [{}]\t Average validation loss {:.4f}\t Average validation accuracy {:.4f}'.format(
			epoch, avg_val_loss[-1], avg_val_acc[-1]))
		print()

	return model, avg_train_loss, avg_train_acc, avg_val_loss, avg_val_acc



def train_one_epoch(model, criterion, optimizer, train_set, max_epoch, batch_size, disp_freq, epoch):
	batch_train_loss, batch_train_acc = [], []

	max_train_iteration = train_set.num_examples // batch_size

	for iteration in range(max_train_iteration):
		# Get training data and label
		train_x, train_y = train_set.next_batch(batch_size)

		# Forward pass
		logit = model.forward(train_x)
		criterion.forward(logit, train_y)

		# Backward pass
		delta = criterion.backward()
		model.backward(delta)

		# Update weights, see optimize.py
		optimizer.step(model)

		# Record loss and accuracy
		batch_train_loss.append(criterion.loss)
		batch_train_acc.append(criterion.acc)

		if iteration % disp_freq == 0:
			print("Epoch [{}][{}]\t Batch [{}][{}]\t Training Loss {:.4f}\t Accuracy {:.4f}".format(
				epoch, max_epoch, iteration, max_train_iteration,
				np.mean(batch_train_loss), np.mean(batch_train_acc)))
	return batch_train_loss, batch_train_acc


def validate(model, criterion, val_set, batch_size):
	batch_val_acc, batch_val_loss = [], []
	max_val_iteration = val_set.num_examples // batch_size

	for iteration in range(max_val_iteration):
		# Get validating data and label
		val_x, val_y = val_set.next_batch(batch_size)

		# Only forward pass
		logit = model.forward(val_x)
		loss = criterion.forward(logit, val_y)

		# Record loss and accuracy
		batch_val_loss.append(criterion.loss)
		batch_val_acc.append(criterion.acc)

	return batch_val_loss, batch_val_acc


def test(model, criterion, test_set, batch_size, disp_freq):
	print('Testing...')
	max_test_iteration = test_set.num_examples // batch_size

	batch_test_acc = []

	for iteration in range(max_test_iteration):
		test_x, test_y = test_set.next_batch(batch_size)

		# Only forward pass
		logit = model.forward(test_x)
		loss = criterion.forward(logit, test_y)

		batch_test_acc.append(criterion.acc)

	print("The test accuracy is {:.4f}.\n".format(np.mean(batch_test_acc)))
