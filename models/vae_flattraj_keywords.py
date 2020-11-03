'''Using VAE on keyword occurrence submissions dataset using MLP

The VAE has a modular design. The encoder, decoder and VAE
are 3 models that share weights. After training the VAE model,
the encoder can be used to generate latent vectors.

The decoder can be used to generate eyword occurrence submissions
by sampling the latent vector from a Gaussian distribution
with mean = 0 and std = 1.

[1] Kingma, Diederik P., and Max Welling.
"Auto-Encoding Variational Bayes."
https://arxiv.org/abs/1312.6114

Code based on keras example:
https://blog.keras.io/building-autoencoders-in-keras.html
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Lambda, Input, Dense
from keras.models import Model
from keras.losses import mse
from keras.utils import plot_model
from keras import backend as K
from keras import metrics
from matplotlib import colors
from matplotlib import cm

import argparse
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import sys
from scipy import stats

exp_ints = {
	'no_response' : 0,
	'absolutely_none': 1,
	'other_language' : 2,
	'know_python' : 3,
	'veteran' : 4
	}


# reparameterization trick
# instead of sampling from Q(z|X), sample epsilon = N(0,I)
# z = z_mean + sqrt(var) * epsilon
def sampling(args):
	'''
	Reparameterization trick by sampling from an isotropic unit Gaussian.
	# Arguments
		args (tensor): mean and log of variance of Q(z|X)
	# Returns
		z (tensor): sampled latent vector
	'''
	z_mean, z_log_var = args
	batch = K.shape(z_mean)[0]
	dim = K.int_shape(z_mean)[1]
	# by default, random_normal has mean = 0 and std = 1.0
	epsilon = K.random_normal(shape=(batch, dim))
	return z_mean + K.exp(0.5 * z_log_var) * epsilon


def pad(array, reference_shape, offsets):
    '''
    Pads the array with zeros to fit the reference shape at the offsets given
    [0, 0] to align with top left
    :param array: Array to be padded
    :param reference_shape: tuple of size of ndarray to create
    :param offsets: list of offsets (number of elements must be equal to the dimension of the array)
    will throw a ValueError if offsets is too big and the reference_shape cannot handle the offsets
    '''
    result = np.zeros(reference_shape)
    insertHere = [slice(offsets[dim], offsets[dim] + array.shape[dim]) for dim in range(array.ndim)]
    result[insertHere] = array
    return result


def plot_training_loss(training_history, epochs, unit, problem):
	'''
	Plots the losses during training given the History object,
	number of epochs, and keyword list. Plots total, reconstruction,
	KL divergence, and feature-specific losses.
	:param training_history: the training history returned from model.fit
	:param epochs: number of epochs to train
	:param unit: course unit
	:param problem: course problem
	'''
	l = np.array(training_history.history['loss'])
	r_l = np.array(training_history.history['mse'])
	k_l = np.array(training_history.history['kl'])
	v_l = np.array(training_history.history['val_loss'])
	plt.plot(np.linspace(1, epochs, epochs), l, 'b+')
	plt.plot(np.linspace(1, epochs, epochs), r_l, 'c.')
	plt.plot(np.linspace(1, epochs, epochs), k_l, 'g.')
	plt.plot(np.linspace(1, epochs, epochs), v_l, 'r+')
	plt.legend(['training loss', 'reconstruction loss', 'KL divergence', 'validation loss'])
	plt.xlabel('epoch')
	plt.ylabel('loss')
	plt.title('Training Loss MSE+KL for Problem {}-{}'.format(unit, problem))
	os.makedirs('loss_plots', exist_ok=True)
	plt.savefig('loss_plots/flattraj_training_loss_{}_{}'.format(unit, problem))
	plt.close()


def plot_mean_results(models,
				 x_test,
				 y_test,
				 y1_test,
				 unit,
				 problem,
				 batch_size=128):
	'''
	Plots labels and vis of samples
		as a function of the 2D latent vector
	# Arguments
		models (tuple): encoder and decoder models
		x_test: test data
		y_test: first set of labels (experience)
		y1_test: second set of labels (video engagement)
		unit (int): course unit
		problem (int): course problem
		batch_size (int): prediction batch size
	'''
	encoder, decoder = models

	os.makedirs('means', exist_ok=True)
	
	# display a 2D plot of the experience classes in the latent space
	z_mean, _, _ = encoder.predict(x_test, batch_size=batch_size)
	
	for this_y in [y_test, y1_test]:
		plt.figure(figsize=(12, 10))
		sns.scatterplot(x=z_mean[:, 0], y=z_mean[:, 1], hue=this_y)
		plt.xlabel('z[0]')
		plt.ylabel('z[1]')
		if this_y is y_test:
			filename = 'means/vae_flattraj_mean_{}_{}_ytest.png'.format(unit, problem)
		else:
			filename = 'means/vae_flattraj_mean_{}_{}_y1test.png'.format(unit, problem)
		plt.savefig(filename)
		plt.close()

	# plot a density map in the latent space to contextualize the scatterplots
	filename = 'means/vae_flattraj_mean_{}_{}_kde.png'.format(unit, problem)
	plt.figure(figsize=(12, 10))
	cmap = sns.dark_palette("purple", as_cmap=True)
	ax = sns.kdeplot(z_mean[:, 0], z_mean[:, 1], n_levels=30, cmap=cmap)
	plt.xlabel('z[0]')
	plt.ylabel('z[1]')
	plt.savefig(filename)

	# mutate all y data to integers
	y_test = np.array([exp_ints[exp] for exp in y_test])
	y1_test = np.array([0 if vid==False else 1 for vid in y1_test])

	for this_y in [y_test, y1_test]:
		if this_y is y_test:
			print('EXPERIENCE')
		else:
			print('VIDEO ENGAGEMENT')
		_, _, r2, _, _ = stats.linregress(z_mean[:, 0], this_y)
		print('r-squared from regression z0 and this y:', r2)
		_, _, r2, _, _ = stats.linregress(z_mean[:, 1], this_y)
		print('r-squared from regression z1 and this y:', r2)

		pear = stats.pearsonr(z_mean[:, 0], this_y)
		print('r-squared from pearson correlation z0 and this y:', pear)
		pear = stats.pearsonr(z_mean[:, 1], this_y)
		print('r-squared from pearson correlation z1 and this y:', pear)

	return

	
def plot_latent_sampled(models,
				 x_test,
				 y_test,
				 y1_test,
				 unit,
				 problem):
	'''
	Plots vis of samples from latent space
	as a function of the 2D latent vector
	:param models (tuple): trained encoder and decoder networks
	:param x_test: test data
	:param y_test: labels
	:param unit: course unit
	:param problem: course problem
	'''
	encoder, decoder = models

	# display a 2D manifold of submissions
	n = 10
	num_keywords = 10
	max_subs = 30
	figure = np.zeros((1 * n, num_keywords * max_subs * n))
	latent_l1 = np.zeros((n, n))
	latent_l2 = np.zeros((n, n))
	
	# linearly spaced coordinates corresponding to
	# 2D plot of digit classes in latent space
	grid_x = np.linspace(-4, 4, n)
	grid_y = np.linspace(-4, 4, n)[::-1]

	for i, yi in enumerate(grid_y):
		for j, xi in enumerate(grid_x):
			z_sample = np.array([[xi, yi]])
			x_decoded = decoder.predict(z_sample)
			digit = x_decoded[0].reshape(1, num_keywords*max_subs)
			figure[i : (i + 1), j * num_keywords*max_subs: (j + 1) * num_keywords*max_subs] = np.rint(30*digit)
			latent_l1[i:i+1, j:j+1] = np.sum(np.rint(30*digit))
			latent_l2[i:i+1, j:j+1] = np.linalg.norm(np.rint(30*digit))

	viridis = cm.get_cmap('viridis', 256)

	os.makedirs('sample_vis', exist_ok=True)
	for norm in ['l1', 'l2']:
		filename = 'sample_vis/flattraj_subs_over_latent_{}_{}_{}.png'.format(norm, unit, problem)

		fig, ax = plt.subplots()
		if norm == 'l2':
			psm = ax.pcolormesh(latent_l2, cmap=viridis, rasterized=True)
		else:
			psm = ax.pcolormesh(latent_l1, cmap=viridis, rasterized=True)
		fig.colorbar(psm, ax=ax)

		ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)
		ax.set_xticks([])
		ax.set_yticks([])

		plt.xlabel('z[0]')
		plt.ylabel('z[1]')
		if norm == 'l2':
			latent_title = 'L2 norm of submissions generated \n from sampled 2D latent space for Problem {}-{}'
		else:
			latent_title = 'L1 norm of submissions generated \n from sampled 2D latent space for Problem {}-{}'
		plt.title(latent_title.format(unit, problem), fontsize=10)
		plt.savefig(filename)
	return

# specify a save name based on the features you're using
# this will be added to output filenames of generated figures and plots
savename = 'vae_flattraj_kw_occ'

# specify problem to consider
unit = 2
pb = 3

# load keyword occurrence submissions dataset
sub_data_csv = '../../data/keyword_occurrence/trajectory/unit{}_pb{}_kw_occ_traj.csv'
sub_data = pd.read_csv(sub_data_csv.format(unit, pb), index_col=None)

sub_data['kw_occ_matrix'] = sub_data['kw_occ_matrix'].apply(lambda x:eval(x))
x_data = np.array([pad(np.array(row), (30, 10), [0, 0]) for row in sub_data['kw_occ_matrix'].to_numpy()])
# flatten trajectory matrices
original_dim = x_data[0].shape[0] * x_data[0].shape[1]
x_data = np.reshape(x_data, [-1, original_dim])

# in this case, y is experience level
exp_data = sub_data['user_exp'].to_numpy()
y_data = exp_data

# in this case, y1 is video engagement
video_data = sub_data['video_engaged'].to_numpy()
y1_data = video_data

# split train / test
train_pct = 0.7
i = math.ceil(0.7*x_data.shape[0])
x_train, x_test = x_data[0:i, :], x_data[i:, :]
y_train, y_test = y_data[0:i], y_data[i:]
y1_train, y1_test = y1_data[0:i], y1_data[i:]

# normalize
largest = x_data.max()
print('largest', largest)
x_train, x_test = x_train.astype('float32') / largest, x_test.astype('float32') / largest

print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)
print(y1_train.shape, y1_test.shape)

# Keyword occurrence network params
input_shape = (original_dim, )
batch_size = 50
intermediate_dim = 100
latent_dim = 2
epochs = 75

# VAE model = encoder + decoder
# build encoder model
inputs = Input(shape=input_shape, name='encoder_input')
print('input matrix is ', inputs)

x = Dense(intermediate_dim, activation='relu')(inputs)
z_mean = Dense(latent_dim, name='z_mean')(x)
z_log_var = Dense(latent_dim, name='z_log_var')(x)

# use reparameterization trick to push the sampling out as input
# note "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

# instantiate encoder
encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
encoder.summary()
plot_model(encoder, to_file='{}_encoder.png'.format(savename), show_shapes=True)

# build decoder
latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
x = Dense(intermediate_dim, activation='relu')(latent_inputs)
outputs = Dense(original_dim, activation='sigmoid')(x)

# instantiate decoder
decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()
plot_model(decoder, to_file='{}_decoder.png'.format(savename), show_shapes=True)

# instantiate VAE
outputs = decoder(encoder(inputs)[2])
vae = Model(inputs, outputs, name='vae_flattraj')

if __name__ == '__main__':
	
	print('inputs \n', inputs)
	print('input shape \n', inputs.shape)

	parser = argparse.ArgumentParser()
	
	help_ = "Load h5 model trained weights"
	parser.add_argument("-w", "--weights", help=help_)
	
	args = parser.parse_args()
	
	models = (encoder, decoder)

	def recon_loss(inp, out, in_dim):
		reconstruction_loss = mse(inp, out)
		reconstruction_loss *= in_dim
		return reconstruction_loss

	def kl_div_loss(mean, log_var):
		kl_loss = 1 + log_var - K.square(mean) - K.exp(log_var)
		kl_loss = K.sum(kl_loss, axis=-1)
		kl_loss *= -0.5
		return kl_loss
	
	reconstruction_loss = recon_loss(inputs, outputs, original_dim)
	kl_loss = kl_div_loss(z_mean, z_log_var)
	vae_loss = K.mean(reconstruction_loss + kl_loss)
	vae.add_loss(vae_loss)

	vae.compile(loss=None, optimizer='adam')
	# https://github.com/keras-team/keras/issues/9526
	vae.summary()
	plot_model(vae, to_file='{}.png'.format(savename), show_shapes=True)
	
	# add custom metrics
	# https://github.com/keras-team/keras/issues/9459
	vae.metrics_tensors.append(K.mean(reconstruction_loss))
	vae.metrics_names.append("mse")

	vae.metrics_tensors.append(K.mean(kl_loss))
	vae.metrics_names.append("kl")

	if args.weights:
		vae.load_weights(args.weights)
	else:
		# train autoencoder
		train_h = vae.fit(x_train,
				epochs=epochs,
				batch_size=batch_size,
				validation_data=(x_test, None))
		
		# inspect training losses
		plot_training_loss(train_h, epochs, unit, pb)
		
		# save trained weights
		os.makedirs('weights', exist_ok=True)
		w_filename = 'weights/{}_{}_{}.h5'.format(savename, unit, pb)
		vae.save_weights(w_filename)

	# show means of test set in latent space
	plot_mean_results(models,
				 x_test,
				 y_test,
				 y1_test,
				 unit,
				 pb,
				 batch_size=batch_size)

	# visualize generated vectors from sampling the latent space
	plot_latent_sampled(models, x_test, y_test, y1_test, unit, pb)


