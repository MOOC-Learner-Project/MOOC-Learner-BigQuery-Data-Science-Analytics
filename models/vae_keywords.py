'''Using VAE on keyword occurrence submissions dataset using MLP

The VAE has a modular design. The encoder, decoder and VAE
are 3 models that share weights. After training the VAE model,
the encoder can be used to generate latent vectors.

The decoder can be used to generate keyword occurrence submissions
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
import sys

keyword_indices = {'if' : 0, 'elif' : 1, 'else' : 2,
				'for' : 3, 'while' : 4,
				'break' : 5, 'continue' : 6,
				'def' : 7, 'return' : 8,
				'print' : 9}
keyword_list = ['if', 'elif', 'else', 'for', 'while', 'break', 'continue', 'def', 'return', 'print']

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


def plot_training_loss(training_history, epochs, kw_list, unit, problem):
	'''
	Plots the losses during training given the History object,
	number of epochs, and keyword list. Plots total, reconstruction,
	KL divergence, and feature-specific losses.
	:param training_history: the training history returned from model.fit
	:param epochs: number of epochs to train
	:param kw_list: Python list of keywords
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
	if final_only:
		plt.savefig('loss_plots/training_loss_{}_{}_final'.format(unit, problem))
	else:
		plt.savefig('loss_plots/training_loss_{}_{}'.format(unit, problem))
	plt.close()

	per_ep = np.linspace(1, epochs, epochs)
	for kw in kw_list:
		kw_hist = np.array(training_history.history['{}_loss'.format(kw)])
		plt.plot(per_ep, kw_hist, '-')
	plt.legend(kw_list)
	plt.xlabel('epoch')
	plt.ylabel('loss')
	plt.title('Component MSE Loss during Training for Problem {}-{}'.format(unit, problem))
	if final_only:
		plt.savefig('loss_plots/training_comp_loss_{}_{}_final'.format(unit, problem))
	else:
		plt.savefig('loss_plots/training_comp_loss_{}_{}'.format(unit, problem))
	plt.close()


def keyword_loss(keyword, inp, out):
	'''
	Returns a function call to the keyword-specific
	loss function given the inputs and outputs,
	giving the mse for that keyword feature.
	:param keyword: Python keyword for loss calculation
	:param inp: actual input
	:param out: reconstructed output
	'''
	def this_keyword_loss(inp, out):
		i = keyword_indices[keyword]
		return mse(inp[ :, i:i+1], out[ :, i:i+1])
		# each row is a datapoint, each col is a keyword feature
	return this_keyword_loss(inp, out)


def plot_mean_results(models,
				 x_test,
				 y_test,
				 unit,
				 problem,
				 batch_size=128):
	'''
	Plots labels and vis of samples
		as a function of the 2D latent vector
	# Arguments
		models (tuple): encoder and decoder models
		x_test: test data
		y_test: labels
		unit (int): course unit
		problem (int): course problem
		batch_size (int): prediction batch size
	'''
	encoder, decoder = models

	if final_only:
		os.makedirs('means_final', exist_ok=True)
		filename = 'means_final/vae_mean_{}_{}_final.png'.format(unit, problem)
	else:
		os.makedirs('means', exist_ok=True)
		filename = 'means/vae_mean_{}_{}.png'.format(unit, problem)
	
	# display a 2D plot of the experience classes in the latent space
	#z_mean, _, _ = encoder.predict(x_test, batch_size=batch_size)
	z_mean, _, _ = encoder.predict(x_test, batch_size=batch_size)
	plt.figure(figsize=(12, 10))
	plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y_test)
	plt.colorbar()
	plt.xlabel("z[0]")
	plt.ylabel("z[1]")
	plt.savefig(filename)
	plt.close()


	df_with_latents = pd.DataFrame(columns=['x_test', 'y_test', 'z_mean_0', 'z_mean_1'])

	n = x_test.shape[0]
	for i in range(n):
		df_with_latents = df_with_latents.append({'x_test' : x_test[i, :],
												'y_test' : y_test[i],
												'z_mean_0' : z_mean[i, 0],
												'z_mean_1' : z_mean[i, 1]},
												ignore_index=True)
	os.makedirs('significance', exist_ok=True)
	csv_filename = 'significance/vae_sig_{}_{}.csv'.format(unit, problem)
	if final_only:
		csv_filename = 'significance/vae_sig_{}_{}_final.csv'.format(unit, problem)
	df_with_latents.to_csv(csv_filename, index=False)

	# get standard deviations and means of all vars
	y_m, y_std = df_with_latents['y_test'].mean(), df_with_latents.loc[: ,'y_test'].std()
	z0_m, z0_std = df_with_latents['z_mean_0'].mean(), df_with_latents.loc[: ,'z_mean_0'].std()
	z1_m, z1_std = df_with_latents['z_mean_1'].mean(), df_with_latents.loc[: ,'z_mean_1'].std()
	coef = 1.0/(n-1)
	denom_z0 = y_std * z0_std
	denom_z1 = y_std * z1_std
	
	rsqu_z0 = (coef/denom_z0 * np.sum([(y_test[i]-y_m)*(z_mean[i, 0]-z0_m) for i in range(n)]))**2
	rsqu_z1 = (coef/denom_z1 * np.sum([(y_test[i]-y_m)*(z_mean[i, 1]-z0_m) for i in range(n)]))**2

	print('r-squared value z0 and y:', rsqu_z0)
	print('r-squared value z1 and y:', rsqu_z1)
	
def plot_latent_sampled(models,
				 x_test,
				 y_test,
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
	figure = np.zeros((1 * n, num_keywords * n))
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
			digit = x_decoded[0].reshape(1, num_keywords)
			figure[i : (i + 1), j * num_keywords: (j + 1) * num_keywords] = np.rint(30*digit)
			latent_l1[i:i+1, j:j+1] = np.sum(np.rint(30*digit))
			latent_l2[i:i+1, j:j+1] = np.linalg.norm(np.rint(30*digit))

	viridis = cm.get_cmap('viridis', 256)

	os.makedirs('sample_vis', exist_ok=True)
	for norm in ['l1', 'l2']:
		if final_only:
			filename = 'sample_vis/subs_over_latent_{}_{}_{}_final.png'.format(norm, unit, problem)
		else:
			filename = 'sample_vis/subs_over_latent_{}_{}_{}.png'.format(norm, unit, problem)

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


# specify a save name based on the features you're using
# this will be added to output filenames of generated figures and plots
savename = 'vae_mlp_kw_occ'

# specify problem to consider
unit = 1
pb = 1

# specify whether to consider the final problem only
final_only = False

# load keyword occurrence submissions dataset, OR replace with your own features
sub_data_csv = '../../data/keyword_occurrence/single/unit{}_pb{}_kw_occ_single.csv'
sub_data = pd.read_csv(sub_data_csv.format(unit, pb), index_col=None)

# if final_only bool set, consider final submission data only
# you may have to change the logic and headers depending on your data
if final_only:
	sub_data = sub_data.loc[sub_data['total_submissions'] == sub_data['this_submission']]

sub_data['stripped_sub_kw_occ'] = sub_data['stripped_sub_kw_occ'].apply(lambda x:eval(x))
x_data = np.array([row for row in sub_data['stripped_sub_kw_occ'].to_numpy()])

exp_data = sub_data['user_exp'].to_numpy()
y_data = np.array([exp_ints[exp] for exp in exp_data])

# split train / test
train_pct = 0.7
i = math.ceil(0.7*x_data.shape[0])
x_train, x_test = x_data[0:i, :], x_data[i:, :]
y_train, y_test = y_data[0:i], y_data[i:]

# normalize
largest = x_data.max()
print('largest', largest)
x_train, x_test = x_train.astype('float32') / largest, x_test.astype('float32') / largest

# sanity check that matrices look reasonable
print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)

input_dim = len(x_train[0])


# Keyword occurrence network params
# You may have to change for your features!
input_shape = (input_dim, )
intermediate_dim = 5
batch_size = 50
latent_dim = 2
epochs = 25

# VAE model = encoder + decoder
# build encoder model
inputs = Input(shape=input_shape, name='encoder_input')
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
outputs = Dense(input_dim, activation='sigmoid')(x)

# instantiate decoder
decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()
plot_model(decoder, to_file='{}_decoder.png'.format(savename), show_shapes=True)

# instantiate VAE
outputs = decoder(encoder(inputs)[2])
vae = Model(inputs, outputs, name='vae_mlp')

if __name__ == '__main__':
	
	print('inputs \n', inputs)
	print('input shape \n', inputs.shape)

	parser = argparse.ArgumentParser()
	
	# can used saved weights rather than retraining
	# pass as optional command line arguments
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
	
	reconstruction_loss = recon_loss(inputs, outputs, input_dim)
	kl_loss = kl_div_loss(z_mean, z_log_var)
	vae_loss = K.mean(reconstruction_loss + kl_loss)
	vae.add_loss(vae_loss)

	vae.compile(loss=None, optimizer='adam')
	# https://github.com/keras-team/keras/issues/9526
	vae.summary()
	plot_model(vae, to_file='{}.png'.format(savename), show_shapes=True)
	
	# add custom metrics
	# https://github.com/keras-team/keras/issues/9459
	
	# the losses are specified per keyword, for 10 keywords
	# comment this loop if not using keywords
	for kw in keyword_indices.keys():
		vae.metrics_tensors.append(K.mean(keyword_loss(kw, inputs, outputs)))
		vae.metrics_names.append('{}_loss'.format(kw))

	# should be generic
	vae.metrics_tensors.append(K.mean(reconstruction_loss))
	vae.metrics_names.append("mse")

	# should be generic
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
		plot_training_loss(train_h, epochs, keyword_list, unit, pb)
		
		# save trained weights
		os.makedirs('weights', exist_ok=True)
		if final_only:
			w_filename = 'weights/{}_{}_{}_final.h5'.format(savename, unit, pb)
		else:
			w_filename = 'weights/{}_{}_{}.h5'.format(savename, unit, pb)
		vae.save_weights(w_filename)

	# show means of test set in latent space
	plot_mean_results(models,
				 x_test,
				 y_test,
				 unit,
				 pb,
				 batch_size=batch_size)

	# visualize generated vectors from sampling the latent space
	plot_latent_sampled(models,
				 x_test,
				 y_test,
				 unit,
				 pb)

