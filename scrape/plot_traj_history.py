import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns; sns.set()
import warnings; warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

from matplotlib.colors import ListedColormap
from matplotlib.ticker import MaxNLocator
from pylab import *
from sklearn.metrics.pairwise import euclidean_distances

import keywords
rcParams['axes.titlesize'] = 9
rcParams['axes.titlepad'] = 2

# change if needed
limit_plots = True
limit_num = 50 # plot for this many users when limit_plots is True
verbose = True
similarity_threshold = 0.5
similarity_thresholds_list = np.linspace(0, 1, 55)

# specify year and problem number
unit = int(sys.argv[1])
pb = int(sys.argv[2])

u_pb = '{}-{}'.format(unit, pb) # in format 1-1, 5-3, etc

# set substrings to count/track through user submission history
substrs = keywords.python_keywords
keyword_groups = keywords.python_keyword_colors

# load user histories
kw_occ_traj_path = '../../data/keyword_occurrence/trajectory/unit{}_pb{}_kw_occ_traj.csv'
kw_occ_traj_df = pd.read_csv(kw_occ_traj_path.format(u_pb[0], u_pb[2]), index_col=None)

# load sample solutions
sample_sols = pd.read_csv('sample_sols.csv', index_col=None)

# where to save plots
save_dir = 'submission_history_plots' # saves in this_dir/problem_subfolder
os.makedirs('{}/{}-{}'.format(save_dir, unit, pb), exist_ok=True)


def main():
	all_u_kw_occ_comments = pd.Series(kw_occ_traj_df.kw_occ_matrix.values,
		index=kw_occ_traj_df.user_id).to_dict()
	all_u_kw_occ_stripped = pd.Series(kw_occ_traj_df.stripped_kw_occ_matrix.values,
		index=kw_occ_traj_df.user_id).to_dict()
	all_u_exp = pd.Series(kw_occ_traj_df.user_exp.values,
		index=kw_occ_traj_df.user_id).to_dict()
	all_final_correct_bool = pd.Series(kw_occ_traj_df.final_correct_bool.values,
		index=kw_occ_traj_df.user_id).to_dict()

	plot_count = 0
	make_plots = False

	similar_correct = 0
	different_correct = 0
	similar_incorrect = 0
	different_incorrect = 0
	exp_dict_counts = {
		'absolutely_none' : [0, 0, 0, 0],
		'other_language' : [0, 0, 0, 0],
		'know_python' : [0, 0, 0, 0],
		'veteran': [0, 0, 0, 0],
		'no_response' : [0, 0, 0, 0]
		}

	for u_id in kw_occ_traj_df['user_id'].to_list():

		if limit_plots and plot_count == limit_num:
			make_plots = False

		# heatmap using comments stripped counts
		stripped_counts = eval(all_u_kw_occ_stripped[u_id])
		cols_as_subs = np.array(stripped_counts).T
		sub_name_cols = ['Sub {}'.format(i+1) for i in range(len(cols_as_subs[0]))]
		df_to_corr = pd.DataFrame(cols_as_subs, columns=sub_name_cols)
		sample = sample_sols.loc[sample_sols['problem']==u_pb, 'solution'].iloc[0] # extract
		df_to_corr['Sample'] = subs_substr_count([sample], substrs)[0]
		
		dist = df_to_corr.corr() # used for plotting
		dist_matrix = dist.to_numpy() # used for similarity count
		n = len(dist_matrix)
		sample_similarity = dist_matrix[n-1][n-2]

		exp_level = all_u_exp[u_id]
		# compare correctness, similarity to sample using sim threshold
		if all_final_correct_bool[u_id]:
			if sample_similarity >= similarity_threshold:
				similar_correct += 1
				exp_dict_counts[exp_level][0] += 1
			else:
				different_correct += 1
				exp_dict_counts[exp_level][1] += 1
		else:
			if sample_similarity >= 0.5:
				similar_incorrect += 1
				exp_dict_counts[exp_level][2] += 1
			else:
				different_incorrect += 1
				exp_dict_counts[exp_level][3] += 1

		
		if make_plots:
			# over time, including comments
			f1 = plt.figure()
			counts = eval(all_u_kw_occ_comments[u_id])
			subs_df = pd.DataFrame(columns=(['Submission']+substrs), data=add_idx(counts))
			ax = subs_df.set_index('Submission').plot(kind='bar', stacked=True,
				colormap=ListedColormap(sns.xkcd_palette(keyword_groups)), figsize=(8,6))
			ax.yaxis.set_major_locator(MaxNLocator(integer=True))
			plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
			f1_title = 'Keyword occurrences in submission history \n of user {} on Problem {}'
			plt.suptitle(f1_title.format(u_id, u_pb))
			plt.savefig('{}/{}/{}_comments.png'.format(save_dir, u_pb, u_id), bbox_inches='tight')

			# heatmaps
			f2 = plt.figure()
			mask = np.zeros_like(dist)
			mask[np.triu_indices_from(mask)] = True # show only below diagonal
			sns.heatmap(dist, vmin=0.0, vmax=1.0, mask=mask, square=True, cmap='coolwarm', linewidths=.5)
			plt.yticks(rotation=0)
			plt.xticks(rotation=90)
			f2_title = 'Code submission history correlations \n of user {} on Problem {}'
			plt.suptitle(f2_title.format(u_id, u_pb))
			plt.savefig('{}/{}/{}_heatmap.png'.format(save_dir, u_pb, u_id), bbox_inches='tight')
		
			# over time, comments stripped
			f3 = plt.figure()
			stripped_subs_df = pd.DataFrame(columns=(['Submission']+substrs), data=add_idx(stripped_counts))
			ax = stripped_subs_df.set_index('Submission').plot(kind='bar', stacked=True,
				colormap=ListedColormap(sns.xkcd_palette(keyword_groups)), figsize=(8,6))
			ax.yaxis.set_major_locator(MaxNLocator(integer=True))
			plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
			f3_title = 'Term occurrences in comment-stripped submission history \n of user {} on Problem {}'
			plt.suptitle(f3_title.format(u_id, u_pb))
			plt.xlabel('Submission on Problem {}'.format(u_pb), fontsize=18)
			plt.savefig('{}/{}/{}_stripped.png'.format(save_dir, u_pb, u_id), bbox_inches='tight')

		plt.close('all')
		plot_count += 1

	if verbose:

		total_correct = float(similar_correct + different_correct)
		total_incorrect = float(similar_incorrect + different_incorrect)

		print('Total correct students: {}'.format(total_correct))
		print('Total incorrect students: {} \n'.format(total_incorrect))

		print('Correct students with similar solutions: {}'.format(similar_correct))
		print('Percentage of correct: {} \n'.format(similar_correct/total_correct))

		print('Correct students with very different solutions: {}'.format(different_correct))
		print('Percentage of correct: {} \n'.format(different_correct/total_correct))

		print('Incorrect students with similar solutions: {}'.format(similar_incorrect))
		print('Percentage of incorrect: {} \n'.format(similar_incorrect/total_incorrect))

		print('Incorrect students with very different solutions: {}'.format(different_incorrect))
		print('Percentage of incorrect: {} \n'.format(different_incorrect/total_incorrect))

	do_sim_cor_results(similar_correct,
		different_correct,
		similar_incorrect,
		different_incorrect,
		unit, pb, similarity_threshold)

	do_sim_cor_by_exp(exp_dict_counts, unit, pb, similarity_threshold)



def subs_substr_count(submission_list, substr_list):
	'''
	Counts desired substrings in each submission string.
	Both strings and substrings given in lists.
	:param submission_list: python list of code submissions, as strings
	:param substr_list: specifies strings for which to count occurrences in code
	'''
	sub_substrs_matrix = []
	for sub in submission_list:
		sub_substrs = [sub.count(substr) for substr in substr_list]
		sub_substrs_matrix.append(sub_substrs)
	return sub_substrs_matrix



def add_idx(cts):
	'''
	Prepends the 1-indexed index to each row in the 2D array cts
	:param cts: the 2D array representing the substring counts
	(mutator function - beware when using)
	'''
	for i in range(len(cts)):
		cts[i] = [i+1] + cts[i]
	return cts



def remove_comments_str(astr):
	'''
	Removes comment lines and extra newlines from astr, returns astr.
	:param astr: input string to process
	'''
	astr = re.sub(r'(#+)(.*)\r\n', '\r\n', astr) # single line comments - keep return
	astr = re.sub(r'\"\"\"[\s\S]*?\"\"\"', '', astr) # multiline comments
	astr = re.sub(r'(\r\n)+[\s]*\r\n', '\r\n', astr) # extra newlines
	return astr



def do_sim_cor_results(sim_cor, diff_corr, sim_in, diff_in, unit, problem, sim_th):
	'''
	Prints out numbers of students by similarity and correctness.
	Also saves results to a quadrant heatmap plot.
	:param :
	:param :
	:param :
	:param :
	'''
	sim_cor_counts = {'correctness': ['correct', 'correct', 'incorrect', 'incorrect'],
			'similarity':['similar', 'different', 'similar', 'different'],
			'students': [sim_cor, diff_corr, sim_in, diff_in]}
	sim_cor_counts_df = pd.DataFrame(sim_cor_counts).pivot('correctness', 'similarity', 'students')
	cmap = sns.cubehelix_palette(light=0.95, as_cmap=True)
	ax = sns.heatmap(sim_cor_counts_df, cmap=cmap, annot=True, fmt='d')
	quad_plot_title = 'Student final submission counts \n by correctness-similarity for Problem {}-{} \n thresholded at {} similarity to sample'
	plt.title(quad_plot_title.format(unit, problem, sim_th))
	os.makedirs('quad_plots', exist_ok=True)
	plt.savefig('quad_plots/sim_{}_cor_{}_{}'.format(int(sim_th*100), unit, pb))



def do_sim_cor_by_exp(experience_counts, unit, problem, sim_th):
	'''
	'''
	fig,axn = plt.subplots(2, 2, sharex=True, sharey=True)
	cbar_ax = fig.add_axes([.91, .3, .03, .4])

	exp_is = {0:'absolutely_none', 1:'other_language', 2:'know_python', 3:'veteran'}

	for i, ax in enumerate(axn.flat):
		sim_cor_counts = {'correctness': ['correct', 'correct', 'incorrect', 'incorrect'],
			'similarity':['similar', 'different', 'similar', 'different'],
			'students': experience_counts[exp_is[i]]}
		exp_sim_cor_counts_df = pd.DataFrame(sim_cor_counts).pivot('correctness', 'similarity', 'students')
		exp_np = exp_sim_cor_counts_df.to_numpy()
		cmap = sns.cubehelix_palette(light=0.95, as_cmap=True)
		sns.heatmap(exp_sim_cor_counts_df, cmap=cmap, annot=exp_np/exp_np.sum(), fmt='.1%',
			ax=ax,
			cbar=i == 0,
			vmin = min(min(experience_counts.values())),
			vmax = max(max(experience_counts.values())),
			cbar_ax=None if i else cbar_ax,
			cbar_kws=dict(ticks=None))
		ax.title.set_text(exp_is[i])

	exp_plot_title = 'Correctness-similarity by experience for Problem {}-{}, thresholded at {} similarity to sample'
	plt.suptitle(exp_plot_title.format(unit, problem, sim_th), fontsize=10, y=0.99)

	fig.tight_layout(rect=[0, 0, .9, 1])
	plt.title('students', fontsize=9)

	os.makedirs('quad_exp_plots', exist_ok=True)
	plt.savefig('quad_exp_plots/exp_sim_{}_cor_{}_{}'.format(int(sim_th*100), unit, pb))



def similarity_sweep(kw_occ_traj_df, sample_sols_df, sim_th_list, unit, problem):
	'''
	'''
	all_u_kw_occ_stripped = pd.Series(kw_occ_traj_df.stripped_kw_occ_matrix.values,
		index=kw_occ_traj_df.user_id).to_dict()
	all_final_correct_bool = pd.Series(kw_occ_traj_df.final_correct_bool.values,
		index=kw_occ_traj_df.user_id).to_dict()

	sim, diff = [], []

	for th in sim_th_list:
		sim_cor, diff_cor = 0, 0
		for u_id in kw_occ_traj_df['user_id'].to_list():

			stripped_counts = eval(all_u_kw_occ_stripped[u_id])
			cols_as_subs = np.array(stripped_counts).T
			sub_name_cols = ['Sub {}'.format(i+1) for i in range(len(cols_as_subs[0]))]
			df_to_corr = pd.DataFrame(cols_as_subs, columns=sub_name_cols)
			sample = sample_sols_df.loc[sample_sols['problem']==u_pb, 'solution'].iloc[0] # extract
			df_to_corr['Sample'] = subs_substr_count([sample], substrs)[0]
			
			dist_matrix = df_to_corr.corr().to_numpy() # used for similarity count
			n = len(dist_matrix)
			sample_similarity = dist_matrix[n-1][n-2]

			# compare correctness, similarity to sample using sim threshold
			if all_final_correct_bool[u_id]:
				if sample_similarity >= th:
					sim_cor += 1
				else:
					diff_cor += 1
			# else was incorrect
		sim.append(sim_cor)
		diff.append(diff_cor)
	total=sim_cor+diff_cor

	plt.plot(sim_th_list, np.array(sim)/total, 'cX')
	plt.plot(sim_th_list, np.array(diff)/total, 'rX')
	plt.legend(['pct similar', 'pct different'])
	plt.xlabel('threshold similarity')
	plt.ylabel('percentage')
	plt.suptitle('Sensitivity of similarity threshold for Problem {}-{}'.format(unit, problem), fontsize=14)

	os.makedirs('sensitivity', exist_ok=True)
	plt.savefig('sensitivity/sim_sens_{}_{}'.format(unit, pb))
	return



def plot_samples_bar_plot(sample_sols_all, keywords):
	'''
	Plots all the sample solution (gold standards)
	on a single stacked bar plot for comparison.
	:param sample_sols_all: pandas df of gold standard problems and solutions
	:param keywords: specifies words for which to count occurrences in code
	'''
	samples_problem_list = sample_sols_all['problem'].to_list()
	samples_solution_list = sample_sols_all['solution'].to_list()
	samples_counts = subs_substr_count(samples_solution_list, substrs)
	samples_df = pd.DataFrame(columns=substrs, data=samples_counts)
	
	samples_df['Problem'] = samples_problem_list
	print(samples_df)
	ax1 = samples_df.set_index('Problem').plot(kind='bar', stacked=True,
		colormap=ListedColormap(sns.xkcd_palette(keyword_groups)), figsize=(9,6))
	ax1.yaxis.set_major_locator(MaxNLocator(integer=True))
	locs, labels = plt.xticks()
	plt.setp(labels, rotation=0)
	plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
	plt.title('Keyword occurrences in sample solutions for 6.00.1x', fontsize=18)
	plt.xlabel('Problem', fontsize=18)
	plt.savefig('{}/sample_solution_barplot.png'.format(save_dir), bbox_inches='tight')



if __name__ == "__main__":
	main()
	#similarity_sweep(kw_occ_traj_df, sample_sols, similarity_thresholds_list, unit, pb)
	#plot_samples_bar_plot(sample_sols, substrs)


