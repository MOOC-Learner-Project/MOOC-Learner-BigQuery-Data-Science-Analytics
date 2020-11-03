import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pylab import *
import re
from sklearn.metrics.pairwise import euclidean_distances

import keywords
import problem_params

pb_video_id_dict = {
	'1-1' : 'ID',
}

# specify problem number
unit = int(sys.argv[1])
pb = int(sys.argv[2])

u_pb = '{}-{}'.format(unit, pb) # in format 1-1, 5-3, etc
selected = u_pb in ['1-1', '2-3', '4-7']
problem_id = problem_params.problem_id_dict['{}-{}'.format(unit, pb)]

# set substrings to count/track through user submission history
substrs = keywords.python_keywords

# load user histories for each year/offering
yr_dfs = []
for yr in ['2016',]:
	yr_sub_histories = pd.read_csv('../../data/csv/{}_unit{}_pb{}_history.csv'.format(yr, unit, pb), index_col=None)
	yr_sub_histories['year'] = yr
	yr_dfs.append(yr_sub_histories)

sub_histories = pd.concat(yr_dfs)

if selected:
	vid_stats_day = pd.read_csv('../../data/video_stats_day_id_1-1_2-3_4-7.csv')
	vid_id = pb_video_id_dict['{}-{}'.format(unit, pb)]

# where to save feature files
kw_occ_traj_path = '../../data/keyword_occurrence/trajectory/unit{}_pb{}_kw_occ_traj.csv'.format(unit, pb)
kw_occ_single_path = '../../data/keyword_occurrence/single/unit{}_pb{}_kw_occ_single.csv'.format(unit, pb)


def main():
	
	all_u_subs = pd.Series(sub_histories.unique_submission_history.values,
		index=sub_histories.user_id).to_dict()
	all_u_yrs = pd.Series(sub_histories.year.values,
		index=sub_histories.user_id).to_dict()
	all_u_exp = pd.Series(sub_histories.user_exp.values,
		index=sub_histories.user_id).to_dict()
	all_u_final_bool = pd.Series(sub_histories.final_correct_bool.values,
		index=sub_histories.user_id).to_dict()

	# create list of dicts for plain kw occ features, comment-stripped kw occ features
	kw_occ_traj = []
	kw_occ_single = []

	for u_id in all_u_subs.keys():

		# first, check video engagement for user
		if selected:
			vid_eng = any((vid_stats_day['user_id']==u_id) & (vid_stats_day['video_id']==vid_id))
		
		# get submissions as Python objects
		subs = eval(all_u_subs[u_id])
		u_num_subs = len(subs)
	
		# including comments
		counts = subs_substr_count(subs[::-1], substrs)

		# comment stripping
		stripped_subs = [remove_comments_str(sub) for sub in subs]
		stripped_counts = subs_substr_count(stripped_subs[::-1], substrs)

		# append dict to list
		if selected:
			kw_occ_traj.append({
				'user_id' : u_id,
				'user_exp' : all_u_exp[u_id],
				'video_engaged' : vid_eng,
				'total_submissions' : u_num_subs,
				'final_correct_bool' : all_u_final_bool[u_id],
				'kw_occ_matrix' : counts,
				'stripped_kw_occ_matrix' : stripped_counts,
				'year' : all_u_yrs[u_id]
				})
		else:
			kw_occ_traj.append({
				'user_id' : u_id,
				'user_exp' : all_u_exp[u_id],
				'total_submissions' : u_num_subs,
				'final_correct_bool' : all_u_final_bool[u_id],
				'kw_occ_matrix' : counts,
				'stripped_kw_occ_matrix' : stripped_counts,
				'year' : all_u_yrs[u_id]
				})

		# do the same for each comment-stripped individual submission
		if selected:
			for i in range(len(stripped_counts)):
				kw_occ_single.append({
					'user_id' : u_id,
					'user_exp' : all_u_exp[u_id],
					'video_engaged' : vid_eng,
					'total_submissions' : u_num_subs,
					'this_submission' : i+1,
					'stripped_sub_kw_occ' : stripped_counts[i],
					'year' : all_u_yrs[u_id]
					})
		else:
			for i in range(len(stripped_counts)):
					kw_occ_single.append({
						'user_id' : u_id,
						'user_exp' : all_u_exp[u_id],
						'total_submissions' : u_num_subs,
						'this_submission' : i+1,
						'stripped_sub_kw_occ' : stripped_counts[i],
						'year' : all_u_yrs[u_id]
						})

	# create pandas df to write to csv
	if selected:
		kw_occ_traj_features = pd.DataFrame(kw_occ_traj,
			columns=['user_id',
			'user_exp',
			'video_engaged',
			'problem_id',
			'total_submissions',
			'final_correct_bool',
			'kw_occ_matrix',
			'stripped_kw_occ_matrix',
			'year'])
	else:
		kw_occ_traj_features = pd.DataFrame(kw_occ_traj,
			columns=['user_id',
			'user_exp',
			'problem_id',
			'total_submissions',
			'final_correct_bool',
			'kw_occ_matrix',
			'stripped_kw_occ_matrix',
			'year'])
	kw_occ_traj_features['problem_id'] = problem_id
	kw_occ_traj_features.to_csv(kw_occ_traj_path, index=False)

	if selected:
		kw_occ_single_features = pd.DataFrame(kw_occ_single,
			columns=['user_id',
			'user_exp',
			'video_engaged',
			'problem_id',
			'total_submissions',
			'this_submission',
			'stripped_sub_kw_occ',
			'year'])
	else:
		kw_occ_single_features = pd.DataFrame(kw_occ_single,
			columns=['user_id',
			'user_exp',
			'problem_id',
			'total_submissions',
			'this_submission',
			'stripped_sub_kw_occ',
			'year'])
	kw_occ_single_features['problem_id'] = problem_id
	kw_occ_single_features.to_csv(kw_occ_single_path, index=False)


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


def remove_comments_str(astr):
	'''
	Removes comment lines and extra newlines from astr, returns astr.
	:param astr: input string to process
	'''
	astr = re.sub(r'(#+)(.*)\r\n', '\r\n', astr) # single line comments - keep return
	astr = re.sub(r'\"\"\"[\s\S]*?\"\"\"', '', astr) # multiline comments
	astr = re.sub(r'(\r\n)+[\s]*\r\n', '\r\n', astr) # extra newlines
	return astr


if __name__ == "__main__":
	main()


