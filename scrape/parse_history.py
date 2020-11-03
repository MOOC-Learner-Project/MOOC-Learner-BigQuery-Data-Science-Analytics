from bs4 import BeautifulSoup
from datetime import datetime
import functools
import pandas as pd
import pickle
import re
import sys

import parsing_anomalies
import problem_params

# change if needed
verbose = False
use_datetime_format = False

# specify problem and year
yr = sys.argv[1]
unit = int(sys.argv[2])
pb = int(sys.argv[3])

# get problem id and users with known anomalies for this problem
problem_id = problem_params.problem_id_dict['{}-{}'.format(unit, pb)]
anomaly_ids = parsing_anomalies.anomalies_dict['{}-{}'.format(unit, pb)]

# specify plain words to replace with Python equivalents
python_replacements = ('null', 'None'), ('true', 'True'), ('false', 'False')

id_exp = pd.read_csv('../../data/id_exp.csv', index_col=None)
filename = '../../data/pickle/{}_unit{}_pb{}_history.pkl'.format(yr, unit, pb)
out_csv = '../../data/csv/{}_unit{}_pb{}_history.csv'.format(yr, unit, pb)


def main():
	# import from scraped pickle file
	infile = open('../../data/pickle/{}'.format(filename),'rb')
	all_submissions = pickle.load(infile)[1] # take submissions dict out of (i, dict) tuple
	infile.close()

	clean_histories = []
	total_users_without_subs = 0

	# for each user, grab the contents of every <pre></pre> block
	# and make list of student_answer from each block's attributes dict
	for user_id in all_submissions.keys():

		if verbose:
			print('Trying for user with ID: ', user_id)
		
		if user_id in anomaly_ids:
			if verbose:
				print('Known anomaly. Skipping user with ID: ', user_id)
			total_users_without_subs += 1
			continue

		# make soup and separate by divs
		u_soup = BeautifulSoup(all_submissions[user_id], 'lxml')
		u_div_blocks = u_soup.findAll('div')

		top_dict = eval(convert_str_python(u_div_blocks[0].find('pre').get_text(), python_replacements))
		if 'attempts' not in top_dict.keys():
			if verbose:
				print('not attempted', '\n')
			total_users_without_subs += 1
			continue

		# initialize for each user
		num_u_attempts = 0
		resubmitted_same_code = False
		u_submissions = []
		u_correctness = []
		u_timestamps = [] # timestamp should ALWAYS be present
		u_scores = [] # score should always be present

		# process each submission
		for u_div_block in u_div_blocks:

			u_top_blocks = u_div_block.text.split('\n', 3)

			# pull out timestamp
			#u_str_time = u_top_blocks[1].replace('#%d', '')
			u_str_time = re.sub('#\d*:', '', u_top_blocks[1])
			u_time = u_str_time.strip()
			if use_datetime_format:
				u_time = datetime.strptime(u_str_time.strip(), '%Y-%m-%d %H:%M:%S %Z')

			# pull out score
			u_str_score = u_top_blocks[2].replace('Score:', '')
			u_score = eval_str_if_no_none(u_str_score)

			# pull out submission
			u_sub_block = u_div_block.find('pre')
			u_dict = eval(convert_str_python(u_sub_block.get_text(), python_replacements))
			
			if 'student_answers' in u_dict.keys():
				# weird syntax to deal with record structure
				current_sub = list(u_dict['student_answers'].values())[0] # only 1 value
			if 'attempts' in u_dict.keys():
				tmp = int(u_dict['attempts'])
			if 'correct_map' in u_dict.keys():
				if u_dict['correct_map']:
					# weird syntax to deal with record structure
					u_attempt_correctness = list(u_dict['correct_map'].values())[0]['correctness']
				else:
					u_attempt_correctness = None # for now, maybe revert to most recent bool later
				num_u_attempts = max(num_u_attempts, tmp)

			# note we add to list only if distinct from previous
			if u_submissions == [] or u_submissions[-1] != current_sub:
				u_submissions.append(current_sub)
				u_correctness.append(u_attempt_correctness)
				u_timestamps.append(u_time)
				u_scores.append(u_score)

		
		# length of list should agree with number of attempts
		# recorded for that student and problem
		if len(u_submissions) != num_u_attempts:
			resubmitted_same_code = True
		
		distinct = len(u_submissions)
		unique = len(set(u_submissions))
		final_correct_bool = (u_correctness[0]=='correct')
		distinct_unique_diff = len(set(u_submissions))!=len(u_submissions)

		if verbose:
			print('correctness: ', u_correctness)
			print('submissions: ', u_submissions)
			print('timestamps: ', u_timestamps)
			print('number of distinct submissions: ', distinct)
			print('number of unique submissions: \t', unique)
			print('number of submissions: \t\t', num_u_attempts)
			print('final correctness: \t\t', final_correct_bool)
			print('resubmitted same code: \t\t', resubmitted_same_code)
			print('distinct and unique different: \t', distinct_unique_diff)
			print('\n')
			if len(u_correctness) != len(u_submissions):
				print('WARNING: lengths of correctness bool list and submissions list do not match')

		# for now, just running for a few
		if num_u_attempts > 0: # add to tmp plotting df
			clean_histories.append({'user_id' : user_id,
				'user_exp' : id_exp.loc[id_exp['user_id']==int(user_id), 'response'].iloc[0],
				'problem_id' : problem_id,
				'final_correct_bool' : final_correct_bool,
				'num_distinct' : distinct,
				'num_unique' : unique,
				'num_submitted' : num_u_attempts,
				'resubmitted_same_code' : resubmitted_same_code,
				'distinct_unique_diff': distinct_unique_diff,
				'unique_correctness_list' : u_correctness,
				'score_list' : u_scores,
				'timestamp_list' : u_timestamps,
				'unique_submission_history' : u_submissions
				})

	# saving from a pandas dataframe to a csv
	# cols: user, user_id, final_correct_bool,
	# 	num_unique, num_submitted, resubmitted_same_code, distinct_unique_diff
	# 	submission_attempts, correctness_list
	# not including term b/c keeping files separate, could also include problem_id in filename
	history_features = pd.DataFrame(clean_histories,
		columns=['user_id',
		'user_exp',
		'problem_id',
		'final_correct_bool',
		'num_distinct',
		'num_unique',
		'num_submitted',
		'resubmitted_same_code',
		'distinct_unique_diff',
		'unique_correctness_list',
		'score_list',
		'timestamp_list',
		'unique_submission_history'])
	history_features.to_csv(out_csv)

	total_users = len(all_submissions.keys())
	total_users_with_subs = total_users - total_users_without_subs
	diff = total_users - total_users_with_subs
	user_totals_str = 'Parsed {} Problem {}-{} for {} users, {} with submission attempts. Diff: {}'
	print(user_totals_str.format(yr, unit, pb, total_users, total_users_with_subs, diff))


def convert_str_python(a_string, rep_tups):
	'''Takes a string and performs the string replaces
	based on the passed tuples (old, new)
	:param a_string: string on which to perform replacements
	:param rep_tups: any number of tuples of the form (old, new), comma separated
	'''
	return functools.reduce(lambda a, kv: a.replace(*kv), rep_tups, a_string)


def eval_str_if_no_none(a_string):
	'''Takes a string and evaluates it
		if it does not contain None.
		Returns None otherwise.
		:param a_string: string to potentially eval
	'''
	if 'None' in a_string:
		return None
	return eval(a_string)


if __name__ == "__main__":
	main()
