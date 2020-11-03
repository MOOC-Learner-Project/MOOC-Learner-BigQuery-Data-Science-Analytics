import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import sys
from scipy import stats

def except_divide(n, d):
    return n*1.0 / d if d else 0.0

exp_list = ['veteran', 'know_python', 'other_language', 'absolutely_none', 'no_response']

unit = 1
pb = 1

traj_data_csv = '../../data/keyword_occurrence/trajectory/unit{}_pb{}_kw_occ_traj.csv'
traj_data = pd.read_csv(traj_data_csv.format(unit, pb), index_col=None)

traj_data['user_exp'] = pd.Categorical(traj_data['user_exp'], exp_list)
traj_data = traj_data.sort_values(by=['user_exp'], ascending=False)


# plots
attempts = traj_data['total_submissions'].to_numpy()
vid = traj_data['video_engaged'].to_numpy()
exp = traj_data['user_exp'].to_numpy()

plt.figure(figsize=(12, 8))
sns.scatterplot(x=attempts, y=exp, hue=vid)
plt.title('Number of attempts vs. prior experience for Problem {}-{}'.format(unit, pb))
plt.legend(title='Video engaged')
plt.show()

# worried the above is misleading because of overlapping points
traj_data_vid_eng = traj_data.loc[traj_data['video_engaged'] == True]
traj_data_vid_no = traj_data.loc[traj_data['video_engaged'] == False]


plt.figure(figsize=(12, 8))
sns.scatterplot(data=traj_data_vid_eng, x='total_submissions', y='user_exp')
#traj_data_vid_eng = traj_data_vid_eng.pivot('total_submissions', 'user_exp', 'video_engaged')
#sns.histogram(traj_data_vid_eng)
plt.title('ENGAGED Number of attempts vs. prior experience for Problem {}-{}'.format(unit, pb))
plt.show()


plt.figure(figsize=(12, 8))
sns.scatterplot(data=traj_data_vid_no, x='total_submissions', y='user_exp')
plt.title('NOT ENGAGED Number of attempts vs. prior experience for Problem {}-{}'.format(unit, pb))
plt.show()


# make heatmap of counts per no. attempts and experience
traj_data_counts = pd.DataFrame(columns=['attempts', 'experience', 'count'])
for exp in exp_list:
	for i in range(1, 31):
		exp_sub_cond = (traj_data['user_exp']==exp) & (traj_data['total_submissions']==i)
		count_dict = {'attempts' : i,
					'experience' : exp,
					'count' : float(sum(exp_sub_cond)),
					'w_vid_count' : float(sum(exp_sub_cond & (traj_data['video_engaged']==True))),
					'no_vid_count' : float(sum(exp_sub_cond & (traj_data['video_engaged']==False))),
					'w_vid_pct' : except_divide(sum(exp_sub_cond & (traj_data['video_engaged']==True)), sum(exp_sub_cond)),
					'no_vid_pct' : except_divide(sum(exp_sub_cond & (traj_data['video_engaged']==False)), sum(exp_sub_cond))}
		traj_data_counts = traj_data_counts.append(count_dict, ignore_index=True)
traj_data_counts = traj_data_counts.dropna(axis=1)

plt.figure(figsize=(10, 10))
traj_data_counts_both = traj_data_counts.pivot('attempts', 'experience', 'count')
print(traj_data_counts_both)
sns.heatmap(traj_data_counts_both)
plt.show()


# plot video engagement across experience levels
traj_data_w_vid_counts = traj_data_counts.pivot('attempts', 'experience', 'w_vid_pct')
traj_data_no_vid_counts = traj_data_counts.pivot('attempts', 'experience', 'no_vid_pct')
# plots above counts over no. attempts for each experience level
f, axarr = plt.subplots(5, sharex=True, figsize=(8, 11))
x = range(1, 16)
for i, exp in enumerate(exp_list):	
	y = traj_data_w_vid_counts[exp].to_numpy()[:15]
	y1 = traj_data_no_vid_counts[exp].to_numpy()[:15]
	axarr[i].plot(x, y, 'b.-')
	axarr[i].plot(x, y1, 'r.-')
	exp_students = sum((traj_data['user_exp']==exp))
	axarr[i].set_title('{} ({})'.format(exp, exp_students), fontsize=8)
axarr[0].legend(labels=['video engaged', 'video not engaged'], loc=1)
plt.suptitle('Fraction of video engagement for each \n number of attempts x experience category bin for Problem {}-{}'.format(unit, pb))
os.makedirs('fraction_plots', exist_ok=True)
plt.savefig('fraction_plots/frac_vid_eng_{}-{}.png'.format(unit, pb))


 