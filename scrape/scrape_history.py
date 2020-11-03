import pandas as pd
import pickle
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import Select
from selenium.common.exceptions import StaleElementReferenceException, NoSuchElementException, NoSuchWindowException, InvalidElementStateException, UnexpectedAlertPresentException
from selenium.webdriver.common.by import By
import sys
from threading import Thread
import time

import login_cred
import problem_params

course_info = {
	'2017' : ['Name', 'Offering', 'URL'],
}

yr = sys.argv[1]
unit = int(sys.argv[2])
pb = int(sys.argv[3])
i = int(sys.argv[4])
#sys.argv[5] is 'reload', giving option to restart scraping from a later index


tmpName = '../../data/pickle/{}_unit{}_pb{}_history.pkl'.format(yr, unit, pb)

# points to data dirs with csvs of usernames
unameData = 'u_data/{}_name_id.csv'.format(yr)

cnum, tnum, unit_url = course_info[yr][0], course_info[yr][1], course_info[yr][2]

urls = {
	1 : '{}{}{}'.format(unit_url, cnum, tnum),
}

pset1 = problem_params.pset1
pset2 = problem_params.pset2
pset3 = problem_params.pset3
pset4 = problem_params.pset4
pset5 = problem_params.pset5

url = urls[unit] # url for unit pset

# get problem tab webpage id and problem id
(tab_id, problem_id) = eval('pset{}'.format(unit))[pb]

# get button, modal, and form history webpage ids
button_id = '{}_history_trig'.format(problem_id)
modal_id = '{}_history_student_username'.format(problem_id)
form_history_id = '{}_history_text'.format(problem_id)

# get usernames
zusernames = pd.read_csv(unameData)
usernames = list(map(str, zusernames.username))
user_ids = list(map(str, zusernames.user_id))

results = {}

# command line option to reload - start from particular index
if sys.argv[5] == 'reload':
	with open(tmpName, "rb") as f:
		i, results = pickle.load(f)

browsers = []
browserIdx = 0


def addBrowser():
	path_to_chromedriver = 'chromedriver' # change path as needed
	browser = webdriver.Chrome(executable_path = path_to_chromedriver)
	browser.get(url) # send browser to correct page for unit pset

	browser.find_element_by_id("login-email").send_keys(login_cred.login_u);
	browser.find_element_by_id("login-password").send_keys(login_cred.login_p);
	browser.find_element_by_id("login-password").send_keys(Keys.ENTER);
	time.sleep(15) # wait for problem page to load

	# once on page for pset, navigate to problem tab
	tab =  browser.find_element_by_id(tab_id)
	tab.click();
	time.sleep(2)

	# click button to view submission history
	button = browser.find_element_by_id(button_id)
	button.click();
	time.sleep(2)

	browsers.append(browser)

def killBrowser(bIdx):
	browsers[bIdx].quit()
	if bIdx + 2 > len(browsers):
		print("adding two browsers")
		Thread(target = addBrowser).start()
		Thread(target = addBrowser).start()
		Thread(target = addBrowser).start()
		time.sleep(15)
	return bIdx + 1


addBrowser()
addBrowser()

new_window = True

while i < len(usernames):

	u, u_id = usernames[i], user_ids[i]
	print("%i,  %s of %i" % (i, u, len(usernames)))
	browser = browsers[browserIdx]

	try:
		# enter the username in the form and hit enter
		modal = browser.find_element_by_id(modal_id)
		modal.clear(); # clears the last username
		modal.send_keys(u);
		modal.send_keys(Keys.ENTER);
		time.sleep(10)
		
		submissionsElt = browser.find_element_by_id(form_history_id)
		
	except (StaleElementReferenceException, InvalidElementStateException, NoSuchElementException) as e:
		browserIdx = killBrowser(browserIdx)
		print("caught exception, retrying...")
		print(e)
		continue
		
	try:
		# get submission history from form HTML
		response = submissionsElt.get_attribute("innerHTML")
	
	except UnexpectedAlertPresentException as e:
		response = ''
		browserIdx = killBrowser(browserIdx)

	# save response and write to file only if attempted
	if 'attempts' in response:
		results[u_id] = response

		with open(tmpName, "wb") as f:
			pickle.dump((i, results), f)
		print("Wrote response for user {} ({}).".format(u, u_id))
	else:
		print("{} ({}) did not attempt".format(u, u_id))
	
	i += 1
	
for bi in range(len(browsers)):
	browsers[bi].quit()
	