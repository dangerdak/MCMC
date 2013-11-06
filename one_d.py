import math
import random
from argparse import ArgumentParser

import numpy
import matplotlib.pyplot as plt


def simulate_experiment(mean, stdev, N):
	""" Generate values for N independently Gaussian distributed 'measurements' """
	random.seed(0)
	data = [random.gauss(mean, stdev) for i in range(N)]

	return data

def sigma_range(mean, stdev, width):
	""" Find range of x values to be included, based on number of standard deviations to be included """
	lower = mean - width * stdev
	upper = mean + width * stdev
	bounds = {'min': lower, 'max': upper}

	return bounds

def posterior_info(sample_mean, prior_stdev, sample_size, population_stdev):
	""" Calculate the posterior mean and standard deviation """
	posterior_stdev = (1/(prior_stdev**2) + sample_size/(population_stdev**2))**(-1/2)
	posterior_mean = sample_mean * prior_stdev**2 / (prior_stdev**2 + population_stdev**2/sample_size)
	posterior_stats = {'mean': posterior_mean, 'stdev': posterior_stdev}

	return posterior_stats

def setup(population_mean, population_stdev, sample_size, prior_stdev, width):
	data = simulate_experiment(population_mean, population_stdev, sample_size)
	sample_mean = numpy.mean(data)
	posterior_stats = posterior_info(sample_mean, prior_stdev, sample_size, population_stdev)
	theta_range = sigma_range(posterior_stats['mean'], posterior_stats['stdev'], width)

	return posterior_stats, theta_range

def analytical(posterior_stats, theta_range, data_points):
	""" Generate theta values within desired range and calculate corresponding posteriors """
	thetas = numpy.linspace(theta_range['min'], theta_range['max'], data_points)
	posteriors = numpy.exp(-0.5 * ((thetas - posterior_stats['mean'])/posterior_stats['stdev'])**2)

	return thetas, posteriors

def rejection_sampling(posterior_stats, theta_range, iterations):
	""" Numerical solution (Rejection sampling) """
	posterior_min = 0
	posterior_max = 1

	# Generate random uniformly distributed x and y coordinates in appropriate ranges
	x = [random.uniform(theta_range['min'], theta_range['max']) for _ in range(iterations)]
	y = [random.uniform(posterior_min, posterior_max) for _ in range(iterations)]

	# Calculate posterior at each x value
	comparison_posterior = [calculate_posterior(i, posterior_stats) for i in x]
	x_accepts = []
	for i,j,k in zip(x,y,comparison_posterior):
		if j <= k:
			x_accepts.append(i)

	return x_accepts

def generate_candidate(theta_current, proposal_stdev):
	""" Generate a candidate theta value """
	theta_proposed = random.gauss(theta_current, proposal_stdev)

	return theta_proposed

def calculate_posterior(theta, posterior_stats):
	""" Calculate the value of the posterior for a given theta and posterior mean and standard deviation """
	posterior = math.exp(-0.5 * ((theta - posterior_stats['mean'])/posterior_stats['stdev'])**2) 

	return posterior

def calculate_acceptance_probability(posterior_current, posterior_proposed):
	""" Calculate 'probability' of accepting proposed theta """
	acceptance_probability = posterior_proposed / posterior_current

	return acceptance_probability

def metropolis_hastings(posterior_stats, theta_initial, proposal_stdev, iterations):
	""" Numerical solution (Metropolis-Hastings algorithm) """	
	
	thetas_mh = []
	# For burn-in plot
	posteriors_mh = []
	theta_current = theta_initial
	posterior_current = calculate_posterior(theta_current, posterior_stats)
	# For acceptance rate plot
	accepts = 0
	for i in range(iterations):
		theta_proposed = generate_candidate(theta_current, proposal_stdev)
		posterior_proposed = calculate_posterior(theta_proposed, posterior_stats)
		acceptance_probability = calculate_acceptance_probability(posterior_current, posterior_proposed)

		# Always accept proposed value if it is more likely than current value
		# If proposed value less likely than current value, accept with probability 'acceptance_proability'
		if (acceptance_probability >= 1) or (random.uniform(0,1) <= acceptance_probability):
			theta_current = theta_proposed
			posterior_current = posterior_proposed
			accepts += 1
		
		thetas_mh.append(theta_current)
		posteriors_mh.append(posterior_current)

	return thetas_mh, posteriors_mh, accepts

def plot_rejection_sampling(thetas, posteriors, x_accepts, bins):
	""" Plot analytical solution and rejection sampling solution on same graph """
	fig, ax = plt.subplots()
	plt.plot(thetas, posteriors, linewidth=3)
	
	#Rejection sampling plot
	hist, bin_edges = numpy.histogram(x_accepts, bins)	
	bin_width = bin_edges[1] - bin_edges[0]
	hist = hist / max(hist)
	ax.bar(bin_edges[:-1], hist, bin_width, color='green')

	# Create strings to show numerical mean and standard deviation on graphs
	mean = numpy.mean(x_accepts)
	stdev = numpy.std(x_accepts)
	display_string = ('$\mu_{{MC}} =$ {0:.3f} \n$\sigma_{{MC}} =$ {1:.3f}').format(mean, stdev)

	plt.xlabel(r'$\theta$', fontsize=16)
	plt.ylabel(r'$\propto P(\theta|x)$', fontsize=16)
	plt.text(0.7, 0.8, display_string, transform=ax.transAxes, fontsize=16)
	plt.savefig('rejection.png', bbox_inches='tight')
	plt.show()
	

def plot_metropolis_hastings(thetas, posteriors, thetas_mh, bins):
	""" Plot analytical solution and Metropolis-Hastinga solution on same graph """
	fig, ax = plt.subplots()
	plt.plot(thetas, posteriors, linewidth=3)

	# Metropolis-Hastings plot
	hist, bin_edges = numpy.histogram(thetas_mh, bins)	
	bin_width = bin_edges[1] - bin_edges[0]
	hist = hist / max(hist)
	ax.bar(bin_edges[:-1], hist, bin_width, color='green')

	# Create strings to show numerical mean and standard deviation on graphs
	mean = numpy.mean(thetas_mh)
	stdev = numpy.std(thetas_mh)
	display_string = ('$\mu_{{MC}} =$ {0:.3f} \n$\sigma_{{MC}} =$ {1:.3f}').format(mean, stdev)

	plt.xlabel(r'$\theta$', fontsize=16)
	plt.ylabel(r'$\propto P(\theta|x)$', fontsize=16)
	plt.text(0.7, 0.8, display_string, transform=ax.transAxes, fontsize=16)
	plt.savefig('metropolishastings.png', bbox_inches='tight')
	plt.show()

def plot_log(thetas, posteriors, numerical_thetas, bins):
	""" Plot logarithm of histogram for numerical methods """
	fig, ax = plt.subplots()
	plt.plot(thetas, -numpy.log(posteriors), linewidth=3)

	hist, bin_edges = numpy.histogram(numerical_thetas, bins)	
	bin_width = bin_edges[1] - bin_edges[0]
	hist = hist / max(hist)
	plt.bar(bin_edges[:-1], -numpy.log(hist), bin_width, color='green')

	# Create strings to show numerical mean and standard deviation on graphs
	mean = numpy.mean(numerical_thetas)
	stdev = numpy.std(numerical_thetas)
	display_string = ('$\mu_{{MC}} =$ {0:.3f} \n$\sigma_{{MC}} =$ {1:.3f}').format(mean, stdev)

	plt.xlabel(r'$\theta$', fontsize=16)
	plt.ylabel(r'$\propto log(P(\theta|x))$', fontsize=16)
	plt.text(0.5, 0.5, display_string, transform=ax.transAxes, fontsize=16)
	plt.savefig('recentlog.png', bbox_inches='tight')
	plt.show()

def plot_log_both(thetas, posteriors, thetas_r, thetas_mh, bins):
	""" Plot logarithm of histogram for both numerical methods in one figure"""
	fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True)
		
	# Create strings to show numerical mean and standard deviation on graphs
	mean_r = numpy.mean(thetas_r)
	stdev_r = numpy.std(thetas_r)
	display_string_r = ('$\mu_{{MC}} =$ {0:.3f} \n$\sigma_{{MC}} =$ {1:.3f}').format(mean_r, stdev_r)

	mean_mh = numpy.mean(thetas_mh)
	stdev_mh = numpy.std(thetas_mh)
	display_string_mh = ('$\mu_{{MC}} =$ {0:.3f} \n$\sigma_{{MC}} =$ {1:.3f}').format(mean_mh, stdev_mh)
	# Position relative to axes (0,1)
	text_x = 0.55
	text_y = 0.45

	# Rejection sampling plot
	ax1.plot(thetas, -numpy.log(posteriors), linewidth=3)

	hist, bin_edges = numpy.histogram(thetas_r, bins)	
	bin_width = bin_edges[1] - bin_edges[0]
	hist = hist / max(hist)
	ax1.bar(bin_edges[:-1], -numpy.log(hist), bin_width, color='green')
	ax1.text(text_x, text_y, display_string_r, transform=ax1.transAxes, fontsize=16)

	ax1.set_xlabel(r'$\theta$', fontsize=16)
	ax1.set_ylabel(r'$\propto log(P(\theta|x))$', fontsize=16)

	# Metropolis-Hastings plot
	ax2.plot(thetas, -numpy.log(posteriors), linewidth=3)

	hist, bin_edges = numpy.histogram(thetas_mh, bins)	
	bin_width = bin_edges[1] - bin_edges[0]
	hist = hist / max(hist)
	ax2.bar(bin_edges[:-1], -numpy.log(hist), bin_width, color='green')
	ax2.text(text_x, text_y, display_string_mh, transform=ax2.transAxes, fontsize=16)

	ax2.set_xlabel(r'$\theta$', fontsize=16)

	plt.savefig('bothlogs.png')
	plt.show()

def plot_burn_in(iterations, thetas_mh, posteriors_mh):
	""" Burn-in plot for Metropolis-Hastings method """
	step = list(range(1, len(thetas_mh)+1))
	
	plt.plot(range(20000), thetas_mh[1:20001])
	plt.xlim(-100)
	plt.ylim(0.05, 0.6)

	plt.xlabel('Iteration', fontsize=16)
	plt.ylabel(r'$\theta$', fontsize=16)
	plt.savefig('burnin.png', bbox_inches='tight')
	plt.show()

def proposal_stdev_effects(posterior_stats, theta_initial, iterations, proposal_stdev_min = 0.06, proposal_stdev_max = 0.26, data_points = 20):
	""" Returns data showing effects of changing the standard deviation of the proposal distribution """
	mh_stdevs = []
	acceptance_rates = []
	proposal_stdevs = []
	proposal_stdev_interval = (proposal_stdev_max - proposal_stdev_min) / data_points
	proposal_stdev = proposal_stdev_min
	for i in range(data_points):
		thetas_mh, posteriors_mh, accepts = metropolis_hastings(posterior_stats, theta_initial, proposal_stdev, iterations)

		acceptance_rates.append(accepts / iterations)
		proposal_stdevs.append(proposal_stdev)
		mh_stdevs.append(numpy.std(posteriors_mh)) 

		proposal_stdev = proposal_stdev + proposal_stdev_interval

	return(proposal_stdevs, acceptance_rates, mh_stdevs)
	
def plot_proposal(proposal_stdevs, acceptance_rates, mh_stdevs):
	""" Plots showing effect of changing te standard deviation of the proposal distribution """
	plt.figure()
	plt.subplot(1, 2, 1)
	# Plot acceptance rate for different standard deviations of the proposal distribution 
	plt.plot(proposal_stdevs, acceptance_rates, marker='x', linestyle='none')
	plt.xlabel('Proposal Standard Deviation')
	plt.ylabel('Acceptance Ratio')

	plt.subplot(1, 2, 2)
	# Plot standard devation of posterior from MH method against that of proposal distribution 
	plt.plot(proposal_stdevs, mh_stdevs, marker='x', linestyle='none')
	plt.xlabel('Proposal Standard Deviation')
	plt.ylabel('Posterior Standard Deviation')
	plt.savefig('proposalstdev.png', bbox_inches='tight')
	plt.show()
	

def main():

	# Use command line arguments to determine which parts of code to run
	modes = ['rejection', 'metropolis_hastings', 'all', 'proposal']
	parser = ArgumentParser(description='One dimensional MCMC')
	parser.add_argument('--mode', type=str, default='all', choices=modes, help='Specify which section of the program to run.')
	args = parser.parse_args()

	population_mean = 0.5
	population_stdev = 0.1
	sample_size = 10
	prior_stdev = 1

	bins = 100
	# Number of stdevs from the mean over which analytical and rejection sampling results will be found
	width = 5
	iterations = 100000

	posterior_stats, theta_range = setup(population_mean, population_stdev, sample_size, prior_stdev, width)

	# Analytical
	data_points = 100
	thetas, posteriors = analytical(posterior_stats, theta_range, data_points)

	if (args.mode == 'rejection') or (args.mode == 'all'):
		# Rejection sampling
		x_accepts = rejection_sampling(posterior_stats, theta_range, iterations)
		plot_rejection_sampling(thetas, posteriors, x_accepts, bins)
		plot_log(thetas, posteriors, x_accepts, bins)

	if (args.mode == 'metropolis_hastings') or (args.mode == 'proposal') or (args.mode == 'all'):
		# Variables required by metropolis and proposal
		theta_initial = 0.1
		
	if (args.mode == 'metropolis_hastings') or (args.mode == 'all'):
		# Metropolis-Hastings
		proposal_stdev = 0.01
		thetas_mh, posteriors_mh, accepts = metropolis_hastings(posterior_stats, theta_initial, proposal_stdev, iterations)
		plot_metropolis_hastings(thetas, posteriors, thetas_mh, bins)
		plot_log(thetas, posteriors, thetas_mh, bins)

		plot_burn_in(iterations, thetas_mh, posteriors_mh)
	
	if (args.mode =='all'):
		plot_log_both(thetas, posteriors, x_accepts, thetas_mh, bins)

	if (args.mode == 'proposal'):
		# Effects of changing the proposal distributions standard deviation
		proposal_stdevs, acceptance_rates, mh_stdevs = proposal_stdev_effects(posterior_stats, theta_initial, iterations)
		plot_proposal(proposal_stdevs, acceptance_rates, mh_stdevs)

if __name__ == '__main__':
	main()
