import math
import random

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

def analytical(posterior_stats, theta_range, data_points):
	""" Generate theta values within desired range and calculate corresponding posteriors """
	thetas = numpy.linspace(theta_range['min'], theta_range['max'], data_points)
	posteriors = numpy.exp(-0.5 * ((thetas - posterior_stats['mean'])/posterior_stats['stdev'])**2)

	return thetas, posteriors

def rejection_sampling(posterior_stats, theta_range):
	""" Numerical solution (Rejection sampling)

	:param width: number of standard deviations from the mean which are included
	"""
	n_samples = 10000
	posterior_min = 0
	posterior_max = 1

	# Generate random uniformly distributed x and y coordinates in appropriate ranges
	x = [random.uniform(theta_range['min'], theta_range['max']) for _ in range(n_samples)]
	y = [random.uniform(posterior_min, posterior_max) for _ in range(n_samples)]

	# Calculate posterior at each x value
	comparison_posterior = [math.exp(-0.5 * ((i - posterior_stats['mean'])/posterior_stats['stdev'])**2) for i in x]
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

def metropolis_hastings(posterior_stats, theta_initial, proposal_stdev, posterior_evaluations):
	""" Numerical solution (Metropolis-Hastings algorithm) """	
	
	thetas_mh = []
	# For burn-in plot
	posteriors_mh = []
	theta_current = theta_initial
	# For acceptance-ratio plot
	accepts = 0
	# Find number of iterations to complete, based on maximum number of posterior evaluations allowed
	iterations = int(numpy.floor(posterior_evaluations / 2))
	for i in range(iterations):
		theta_proposed = generate_candidate(theta_current, proposal_stdev)
		posterior_current = calculate_posterior(theta_current, posterior_stats)
		posterior_proposed = calculate_posterior(theta_proposed, posterior_stats)
		acceptance_probability = calculate_acceptance_probability(posterior_current, posterior_proposed)

		# Always accept proposed value if it is more likely than current value
		if acceptance_probability >= 1:
			theta_current = theta_proposed
			posterior_current = posterior_proposed
			accepts += 1
		# If proposed value less likely than current value, accept with probability 'acceptance_ratio'
		else:
			random_number = random.uniform(0,1)
			if random_number <= acceptance_probability:
				theta_current = theta_proposed
				posterior_current = posterior_proposed
				accepts += 1
		
		thetas_mh.append(theta_current)
		posteriors_mh.append(posterior_current)

	return thetas_mh, posteriors_mh, accepts

def plot_rejection_sampling(thetas, posteriors, x_accepts, bins):
	""" Plot analytical solution and rejection sampling solution on same graph """
	plt.plot(thetas, posteriors, linewidth=3)
	
	#Rejection sampling plot
	hist, bin_edges = numpy.histogram(x_accepts, bins)	
	bin_width = bin_edges[1] - bin_edges[0]
	hist = hist / max(hist)
	plt.bar(bin_edges[:-1], hist, bin_width, color='green')
	
	plt.xlabel(r'$\theta$', fontsize=15)
	plt.ylabel(r'$\propto P(\theta|x)$', fontsize=15)
	plt.show()
	

def plot_metropolis_hastings(thetas, posteriors, thetas_mh, bins):
	""" Plot analytical solution and Metropolis-Hastinga solution on same graph """
	plt.plot(thetas, posteriors, linewidth=3)

	# Metropolis-Hastings plot
	hist, bin_edges = numpy.histogram(thetas_mh, bins)	
	bin_width = bin_edges[1] - bin_edges[0]
	hist = hist / max(hist)
	plt.bar(bin_edges[:-1], hist, bin_width, color='green')

	plt.xlabel(r'$\theta$', fontsize=15)
	plt.ylabel(r'$\propto P(\theta|x)$', fontsize=15)
	plt.show()

def plot_log(thetas, posteriors, numerical_thetas, bins):
	""" Plot logarithm of histogram for numerical methods """
	plt.plot(thetas, -numpy.log(posteriors), linewidth=3)

	hist, bin_edges = numpy.histogram(numerical_thetas, bins)	
	bin_width = bin_edges[1] - bin_edges[0]
	hist = hist / max(hist)
	plt.bar(bin_edges[:-1], -numpy.log(hist), bin_width, color='green')

	plt.xlabel(r'$\theta$', fontsize=15)
	plt.ylabel(r'$\propto log(P(\theta|x))$', fontsize=15)
	plt.show()

def plot_burn_in(thetas_mh, posteriors_mh):
	""" Burn-in plot for Metropolis-Hastings method """
	step = list(range(1, len(thetas_mh)+1))
	
	plt.plot(numpy.log(step), -numpy.log(posteriors_mh))

	plt.xlabel(r'$log(step)$', fontsize=15)
	plt.ylabel(r'$\propto log(P(\theta|x))$', fontsize=15)
	plt.show()

def proposal_stdev_effects(posterior_stats, theta_initial, posterior_evaluations, proposal_stdev_min = 0.06, proposal_stdev_max = 0.26, data_points = 20):

	# Find number of iterations completed in MH
	iterations = int(numpy.floor(posterior_evaluations / 2))
	
	mh_stdevs = []
	acceptance_ratios = []
	proposal_stdevs = []
	proposal_stdev_interval = (proposal_stdev_max - proposal_stdev_min) / data_points
	proposal_stdev = proposal_stdev_min
	for i in range(data_points):
		thetas_mh, posteriors_mh, accepts = metropolis_hastings(posterior_stats, theta_initial, proposal_stdev, posterior_evaluations)

		acceptance_ratios.append(accepts / iterations)
		proposal_stdevs.append(proposal_stdev)
		mh_stdevs.append(numpy.std(posteriors_mh)) 

		proposal_stdev = proposal_stdev + proposal_stdev_interval

	return(proposal_stdevs, acceptance_ratios, mh_stdevs)
	
def plot_acceptance_ratios(proposal_stdevs, acceptance_ratios):
	""" Plot acceptance ratio for different standard deviations of the proposal distribution """
	plt.plot(proposal_stdevs, acceptance_ratios, marker='x', linestyle='none')

	plt.xlabel('Proposal Standard Deviation')
	plt.ylabel('Acceptance Ratio')
	plt.show()

def plot_mh_stdevs(proposal_stdevs, mh_stdevs):
	""" Plot standard devation of posterior from MH method against that of proposal distribution """
	plt.plot(proposal_stdevs, mh_stdevs, marker='x', linestyle='none')

	plt.xlabel('Proposal Standard Deviation')
	plt.ylabel('Posterior Standard Deviation')
	plt.show()

def main():
	population_mean = 0.5
	population_stdev = 0.1
	sample_size = 10
	prior_stdev = 1

	bins = 100
	# Number of stdevs from the mean over which analytical and rejection sampling results will be found
	width = 6

	data = simulate_experiment(population_mean, population_stdev, sample_size)
	sample_mean = numpy.mean(data)
	posterior_stats = posterior_info(sample_mean, prior_stdev, sample_size, population_stdev)
	theta_range = sigma_range(posterior_stats['mean'], posterior_stats['stdev'], width)

	# Analytical
	data_points = 100
	thetas, posteriors = analytical(posterior_stats, theta_range, data_points)

	# Rejection sampling
	x_accepts = rejection_sampling(posterior_stats, theta_range)
	plot_rejection_sampling(thetas, posteriors, x_accepts, bins)
	plot_log(thetas, posteriors, x_accepts, bins)

	# Metropolis-Hastings
	theta_initial = 0.45
	posterior_evaluations = 80000
	proposal_stdev = 0.3
	thetas_mh, posteriors_mh, accepts = metropolis_hastings(posterior_stats, theta_initial, proposal_stdev, posterior_evaluations)
	plot_metropolis_hastings(thetas, posteriors, thetas_mh, bins)
	plot_log(thetas, posteriors, thetas_mh, bins)

	plot_burn_in(thetas_mh, posteriors_mh)

	proposal_stdevs, acceptance_ratios, mh_stdevs = proposal_stdev_effects(posterior_stats, theta_initial, posterior_evaluations)
	plot_acceptance_ratios(proposal_stdevs, acceptance_ratios)
	plot_mh_stdevs(proposal_stdevs, mh_stdevs)

if __name__ == '__main__':
	main()
