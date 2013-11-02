import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import math
from matplotlib.patches import Ellipse

def import_data(textfile, uncertainty):
	"""Import measurements from file. Each x and y pair on its own line, delimited by ', '
	i.e. 'x, y\n'. Also specify uncertainty for measurements."""
	f = open(textfile, 'r')
	data = f.readlines()
	x = []
	y = []
	for line in data:
		coords = line.strip()
		coords = coords.split(', ')
		x.append(coords[0])
		y.append(coords[1]) 
	
	x = [float(i) for i in x]
	x = np.array(x).reshape((10,1))
	y = [float(i) for i in y]
	y = np.array(y).reshape((10,1))
	data = {'x': x, 'y': y, 'var': uncertainty}

	return data

def get_design_matrix(x):
	"""Find design matrix for our specialized case (observations are fitted with linear model)."""
	F = np.ones((10, 1))
	F = np.hstack((F, x))

	return F

def get_likelihood_fisher_matrix(A):
	likelihood_fisher = np.dot(A.transpose(), A)

	return likelihood_fisher

def get_prior_fisher_matrix():
	"""Prior fisher matrix for this case. """
	prior_fisher = 0.1 * np.eye(2)
	
	return prior_fisher 

def get_posterior_fisher_matrix(likelihood_fisher, P):
	posterior_fisher = likelihood_fisher + P

	return posterior_fisher

def get_mle(likelihood_fisher, A, b):
	mle = np.dot(A.transpose(), b)
	mle = np.dot(np.linalg.inv(likelihood_fisher), mle)

	return mle

def get_posterior_mean(likelihood_fisher, posterior_fisher, mle):
	posterior_mean = np.dot(likelihood_fisher, mle)
	posterior_mean = np.dot(np.linalg.inv(posterior_fisher), posterior_mean)

	return posterior_mean

def setup(measurement_uncertainty):
	"""Do calculations neccessary to find posterior mean and posterior fisher matrix"""
	data = import_data('dataset.txt', measurement_uncertainty)
	design = get_design_matrix(data['x'])
	A = design / measurement_uncertainty
	likelihood_fisher = get_likelihood_fisher_matrix(A)
	prior_fisher = get_prior_fisher_matrix()
	posterior_fisher = get_posterior_fisher_matrix(likelihood_fisher, prior_fisher)
	b = data['y'] / measurement_uncertainty
	mle = get_mle(likelihood_fisher, A, b)
	posterior_mean = get_posterior_mean(likelihood_fisher, posterior_fisher, mle)

	# ??? Print covariance matrix
	covar = np.linalg.inv(posterior_fisher)
	print('covariance matrix:')
	print(covar)

	posterior_stats = {'fisher': posterior_fisher, 'mean': posterior_mean}

	return posterior_stats

def analytical(posterior_stats):
	domain_1 = 0.8
	domain_2 = 1.5
	res = 1000

	theta_1_initial = posterior_stats['mean'][0] - domain_1/2
	theta_2_initial = posterior_stats['mean'][1] - domain_2/2

	thetas = {'1': np.zeros((res,1)), '2':np.zeros((res,1)), 'initial_1': theta_1_initial, 'initial_2': theta_2_initial}

	# ??? Dont think I need this unless analytical contours need it......

	return analytical_posteriors

def calculate_ln_posterior(thetas, posterior_stats):
	"""Calculate the natural logarithm of the posterior for given theta values."""
	ln_posterior = np.dot(posterior_stats['fisher'], (thetas - posterior_stats['mean']))
	ln_posterior = - np.dot((thetas - posterior_stats['mean']).transpose(), ln_posterior) /2

	return ln_posterior

def generate_candidates(thetas, proposal_stdev):
	"""Generate candidate theta values using proposal distribution."""
	thetas_proposed = np.zeros((2, 1))
	thetas_proposed[0][0] = random.gauss(thetas[0][0], proposal_stdev[0][0])
	thetas_proposed[1][0] = random.gauss(thetas[1][0], proposal_stdev[1][0])
	
	return thetas_proposed

def calculate_hastings_ratio(ln_proposed, ln_current):
	ln_hastings = ln_proposed - ln_current
	hastings = np.exp(ln_hastings)

	return hastings
	
def metropolis_hastings(posterior_stats):
	"""Sample from posterior distribution using Metropolis-Hastings algorithm."""
	iterations = 5000
	thetas = np.array([[-0.05], [0.5]])
	proposal_stdev = np.array([[0.1], [0.1]])
	ln_posterior = calculate_ln_posterior(thetas, posterior_stats)
	accepts = 0
	mcmc_samples = thetas 

	for i in range(iterations):
		thetas_proposed = generate_candidates(thetas, proposal_stdev)
		ln_posterior_proposed = calculate_ln_posterior(thetas_proposed, posterior_stats)
		
		hastings_ratio = calculate_hastings_ratio(ln_posterior_proposed, ln_posterior)	
		
		acceptance_probability = min([1, hastings_ratio])

		if (random.uniform(0,1) < acceptance_probability):
			#Then accept proposed thetas
			thetas = thetas_proposed
			ln_posterior = ln_posterior_proposed
			accepts += 1
		mcmc_samples = np.hstack((mcmc_samples, thetas))

	mcmc_mean = np.array([ [np.mean(mcmc_samples[0])], [np.mean(mcmc_samples[1])] ])
	mcmc = {'samples': mcmc_samples, 'mean': mcmc_mean} 
	acceptance_ratio = accepts / iterations

	return mcmc, acceptance_ratio

# Do I uyse this ???
def edges_to_centers(x_edges, y_edges, res):
	"""Given edges and width of bins, find centres."""
	dx = (max(x_edges) - min(x_edges)) / res
	dy = (max(y_edges) - min(y_edges)) / res

	x = x_edges + dx /2 
	y = y_edges + dy /2 
	x = x[:-1]
	y = y[:-1]

	return x, y

def equal_weight(counts, res):
	"""Find equal weight samples."""
	multiplicity = counts / counts.max()
	randoms = np.random.random((res, res))

	equal_weighted_samples = multiplicity < randoms

	return equal_weighted_samples

def sigma_boundary(counts, percentage):
	"""Find boundary values for each sigma-level."""
	# Sort counts in descending order
	counts_desc = sorted(counts.flatten(), reverse=True)
	# Find cumulative sum of sorted counts
	cumulative_counts = np.cumsum(counts_desc)
	# Create a mask for counts outside of percentage boundary
	sum_mask = cumulative_counts < (percentage /100) * np.sum(counts)
	sigma_sorted = sum_mask * counts_desc
	# ??? Assume that density is ??? ellipse equivalent of radially symmetric
	sigma_min = min(sigma_sorted[sigma_sorted.nonzero()])

	return sigma_min

def find_numerical_contours(counts):
	"""Returns array of 3s, 2s, 1s, and 0s, representing one two and three sigma regions respectively."""
	one_sigma_boundary = sigma_boundary(counts, 68)
	one_sigma = counts > one_sigma_boundary
	two_sigma_boundary = sigma_boundary(counts, 95)
	two_sigma = (counts > two_sigma_boundary) & (counts < one_sigma_boundary)
	three_sigma_boundary = sigma_boundary(counts, 98)
	three_sigma = (counts > three_sigma_boundary) & (counts < two_sigma_boundary)

	# Check method: Output actual percentages in each region
	print('total no. samples:')
	print(np.sum(counts))
	print('included in 1st sigma region:')
	print(np.sum(one_sigma * counts) / np.sum(counts))
	print('included in 2 sigma region:')
	print(np.sum(one_sigma * counts) + np.sum(two_sigma * counts) / np.sum(counts))
	print('included in 3 sigma region:')
	print(np.sum(one_sigma * counts) + np.sum(two_sigma * counts) + np.sum(three_sigma * counts) / np.sum(counts))

	filled_numerical_contours = one_sigma * 3 + two_sigma * 2 + three_sigma

	return filled_numerical_contours

def plot_samples(mcmc, res):
	"""Plot numerical equal-weight samples and filled contours."""
	counts, x_edges, y_edges = np.histogram2d(mcmc['samples'][0], mcmc['samples'][1], bins=res)
	counts = np.flipud(np.rot90(counts))
	
	equal_weighted_samples = equal_weight(counts, res)
	
	plt.pcolormesh(x_edges, y_edges, equal_weighted_samples, cmap=plt.cm.gray)
	plt.show()

	filled_numerical_contours = find_numerical_contours(counts)
	plt.pcolormesh(x_edges, y_edges, filled_numerical_contours, cmap=plt.cm.binary)
	plt.show()

	marginalized = marginalize(counts)
	plt.bar(x_edges[:-1], marginalized['theta_1'], x_edges[1]-x_edges[0], color='white')
	plt.show()

	plt.bar(y_edges[:-1], marginalized['theta_2'], y_edges[1]-y_edges[0], color='white')
	plt.show()

	return counts
	
def marginalize(counts):
	"""Find marginalized distribution for each parameter."""
	# Sum columns
	x_counts = np.sum(counts, axis=0)
	# Sum rows
	y_counts = np.sum(counts, axis=1)

	marginalized = {'theta_1': x_counts, 'theta_2': y_counts}

	return marginalized

def plot_marginalized(mcmc, res):
	fig = plt.figure(1, figsize=(7,7))
	fig.subplots_adjust(hspace=0.001, wspace=0.001)
	gs = gridspec.GridSpec(2, 2, width_ratios=[1,4], height_ratios=[4,1])

	counts, x_edges, y_edges = np.histogram2d(mcmc['samples'][0], mcmc['samples'][1], bins=res)
	counts = np.flipud(np.rot90(counts))

	plt.subplot(gs[1])
	plt.pcolormesh(x_edges, y_edges, counts, cmap=plt.cm.binary)
	plt.tick_params(axis='both', labelleft='off', labelbottom='off')

	marginalized = marginalize(counts)
	print(y_edges.shape)
	print(marginalized['theta_2'].shape)


	plt.subplot(gs[3])
	plt.bar(x_edges[:-1], marginalized['theta_1'], x_edges[1]-x_edges[0], color='white')
	plt.tick_params(axis='y', labelleft='off')
	plt.xlabel(r'$\theta_1$')
	plt.ylabel(r'P')

	plt.subplot(gs[0])
	plt.barh(y_edges[:-1], marginalized['theta_2'], y_edges[1]-y_edges[0], color='white')
	plt.tick_params(axis='x', labelbottom='off')
	plt.ylabel(r'$\theta_2$')
	plt.xlabel(r'P')
	plt.show()

def ellipse_coords(mean, eigenval, eigenvec, level):
	chi_square = {'1': 2.30, '2': 6.18, '3': 11.83}
	level = str(level)

	axis1 = []
	axis1.append(mean + (np.sqrt(chi_square[level] * eigenval[0]) * eigenvec[:,0]))
	axis1.append(mean - (np.sqrt(chi_square[level] * eigenval[0]) * eigenvec[:,0]))

	axis2 = []
	axis2.append(mean + (np.sqrt(chi_square[level] * eigenval[1]) * eigenvec[:,1]))
	axis2.append(mean - (np.sqrt(chi_square[level] * eigenval[1]) * eigenvec[:,1]))

	return axis1, axis2

def ellipse_lengths(axis1, axis2):
	dx1 = axis1[1][0] - axis1[0][0]
	dy1 = axis1[0][1] - axis1[1][1]
	length1 = math.sqrt(dx1**2 + dy1**2)

	dx2 = axis2[1][0] - axis2[0][0]
	dy2 = axis2[0][1] - axis2[1][1]
	length2 = math.sqrt(dx2**2 + dy2**2)

	major = {'length': length1, 'coords': axis1, 'dx': dx1, 'dy': dy1}
	minor = {'length': length2, 'coords': axis2, 'dx' : dx2, 'dy': dy2}

	if length1 > length2:
		major = {'length': length1, 'coords': axis1, 'dx': dx1, 'dy': dy1}
		minor = {'length': length2, 'coords': axis2, 'dx' : dx2, 'dy': dy2}
	else:
		major = {'length': length2, 'coords': axis2, 'dx': dx2, 'dy': dy2}
		minor = {'length': length1, 'coords': axis1, 'dx': dx1, 'dy': dy1}

	return major, minor

def ellipse_angle(dx, dy):
	print('\ndx')
	print(dx)
	print('\ndy')
	print(dy)
	print('\ndy/dx')
	print(dy/dx)
	print('\natan(dy/dx)')
	print(math.atan(dy/dx))

	angle = math.atan(dx/dy)

	return angle

def find_ellipse(mean, eigenval, eigenvec, level):
	axis1, axis2 = ellipse_coords(mean, eigenval, eigenvec, level)
	major, minor = ellipse_lengths(axis1, axis2)

	angle = ellipse_angle(major['dx'], major['dy'])
	angle = angle * 180 / math.pi

	ellipse = {'mean': mean, 'width': minor['length'], 'height': major['length'], 'angle':angle}
	
	return ellipse


def analytical_contours(posterior_stats):
	covariance = np.linalg.inv(posterior_stats['fisher'])
	eigenval, eigenvec = np.linalg.eigh(covariance)

	sigma1 = find_ellipse(posterior_stats['mean'].flatten(), eigenval, eigenvec, 1)
	sigma2 = find_ellipse(posterior_stats['mean'].flatten(), eigenval, eigenvec, 2)
	sigma3 = find_ellipse(posterior_stats['mean'].flatten(), eigenval, eigenvec, 3)

	ellipse1 = Ellipse(xy=sigma1['mean'], width=sigma1['width'], height=sigma1['height'], angle=sigma1['angle'], visible=True)	
	ellipse2 = Ellipse(xy=sigma2['mean'], width=sigma2['width'], height=sigma2['height'], angle=sigma2['angle'], visible=True)	
	ellipse3 = Ellipse(xy=sigma3['mean'], width=sigma3['width'], height=sigma3['height'], angle=sigma3['angle'], visible=True)	

	plt.figure()
	ax = plt.gca()
	ax.add_patch(ellipse3)
	ax.add_patch(ellipse2)
	ax.add_patch(ellipse1)
	ax.set_xlim(-0.4, 0.4)
	ax.set_ylim(0.5, 2.0)
	plt.show()

	print('\n angle')
	print(sigma1['angle'])


def main():
	measurement_uncertainty = 0.1
	posterior_stats = setup(measurement_uncertainty)
	print('analytical mean:')
	print(posterior_stats['mean'])

	analytical_contours(posterior_stats)
	#mcmc, acceptance_ratio = metropolis_hastings(posterior_stats)
	#print('mcmc mean:')
	#print(mcmc['mean'])
	#print('acceptance ratio:')
	#print(acceptance_ratio)
	#print('mcmc sample examples:')
	#print(mcmc['samples'].shape)

	#plot_samples(mcmc, 200)
	#plot_marginalized(mcmc, 200)


if __name__ == '__main__':
	main()
