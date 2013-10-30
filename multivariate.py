import numpy as np

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
	P = 0.1 * np.eye(2)
	
	return P

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
	
def metropolis_hastings(posterior_stats):
	iterations = 10000
	thetas, proposal_stdev = initialise()
	ln_posterior = calculate_ln_posterior(thetas, posterior_stats)
	accepts = 0
	mcmc_samples = []

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

		mcmc_samples.append(thetas)

	acceptance_ratio = accepts / iterations
		
	return mcmc_samples, acceptance_ratio
	

def main():
	measurement_uncertainty = 0.1
	posterior_stats = setup(measurement_uncertainty)
	print(posterior_stats['mean'])

	mcmc_samples, acceptance_ratio = metropolis_hastings(posterior_stats)

	plot()

if __name__ == '__main__':
	main()
