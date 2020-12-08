import math

def spherical_fn(x):
	total = 0

	for i in range(len(x)):
		total += x[i]**2
	return total

def rastrigin_fn(x):
	total = 0

	for i in range(len(x)):
		total += ( x[i]**2 - 10 * (math.cos(2 * math.pi * x[i])) + 10 )
	return total

def griewank_fn(x):
	total = 0
	total1 = 0
	total2 = 1

	for i in range(len(x)):
		total1 += ( (x[i]**2 / 4000) )

	for i in range(len(x)):
		total2 *= ( math.cos(x[i] / math.sqrt(i + 1)) )

	total = 1 + total1 - total2
	return total

def rosenbrock_fn(x):
	total = 0

	for i in range(len(x) - 1):
		total += ( 100 * ((x[i + 1] - x[i]**2)**2) + (x[i] - 1)**2 )
	return total