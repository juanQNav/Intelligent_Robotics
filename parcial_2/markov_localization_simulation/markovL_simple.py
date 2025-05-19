#   [0, 0] - No move
#	[0, 1] - Right
#	[0, -1] - Left
#	[1, 0] - Down
#	[-1, 0] - Up

world = [['green', 'green', 'green'],
         ['green', 'red', 'green'],
         ['green', 'green', 'green']]

measurements = ['red', 'green']
motions = [[0, 0], [0, 1]]

sensor_right = 0.8
sensor_wrong = 1 - sensor_right

p_move = 0.7
p_stay = 1 - p_move


def sense(p, world, measurement):
	aux = [[0.0 for row in range(len(p[0]))] for col in range(len(p))]

	s = 0.0
	for i in range(len(p)):
		for j in range(len(p[i])):
			hit = (measurement == world[i][j])
			aux[i][j] = p[i][j] * (hit * sensor_right + (1-hit) * sensor_wrong)
			s += aux[i][j]
	for i in range(len(aux)):
		for j in range(len(p[i])):
			aux[i][j] /= s
	return aux


def move(p, motion):
	aux = [[0.0 for row in range(len(p[0]))] for col in range(len(p))]

	for i in range(len(p)):
		for j in range(len(p[i])):
			aux[i][j] = (p_move * p[(i - motion[0]) % len(p)][(j - motion[1]) % len(p[i])]) + (p_stay * p[i][j])
	return aux


def show(p):
	for i in range(len(p)):
		print(p[i])

def show_world(world):
	for i in range(len(world)):
		print(world[i])
	print("-"*10)

if len(measurements) != len(motions):
	raise ValueError("error in size of measurements/motions vector")

pinit = 1.0 / float(len(world)) / float(len(world[0]))
p = [[pinit for row in range(len(world[0]))] for col in range(len(world))]
print("WORLD"+"-"*10)
show_world(world)

print("init p"+"-"*10)
show(p)
for k in range(len(measurements)):
	p = sense(p, world, measurements[k])
	print("after sense "+f"{measurements[k]}"+"-"*10)
	show(p)

	p = move(p, motions[k])
	print("after move in "+f"mes {measurements[k]} move{motions[k]}"+"-"*10)
	show(p)