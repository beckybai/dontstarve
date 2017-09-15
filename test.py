# 1. find the accordingly words "env.step"
# 2. find&build my controller.
# 3. Does the origin model count the body impair into the reward? (check github) 
# 4. range & iteration. How to judge the failure.


'''
 If the reward is the distance, every step has a reward ?
 Can we know each part of the reward ?
'''

from osim.env import RunEnv

env = RunEnv(visualize=True)
observation = env.reset(difficulty=0)

for j in range(3):
	print("-------")
	for i in range(2):
		eas = env.action_space.sample()
		o,r,d,i = env.step(eas)
		print("eas")
		print(eas)
		print("o")
		print(o)
		print('r')
		print(r)
		print('d')
		print(d)
		print('i')
		print(i)
