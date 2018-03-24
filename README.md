# Reinforcement-Learning-Q-learning-Gridworld-Pytorch
This is a project using Pytorch to fulfill reinforcement learning on a simple game - Gridworld.

The basic introduction of this game is at:

http://outlace.com/rlpart3.html

Also, please refer to the Pytorch tutorial on Reinforcement Learning:

http://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

Most of the game code and test code are copied from the game website. What I do is to use Pytorch rather than Keras to implemet the neural network of Q learning. Also, I have made some changes to make the code more "Pythonic". For instance, I replace the for loop in the experience replay to the vector calculation. This modification can speed up the running by parallel processing.

If you are interested in using Pytorch to create some programs, this game can be a good practice.


Simply run the `main.py`. The program will train the network and test it. The implementation of game is at `gridworld.py` and the implementation of Q learning is at `DQN.py`. Both of them have been imported into `main.py` already.
