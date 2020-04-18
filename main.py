from DQN import ReplayMemory, Transition, hidden_unit, Q_learning
from torch.autograd import Variable
from gridworld import *
import torch.optim as optim
import torch

## Include the replay experience

epochs = 1000
gamma = 0.9 #since it may take several moves to goal, making gamma high
epsilon = 1
model = Q_learning(64, [150,150], 4, hidden_unit)
optimizer = optim.RMSprop(model.parameters(), lr = 1e-2)
# optimizer = optim.SGD(model.parameters(), lr = 0.1, momentum = 0)
criterion = torch.nn.MSELoss()
buffer = 80
BATCH_SIZE = 40
memory = ReplayMemory(buffer)   

for i in range(epochs):
    state = initGridPlayer()
    status = 1
    step = 0
    #while game still in progress
    while(status == 1):   
        v_state = Variable(torch.from_numpy(state)).view(1,-1)
        qval = model(v_state)
        if (np.random.random() < epsilon): #choose random action
            action = np.random.randint(0,4)
        else: #choose best action from Q(s,a) values
            action = np.argmax(qval.data)
        #Take action, observe new state S'
        new_state = makeMove(state, action)
        step +=1
        v_new_state = Variable(torch.from_numpy(new_state)).view(1,-1)
        #Observe reward
        reward = getReward(new_state)
        memory.push(v_state.data, action, v_new_state.data, reward)
        if (len(memory) < buffer): #if buffer not filled, add to it
            state = new_state
            if reward != -1: #if reached terminal state, update game status
                break
            else:
                continue
        transitions = memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))
        state_batch = Variable(torch.cat(batch.state))
        action_batch = Variable(torch.LongTensor(batch.action)).view(-1,1)
        new_state_batch = Variable(torch.cat(batch.new_state))
        reward_batch = Variable(torch.FloatTensor(batch.reward))
        non_final_mask = (reward_batch == -1)
        #Let's run our Q function on S to get Q values for all possible actions
        qval_batch = model(state_batch)
        # we only grad descent on the qval[action], leaving qval[not action] unchanged
        state_action_values = qval_batch.gather(1, action_batch)
        #Get max_Q(S',a)
        with torch.no_grad():
            newQ = model(new_state_batch)
        maxQ = newQ.max(1)[0]
#         if reward == -1: #non-terminal state
#             update = (reward + (gamma * maxQ))
#         else: #terminal state
#             update = reward + 0*maxQ
#         y = reward_batch + (reward_batch == -1).float() * gamma *maxQ
        y = reward_batch
        y[non_final_mask] += gamma * maxQ[non_final_mask]
        y = y.view(-1,1)
        print("Game #: %s" % (i,), end='\r')
        loss = criterion(state_action_values, y)
        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        for p in model.parameters():
            p.grad.data.clamp_(-1, 1)
        optimizer.step()
        state = new_state
        if reward != -1:
            status = 0
        if step >20:
            break
    if epsilon > 0.1:
        epsilon -= (1/epochs)

## Here is the test of AI
def testAlgo(init=0):
    i = 0
    if init==0:
        state = initGrid()
    elif init==1:
        state = initGridPlayer()
    elif init==2:
        state = initGridRand()

    print("Initial State:")
    print(dispGrid(state))
    status = 1
    #while game still in progress
    while(status == 1):
        v_state = Variable(torch.from_numpy(state))
        qval = model(v_state.view(64))
        print(qval)
        action = np.argmax(qval.data) #take action with highest Q-value
        print('Move #: %s; Taking action: %s' % (i, action))
        state = makeMove(state, action)
        print(dispGrid(state))
        reward = getReward(state)
        if reward != -1:
            status = 0
            print("Reward: %s" % (reward,))
        i += 1 #If we're taking more than 10 actions, just stop, we probably can't win this game
        if (i > 10):
            print("Game lost; too many moves.")
            break


testAlgo(init=1)
