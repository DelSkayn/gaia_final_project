#!/usr/bin/env python
# coding: utf-8

# # Hyper parameters

# In[1]:


METHOD = "DQN" # or "PPO"

TRAIN_LENGHT = 2_000_000
EVALUATION_GAMES = 2_000
NUM_ENVS = 1
POLICY = "CNN" # or "MLP"


# If true shows an agent 4 frames instead of 1
# Allows an agent to determin velocity and movement direction when learning from pixels
# Not a default hyper-param but should probably be used
FRAME_STACK = True 

# If true a episode is reset apon losing a life instead of losing all lifes
# Used in the deepmind papers
END_OF_LIFE_IS_END_OF_GAME = False

ENV_SEED = 1

# Disables the effect of the next hyper parameters
DEFAULT_HYPER_PARAMS = True

LEARNING_RATE = 0.00001

DQN_BUFFER_SIZE = 200_000
# Number of frames to simulate between training the network
DQN_TRAIN_FREQ = 2
# Number of initial random frames
DQN_LEARNING_STARTS = 10_000
# Number of frames between updating the target network 
DQN_TARGET_NETWORK_UPDATE_FREQ = 10_000

MODEL_NAME = "model_{}_{}_{}".format(METHOD,POLICY,TRAIN_LENGHT)

# Load a model from a file of name MODEL_NAME
LOAD_MODEL = False
LOAD_NAME = MODEL_NAME + "_checkpoint"

EVAL_FREQUENCY = 10_000
EVAL_EPISODES = 10


# In[2]:


# Setting up the environment if on colab
try:
    import google.colab
    import os
    get_ipython().run_line_magic('tensorflow_version', '1.15')
    IN_COLAB = True
except:
    IN_COLAB = False

if IN_COLAB:
    get_ipython().system('pip install stable-baselines gym[atari] tqdm')
    colab.drive.mount('/content/drive')
    os.chdir('/content/drive/My Drive/CHANGE_THIS')


# In[9]:


from stable_baselines.deepq.policies import MlpPolicy as DQNMlpPolicy, CnnPolicy as DQNCnnPolicy
from stable_baselines.common.policies import MlpPolicy, CnnPolicy 
from stable_baselines.common.cmd_util import make_atari_env
from stable_baselines.common import callbacks
from stable_baselines import DQN, PPO2
from tqdm.notebook import tqdm


if METHOD == "DQN":
    NUM_ENVS = 1

wrapper_kwargs = {
    'frame_stack': FRAME_STACK,
    'episode_life': END_OF_LIFE_IS_END_OF_GAME
}
env = make_atari_env("MsPacmanNoFrameskip-v0",NUM_ENVS,ENV_SEED,wrapper_kwargs=wrapper_kwargs)

eval_env = make_atari_env("MsPacmanNoFrameskip-v0",1,ENV_SEED+10,wrapper_kwargs={'frame_stack':FRAME_STACK})


class EvalScoreLogger(callbacks.EvalCallback):
    def __init__(self,path,*args, **kwargs):
        super(EvalScoreLogger, self).__init__(*args,**kwargs)
        self.file_path = path
    
    def _on_training_start(self):
        self.log_file = open(self.file_path,"w")
        
    def _on_step(self):
        super(EvalScoreLogger,self)._on_step()
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            time_steps = self.n_calls
            self.log_file.write("{},{}\n".format(time_steps,self.last_mean_reward))
            self.log_file.flush()
        
    def _on_training_end(self):
        self.log_file.close()
        

class PBarCallback(callbacks.BaseCallback):
    def __init__(self,verbose=0):
        super(PBarCallback, self).__init__(verbose)
        self.pbar = None
        
    def _on_training_start(self):
        self.pbar = tqdm(total=self.locals['total_timesteps'])
        
    def _on_step(self):
        self.pbar.n = self.n_calls
        self.pbar.update(0)
        return True
        
    def _on_training_end(self):
        self.pbar.n = self.n_calls
        self.pbar.update(0)
        self.pbar.close()
        
#eval_log = open("eval_log.csv","w")

cb = callbacks.CallbackList([
    PBarCallback(TRAIN_LENGHT),
    callbacks.CheckpointCallback(save_freq=10_000,save_path="./checkpoints/",name_prefix=MODEL_NAME),
    EvalScoreLogger('eval_score.csv',eval_env,
                           eval_freq=EVAL_FREQUENCY,
                           n_eval_episodes = EVAL_EPISODES,
                           best_model_save_path="./best/",
                           verbose=1)
])

if METHOD == "DQN":
    policy = DQNMlpPolicy if POLICY == "MLP" else DQNCnnPolicy
    params = {} if DEFAULT_HYPER_PARAMS else {
        'learning_rate': LEARNING_RATE,
        'buffer_size': DQN_BUFFER_SIZE,
        'train_freq': DQN_TRAIN_FREQ,
        'learning_starts': DQN_LEARNING_STARTS,
        'target_network_update_freq': DQN_TARGET_NETWORK_UPDATE_FREQ,
    }
    model = DQN(policy,env,**params)
else:
    policy = MlpPolicy if POLICY == "MLP" else CnnPolicy
    params = {} if DEFAULT_HYPER_PARAMS else {
        'learning_rate': LEARNING_RATE,
    }
    model = PPO2(policy,env,**params)
    
if LOAD_MODEL:
    model.load(LOAD_NAME,env)


# In[ ]:


model.learn(TRAIN_LENGHT,callback=cb)
model.save(MODEL_NAME)


# In[ ]:




