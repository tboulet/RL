import numpy as np
import random as rd

import torch
        
#Memory using lists    
class Memory():
    '''Memory buffer object used for saving transition and sampling them later with various methods.'''
    
    def __init__(self, MEMORY_KEYS: list, max_memory_len: int=float('inf')):
        '''
        MEMORY_KEYS : a list of string representing names of elements to save such as 'observation', 'reward', ...
        max_memory_len : the maximal size of the buffer (in transition or steps), default unlimited.
        '''
        self.max_memory_len = max_memory_len
        self.MEMORY_KEYS = MEMORY_KEYS        
        self.trajectory = {key : list() for key in MEMORY_KEYS}

    def remember(self, transition: tuple):
        '''Memorizes a transition and add it to the buffer.
        transition : a tuple of element corresponding to self.MEMORY_KEYS.
        '''
        for val, key in zip(transition, self.MEMORY_KEYS):
            if type(val) == bool: val = int(val)
            self.trajectory[key].append(val)
        if len(self) > self.max_memory_len:
            for val, key in zip(transition, self.MEMORY_KEYS):
                self.trajectory[key].pop()

    def sample(self, sample_size=None, pos_start=None, method='last', as_tensor = True):
        '''Samples several transitions from memory, using different methods.
        sample_size : the number of transitions to sample, default all.
        pos_start : the position in the memory of the first transition sampled, default 0.
        method : the method of sampling in "all", "last", "random", "all_shuffled", "batch_shuffled", "batch".
        return : a list containing a list of size sample_size for each kind of element stored : [observations, rewards, ...]
        '''
        if sample_size is None:
            sample_size = len(self)
        else:
            sample_size = min(sample_size, len(self))
                    
        if method == 'all':
            #Each elements in order.
            indexes = [n for n in range(len(self))]

        elif method == 'last':
            #sample_size last elements in order.
            indexes = [n for n in range(len(self) - sample_size, len(self))]

        elif method == 'random':
            #sample_size elements sampled.
            indexes = [rd.randint(0, len(self) - 1) for _ in range(sample_size)]

        elif method == 'all_shuffled':
            #Each elements suffled.
            indexes = [n for n in range(len(self))]
            rd.shuffle(indexes)

        elif method == "batch":
            #Element n° pos_start and sample_size next elements, in order.
            indexes = [pos_start + n for n in range(sample_size)]
            
        elif method == 'batch_shuffled':
            #Element n° pos_start and sample_size next elements, shuffled.
            indexes = [pos_start + n for n in range(sample_size)]
            rd.shuffle(indexes)
        
        else:
            raise NotImplementedError('Not implemented sample')

        trajectory = list()
        for elements in self.trajectory.values():
            sampled_elements = np.array([elements[idx] for idx in indexes])
            if len(sampled_elements.shape) == 1:
                sampled_elements = np.expand_dims(sampled_elements, -1)
            if as_tensor:
                sampled_elements = torch.Tensor(sampled_elements)                
            trajectory.append(sampled_elements)

        return trajectory

    def __len__(self):
        return len(self.trajectory[self.MEMORY_KEYS[0]])

    def __empty__(self):
        self.trajectory = {key : list() for key in self.MEMORY_KEYS}
        


class Memory_episodic():
    '''A memory made for offline learning that save elements grouped in episode rather than in transition.
    Main difference is that .sample() return a list of lists [observations, rewards, ...].
    Also remember with self.done whether we just terminated an episode.
    '''
    
    def __init__(self, MEMORY_KEYS: list, max_memory_len: int=float('inf')):
        '''
        MEMORY_KEYS : a list of string representing names of elements to save such as 'observation', 'reward', ...
        max_memory_len : the maximal size (in episode) of the buffer, default unlimited.
        '''
        self.max_memory_len = max_memory_len
        self.MEMORY_KEYS = MEMORY_KEYS
        
        self.memory_trajectory = Memory(MEMORY_KEYS=MEMORY_KEYS)     
        self.episodes = list()
        self.done = False
        self.done_index = MEMORY_KEYS.index('done')

    def remember(self, transition: tuple):
        '''Memorizes a transition and add it to the memory of the current episode.
        transition : a tuple of element corresponding to self.MEMORY_KEYS.
        '''
        #Add transition to trajectory and remember whether we terminated an episode
        self.memory_trajectory.remember(transition)
        self.done = transition[self.done_index]
        if self.done:
            #Save the episode in memory, then empty trajectory memory
            episode = self.memory_trajectory.sample(method = 'all', as_tensor=True)
            self.episodes.append(episode)
            self.memory_trajectory.__empty__()
            #Get rid of oldest episode if buffer is full
            if len(self) > self.max_memory_len:
                self.episodes.pop(0)

    def sample(self, sample_size=None, pos_start=None, method='last'):
        '''Samples several episodes from memory, using different methods.
        sample_size : the number of episodes to sample, default all.
        pos_start : the position in the memory of the first episodes sampled, default 0.
        method : the method of sampling in "all", "last", "random", "all_shuffled", "batch_shuffled", "batch".
        return : a list containing episodes, which are lists of elements : [[observs, rewards, ...], [observs, rewards], ...]
        '''
        if sample_size is None:
            sample_size = len(self)
        else:
            sample_size = min(sample_size, len(self))
                    
        if method == 'all':
            #Each elements in order.
            indexes = [n for n in range(len(self))]

        elif method == 'last':
            #sample_size last elements in order.
            indexes = [n for n in range(len(self) - sample_size, len(self))]

        elif method == 'random':
            #sample_size elements sampled.
            indexes = [rd.randint(0, len(self) - 1) for _ in range(sample_size)]

        elif method == 'all_shuffled':
            #Each elements suffled.
            indexes = [n for n in range(len(self))]
            rd.shuffle(indexes)

        elif method == "batch":
            #Element n° pos_start and sample_size next elements, in order.
            indexes = [pos_start + n for n in range(sample_size)]
            
        elif method == 'batch_shuffled':
            #Element n° pos_start and sample_size next elements, shuffled.
            indexes = [pos_start + n for n in range(sample_size)]
            rd.shuffle(indexes)

        else:
            raise NotImplementedError('Not implemented sample')

        return [self.episodes[idx] for idx in indexes]

    def __len__(self):
        return len(self.episodes)

    def __empty__(self):
        self.episodes = list()