import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

import sys
import math
import random
import os
from moviepy.editor import *
import matplotlib.pyplot as plt
import numpy as np

import gym
from gym.wrappers import Monitor, RecordVideo
import wandb

from config import project, entity

def run(agent, env, episodes, wandb_cb = True, plt_cb = True, video_cb = True):
    print("Run starts.")
    
    config = agent.config
    if wandb_cb: 
        run = wandb.init(project="RL", 
                        entity="timotheboulet",
                        config=config
                        )
    if video_cb:
        n_step_save_video = 10000
        videos_path = "./div/videos/rl-video/"
        env = RecordVideo(env, video_folder=videos_path, step_trigger=lambda step: step % n_step_save_video == 0)
    if plt_cb:
        logs = dict()
        
    k = 0
    for episode in range(1, episodes+1):
        k += 1
        done = False
        obs = env.reset()
        
        while not done:
            action = agent.act(obs)
            next_obs, reward, done, info = env.step(action)
            # print(obs, action, reward, done)
            metrics1 = agent.remember(obs, action, reward, done, next_obs, info)
            metrics2 = agent.learn()
            
            if video_cb and wandb_cb:
                if agent.step % n_step_save_video == 0:
                    path = f'./div/videos/rl-video/rl-video-step-{agent.step}.mp4'
                    wandb.log({"gameplay": wandb.Video(path, fps = 4, format="gif")}, step = agent.step)

            #Feedback
            for metric in metrics1 + metrics2:
                if wandb_cb:
                    wandb.log(metric, step = agent.step)
            
                if plt_cb:
                    for key, value in metric.items():
                        try:
                            logs[key]["steps"].append(agent.step)
                            logs[key]["values"].append(value)    
                        except KeyError:
                            logs[key] = {"steps": [agent.step], "values": [value]}      
                        plt.clf()         
                        plt.plot(logs[key]["steps"][-100:], logs[key]["values"][-100:], '-b')
                        plt.title(key)
                        plt.savefig(f"figures/{key}")
                    
            #If episode ended.
            if done:
                break
            else:
                obs = next_obs
    
    if wandb_cb: run.finish()
    print("End of run.")