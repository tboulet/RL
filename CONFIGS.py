DQN_CONFIG = {"name" : "DQN",
    }

REINFORCE_CONFIG = {"name" : "REINFORCE",
    "learning_rate" : 1e-3,
    "gamma" : 0.99,
    "frames_skipped" : 1,
    "reward_scaler" : 100,
    "batch_size" : 1, #TO IMPLEMENT
    "off_policy" : True,
    }

ACTOR_CRITIC_CONFIG = {"name" : "ACTOR_CRITIC",
    "learning_rate_actor" : 1e-4,
    "learning_rate_critic" : 1e-3,
    "compute_gain_method" : "total_future_reward_minus_state_value",
    "gamma" : 0.99,     
    "reward_scaler" : 100,
    "update_ratio" : 1,      #TO IMPLEMENT   #Algorithm updates update_ratio more often the critic than the policy.
    "batch_size" : 1,        #TO IMPLEMENT
    "gradient_steps_critic" : 1,
    "gradient_steps_policy" : 1,
    "frames_skipped" : 1,
    "clipping" : None,
    }

DUMMY_CONFIG = dict()
