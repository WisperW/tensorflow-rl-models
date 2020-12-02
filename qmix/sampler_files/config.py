class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

experiment_config_dict = {
    'a_dim':12, # 动作空间的维度
    's1_dim':228, # 向量输入的维度
    's2_dim':12, # action mask, 维度与a_dim相同
    'rnn_dim':512,
    'sampler_buf_size':5000, # sampler端每次发送样本的最大长度
    'image_shape':[72, 48, 3], # 图像输入的shape
    'use_bn': True, 
    'use_dropout':False,
    'num_agent':2, 
    'rescue_mouse_num':2, 
    'traj_length':128,
    'lstm_dim' :1024, 
    'hidden_dim':1024, 
    'replay_buffer_size':3000,
    'weight_decay':0.00000001, 
    'is_soft_update' : False,
    'algorithm':'q-learning',
    'start_game_id':32111, # anticheat的启动id, 每个sampler的启动id从start_game_id开始依次递增, 不同实验之间注意增大id的差距
    'experiment_name': 'qmix_t19', # 实验的名字，千万要唯一!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
}

learner_config_dict = {
    'num_collect_sample_process' : 3, 
    'init_lr' : False, # 是否重置lr
}

train_config_dict = {
    'gpu_list':[0, 1, 2, 3], # gpu列表
    'single_gpu_batch_size':32, # 单卡gpu的size
    'each_modelversion_update_train_count':10, # 每训练N个batchsize, 更新一次模型
    'sample_keys':['image_ph', 'obs1_ph', 'obs2_ph', 'next_image_ph', 'next_obs1_ph', 'next_obs2_ph', 'acts', 'rews', 'done', 'td', 'new_traj_start'], # placeholders
    'single_batch_train_count':5, # 每个batchsize训练的次数
}

learning_rate_config_dict = {
    'start_lr' : 5e-5, # 初始lr
    'init_lr' : False, # 是否重置lr
    'restart_lr' : 1e-6,  # 重置后的lr
    'lr_decay_rate':0.95, # lr衰减的比例
    'minimum_lr':1e-6, # lr最小值
    'update_lr_count':20, # 每更新N次模型，更新一次学习率
    'start_lr_decay':50, # 开始学习率衰减的modelversion
    'stop_lr_decay':2000, # 停止学习率衰减的modelversion
}


restart_config_dict = {
    'load_model' : False, # 是否读取本地模型
    'load_best_model' : False, # 是否读取最佳模型
    'explore_ratio' : 0.99,
    'best_win_rate' : 0.35, # 若读取最佳模型，需根据最佳模型的最大值来读取模型
    'modelversion' : 0, # 迭代次数, 默认为0, 若需重新绘制visdom, 请保持大于visdom中图像x轴最大值
    'visdom_new_start' : True, # 是否重新绘制visdom
}

p2p_config_dict = {
    'dragonfly_node' : '10.211.200.3', # 杜鹃的超级节点列表
    'port' : 6029, # grpc 通信端口
    'p2p_url' : '10.91.135.206:6020', # 前面是gpu的ip，要改成自己的，后面的6020千万不能改
}



experiment_config = Struct(**experiment_config_dict)
train_config = Struct(**train_config_dict)
restart_config = Struct(**restart_config_dict)
p2p_config = Struct(**p2p_config_dict)
learning_rate_config = Struct(**learning_rate_config_dict)
learner_config = Struct(**learner_config_dict)
