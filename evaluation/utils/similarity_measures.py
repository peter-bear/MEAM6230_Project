import numpy as np
import similaritymeasures as sm
from dtaidistance import dtw as dtw_lib

def get_RMSE(sim_trajectories, demos, eval_indexes=None, verbose=True):
    RMSE = []
    for k in range(sim_trajectories.shape[1]):
        if verbose:
            print('Calculating RMSE; trajectory:', k + 1)
        if eval_indexes is not None:
            RMSE.append(
                np.sqrt(np.mean((sim_trajectories[eval_indexes[k], k, :] - demos[eval_indexes[k], k, :]) ** 2)))
        else:
            RMSE.append(np.sqrt(np.mean((sim_trajectories[:, k, :] - demos[:, k, :]) ** 2)))
    return RMSE

'''
def get_DTWD(sim_trajectories, demos, eval_indexes, verbose=True):
    DTWD = []
    for k in range(sim_trajectories.shape[1]):
        if verbose:
            print('Calculating dynamic time warping distance; trajectory:', k + 1)
        dtw, d = sm.dtw(sim_trajectories[eval_indexes[k], k, :], demos[eval_indexes[k], k, :])
        DTWD.append(dtw / len(eval_indexes[k]))

    return DTWD
'''
def get_DTWD(sim_trajectories, demos, sim_indexes, demo_indexes, verbose=True):
    DTWD = []
    for k in range(sim_trajectories.shape[1]):
        if verbose:
            print('Calculating dynamic time warping distance; trajectory:', k + 1)
        
        # 获取不等长的完整轨迹 (N x 2)
        traj_k = sim_trajectories[sim_indexes[k], k, :]  # 模拟轨迹 (例如 2000步)
        demo_k = demos[demo_indexes[k], k, :]            # 演示轨迹 (例如 1000步)
        
        print(f'轨迹{k}对齐: 模拟轨迹 {len(sim_indexes[k])} 步 <--> 演示轨迹 {len(demo_indexes[k])} 步')
        
        # 使用 similaritymeasures 计算多维 DTW (和 MATLAB 算法逻辑一致)
        # dtw_dist 是总距离， d 是对齐矩阵
        dtw_dist, _ = sm.dtw(traj_k, demo_k)
        
        DTWD.append(dtw_dist)
    
    return DTWD

def get_FD(sim_trajectories, demos, eval_indexes, verbose=True):
    FD = []
    for k in range(sim_trajectories.shape[1]):
        if verbose:
            print('Calculating Frechet distance; trajectory:', k + 1)
        FD.append(sm.frechet_dist(sim_trajectories[eval_indexes[k], k, :], demos[eval_indexes[k], k, :]))

    return FD