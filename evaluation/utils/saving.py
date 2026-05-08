import torch
from pathlib import Path


def save_best_stats_txt(save_path, best_n_spurious, best_metric, best_RMSE, best_DTWD, best_FD, best_DTWD_std, gpu_status, i):
    output_path = Path(save_path) / 'best_model.txt'
    with output_path.open('w', encoding='utf-8') as file_obj:
        file_obj.write('Number of unsuccessful trajectories: ' + str(best_n_spurious) + '\n')
        file_obj.write('RMSE + DTWD + FD: ' + str(best_metric) + '\n')
        file_obj.write('RMSE: ' + str(best_RMSE) + '\n')
        file_obj.write('DTWD: %.4f +/- %.4f\n' % (best_DTWD, best_DTWD_std))
        file_obj.write('FD: ' + str(best_FD) + '\n')
        file_obj.write('Iteration number: ' + str(i) + '\n')
        file_obj.write('\n\n ###### GPU information ###### \n')
        file_obj.write(gpu_status)


def save_stats_txt(save_path, best_n_spurious, best_metric, best_RMSE, best_DTWD, best_FD, i):
    output_path = Path(save_path) / 'training_evaluation_summary.txt'
    with output_path.open('a', encoding='utf-8') as file_obj:
        file_obj.write('Iteration number: ' + str(i) + '\n')
        file_obj.write('Number of unsuccessful trajectories: ' + str(best_n_spurious) + '\n')
        file_obj.write('RMSE + DTWD + FD: ' + str(best_metric) + '\n')
        file_obj.write('RMSE: ' + str(best_RMSE) + '\n')
        file_obj.write('DTWD: ' + str(best_DTWD) + '\n')
        file_obj.write('FD: ' + str(best_FD) + '\n\n')


def check_gpu():
    # setting device on GPU if available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gpu_status = ''
    gpu_status += '\nUsing device: ' + str(device) + '\n\n'

    # Additional Info when using cuda
    if device.type == 'cuda':
        gpu_status += torch.cuda.get_device_name(0) + '\n'
        gpu_status += 'Memory Usage:\n'
        gpu_status += 'Allocated: ' + str(round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1)) + ' GB\n'
        gpu_status += 'Cached:    ' + str(round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1)) + ' GB\n'

    return gpu_status