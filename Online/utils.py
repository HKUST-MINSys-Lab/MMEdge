import os
import torch
import psutil
import numpy as np

# from audio_model import lipreading as audio
# from video_model import lipreading as video
# from concat_model import lipreading as concat



model_config = {
    'audio': {
        'params': {'mode': 'backendGRU', 'inputDim': 512, 'hiddenDim': 512, 'nClasses': 500, 'frameLen': 29, 'every_frame': False},
        'checkpoint': 'Audiovisual_a_part.pt'
    },
    'video': {
        'params': {'mode': 'backendGRU', 'inputDim': 256, 'hiddenDim': 512, 'nClasses': 500, 'frameLen': 29, 'every_frame': False},
        'checkpoint': 'Audiovisual_v_part.pt'
    },
    'concat': {
        'params': {'mode': 'backendGRU', 'inputDim': 2048, 'hiddenDim': 512, 'nLayers': 2, 'nClasses': 500, 'every_frame': False},
        'checkpoint': 'Audiovisual_c_part.pt'
    }
}


def load_model(model_name, model_path, device):
    # 检查传入的模型名称是否在配置中
    if model_name not in model_config:
        raise ValueError(f"Model '{model_name}' not found in the configuration.")

    # 获取模型的参数和对应的检查点文件
    config = model_config[model_name]
    
    # 使用 eval() 动态实例化模型
    model_class = eval(model_name)  # 根据字符串获取类名
    model = model_class(**config['params'])  # 使用参数创建模型实例

    # 加载模型权重
    checkpoint_path = os.path.join(model_path, config['checkpoint'])
    model = reload_model(model, checkpoint_path)
    
    model.to(device)

    return model


def reload_model(model, path=""):
    if not path:
        print('train from scratch')
        return model
    else:
        model_dict = model.state_dict()
        
        pretrained_dict = torch.load(path)

        # 忽略 num_batches_tracked 相关参数
        matched_dict = {k: v for k, v in pretrained_dict.items() 
                        if k in model_dict and v.shape == model_dict[k].shape}

        total_params_in_pth = len(pretrained_dict)
        loaded_params_count = len(matched_dict)
        total_params_in_model = len(model_dict)
        loaded_model_params_count = sum(1 for k in model_dict if k in matched_dict)

        # 找出没有加载的参数
        unloaded_params = [k for k in model_dict if k not in matched_dict]

        # print(f'Total parameters in checkpoint: {total_params_in_pth}')
        # print(f'Number of matching parameters loaded: {loaded_params_count}')
        # print(f'Total parameters in the model: {total_params_in_model}')
        # print(f'Number of model parameters loaded: {loaded_model_params_count}')

        if unloaded_params:
            print(f'Unloaded model parameters ({len(unloaded_params)}):')
            for param in unloaded_params:
                print(f' - {param}')

        if not matched_dict:
            print('No matching parameters found. Training from scratch.')
            return model

        model_dict.update(matched_dict)
        model.load_state_dict(model_dict, strict=False)  # 允许加载不完全匹配
        print(f'*** {loaded_params_count} parameters from checkpoint have been successfully loaded! ***')
        print(f'*** {loaded_model_params_count} model parameters have been updated with checkpoint weights! ***')
        return model
    

def normalisation(inputs):
    inputs = np.array(inputs)
    inputs_std = inputs.std()
    if inputs_std == 0.:
        inputs_std = 1.
    return (inputs - inputs.mean()) / inputs_std


def CenterCrop(batch_img, size):
    w, h = batch_img[0][0].shape[1], batch_img[0][0].shape[0]
    th, tw = size
    img = np.zeros((len(batch_img), len(batch_img[0]), th, tw))
    for i in range(len(batch_img)):
        x1 = int(round((w - tw))/2.)
        y1 = int(round((h - th))/2.)
        img[i] = batch_img[i, :, y1:y1+th, x1:x1+tw]
    return img


def RandomCrop(batch_img, size):
    w, h = batch_img[0][0].shape[1], batch_img[0][0].shape[0]
    th, tw = size
    img = np.zeros((len(batch_img), len(batch_img[0]), th, tw))
    for i in range(len(batch_img)):
        x1 = random.randint(0, 8)
        y1 = random.randint(0, 8)
        img[i] = batch_img[i, :, y1:y1+th, x1:x1+tw]
    return img


def HorizontalFlip(batch_img):
    for i in range(len(batch_img)):
        if random.random() > 0.5:
            for j in range(len(batch_img[i])):
                batch_img[i][j] = cv2.flip(batch_img[i][j], 1)
    return batch_img


def ColorNormalize(batch_img):
    mean = 0.413621
    std = 0.1700239
    batch_img = (batch_img - mean) / std
    return batch_img


def monitor_cpu(name, pid, end_event):
    process = psutil.Process(pid)
    cpu_usages = []
    while not end_event.is_set():
        cpu_usage = process.cpu_percent(interval=1)
        cpu_usages.append(cpu_usage)
    avg_cpu_usage = sum(cpu_usages) / len(cpu_usages)
    print(f"{name} CPU Usage: {avg_cpu_usage:.2f}")
    
    
def average(l):
    return sum(l) / len(l) if l else 0.0