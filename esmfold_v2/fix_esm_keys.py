import torch

# 加载模型
model_path = "/home/liaoge/.cache/torch/hub/checkpoints/esmfold_3B_v1.pt"
backup_path = "/home/liaoge/.cache/torch/hub/checkpoints/esmfold_3B_v1.backup.pt"

print("备份原始模型...")
state = torch.load(model_path, map_location='cpu')
torch.save(state, backup_path)

print("修复键名...")
new_model_state = {}

for key, value in state['model'].items():
    # 添加缺失的 .linear. 层级
    if 'ipa.linear_q_points.' in key and '.linear.' not in key:
        new_key = key.replace('ipa.linear_q_points.', 'ipa.linear_q_points.linear.')
        new_model_state[new_key] = value
    elif 'ipa.linear_kv_points.' in key and '.linear.' not in key:
        new_key = key.replace('ipa.linear_kv_points.', 'ipa.linear_kv_points.linear.')
        new_model_state[new_key] = value
    else:
        new_model_state[key] = value

state['model'] = new_model_state

# 保存修复后的模型
torch.save(state, model_path)
print(f"✓ 修复完成！原始文件已备份: {backup_path}")
