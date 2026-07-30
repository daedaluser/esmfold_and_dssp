import torch

# 加载模型文件
model_path = "/home/liaoge/.cache/torch/hub/checkpoints/esmfold_3B_v1.pt"
state = torch.load(model_path, map_location='cpu')

# 检查 IPA 相关键名
ipa_keys = [k for k in state['model'].keys() if 'ipa.linear_q_points' in k or 'ipa.linear_kv_points' in k]
print("=== 模型文件中的实际键名 ===")
for k in ipa_keys:
    print(f"  {k}")

# 检查 ESM 库期望的键名
print("\n=== ESM 库期望的键名 ===")
expected_keys = [
    "trunk.structure_module.ipa.linear_q_points.linear.weight",
    "trunk.structure_module.ipa.linear_q_points.linear.bias",
    "trunk.structure_module.ipa.linear_kv_points.linear.weight",
    "trunk.structure_module.ipa.linear_kv_points.linear.bias",
]
for k in expected_keys:
    print(f"  {k} -> {'✓ 存在' if k in state['model'] else '✗ 缺失'}")
