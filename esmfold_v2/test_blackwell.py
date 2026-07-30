import torch
import esm

# 强制 GPU
assert torch.cuda.is_available(), "CUDA不可用"
device = torch.device("cuda")
print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
print(f"✓ Arch: {torch.cuda.get_device_capability(0)}")

# 加载模型
print("加载 ESMFold...")
model = esm.pretrained.esmfold_v1()
model = model.eval().to(device)
print("✓ 模型加载成功在GPU")

# 测试序列
sequence = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"

# 预测
print("预测结构中...")
with torch.no_grad():
    pdb_string = model.infer_pdb(sequence)

# 保存
with open("blackwell_test.pdb", "w") as f:
    f.write(pdb_string)
print("✓ PDB 生成成功")

# 检查显存
print(f"GPU 显存使用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
