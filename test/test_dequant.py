import torch

K = 3584
N = 3584
group_size = 128
pack_num = 8

qweight = torch.randint(0, 2**31 - 1, (K, N // 8), dtype=torch.int32)
qzeros = torch.randint(0, 2**31 - 1, (K // group_size, N // 8), dtype=torch.int32)
scales = torch.randn(K // group_size, N, dtype=torch.float16)

# 1. Promote qweight to allow shifting correctly
qw = qweight.view(K, N // 8, 1).expand(K, N // 8, pack_num) # (K, N // 8, 8)

# 2. Extract 4 bits
shifts = torch.arange(0, 32, 4, dtype=torch.int32).view(1, 1, pack_num)
w_fp = ((qw >> shifts) & 0xF).flatten(1, 2).view(K, N).to(torch.float16)

print("w_fp shape:", w_fp.shape)

qz = qzeros.view(K // group_size, N // 8, 1).expand(K // group_size, N // 8, pack_num)
z_fp = ((qz >> shifts) & 0xF).flatten(1, 2).view(K // group_size, N).to(torch.float16)
print("z_fp shape:", z_fp.shape)

w_fp = w_fp.view(K // group_size, group_size, N)
z_fp = z_fp.view(K // group_size, 1, N)
s_fp = scales.view(K // group_size, 1, N)

w_fp = (w_fp - z_fp) * s_fp
w_fp = w_fp.view(K, N)
print("final w_fp shape:", w_fp.shape)
