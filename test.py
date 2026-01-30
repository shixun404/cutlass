f = "build/examples/92_blackwell_moe_gemm/input"
with open(f, 'w') as f:
    for i in range(16):
        f.write(f"{i} 5120x4096x6144\n")