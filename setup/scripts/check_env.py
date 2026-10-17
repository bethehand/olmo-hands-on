import time, torch
print("torch", torch.__version__, "| built for CUDA", torch.version.cuda, "| cuda available:", torch.cuda.is_available(), "| GPUs:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print("  GPU", i, torch.cuda.get_device_name(i))
x = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
torch.cuda.synchronize(); t = time.time()
for _ in range(20):
    y = x @ x
torch.cuda.synchronize(); dt = time.time() - t
print(f"bf16 matmul OK: {20 * 2 * 8192**3 / dt / 1e12:.0f} TFLOPS measured on GPU 0")
import vllm, deepspeed, flash_attn
print("vllm", vllm.__version__, "| deepspeed", deepspeed.__version__, "| flash_attn", flash_attn.__version__)
