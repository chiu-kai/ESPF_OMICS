#see_memory_usage
import psutil
import subprocess
# Function to get VRAM usage and percentage
def get_vram_usage():
    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,nounits,noheader'], stdout=subprocess.PIPE)
    output = result.stdout.decode().strip().split('\n')
    used_memory, total_memory = map(int, output[0].split(','))
    vram_usage_mb = used_memory  
    vram_percent = (used_memory / total_memory) * 100
    return vram_usage_mb, vram_percent
# Get VRAM usage and percentage
vram_usage_mb, vram_percent = get_vram_usage()
# Get RAM usage and percentage
ram_usage_mb = psutil.virtual_memory().used/(1024*1024)# Convert to MB
ram_percent = psutil.virtual_memory().percent

print("RAM Usage:", ram_usage_mb, "MB")
print("RAM Usage Percentage:", ram_percent, "%")
print("VRAM Usage:", vram_usage_mb, "MB")
print("VRAM Usage Percentage:", vram_percent, "%")