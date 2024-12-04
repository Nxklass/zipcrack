import pycuda.driver as cuda
import pycuda.autoinit

print("CUDA Driver Version:", cuda.get_version())
print("CUDA Device Count:", cuda.Device.count())
for i in range(cuda.Device.count()):
    dev = cuda.Device(i)
    print(f"Device {i}: {dev.name()} - {dev.total_memory() // (1024**2)} MB")