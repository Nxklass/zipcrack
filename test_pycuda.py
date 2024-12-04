import pycuda.driver as cuda
import pycuda.autoinit

print(cuda.Device(0).name())  # Gibt den Namen der ersten GPU aus