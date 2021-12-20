import pycuda.autoinit
import pycuda.driver as drv
import numpy as np
from pycuda.compiler import SourceModule
import time
import cv2
from google.colab.patches import cv2_imshow

FILEPATH = 'image1.jpg'

im = cv2.imread(FILEPATH)

cv2_imshow(im)

HEIGHT, WIDTH, _ = im.shape

r = im[:,:,0].copy().astype(np.float32)
g = im[:,:,1].copy().astype(np.float32)
b = im[:,:,2].copy().astype(np.float32)

## **CUDA**

mod = SourceModule("""
__global__ void rgb_to_yuv_cuda(float *y, float *u, float *v, float *r, float *g, float *b)
{
  int idx = blockIdx.x * gridDim.y + blockIdx.y;
  y[idx] =  0.257 * r[idx] + 0.504 * g[idx] + 0.098 * b[idx] +  16;
  u[idx] = -0.148 * r[idx] - 0.291 * g[idx] + 0.439 * b[idx] + 128;
  v[idx] =  0.439 * r[idx] - 0.368 * g[idx] - 0.071 * b[idx] + 128;
}
""")

rgb_to_yuv_cuda = mod.get_function("rgb_to_yuv_cuda")

y_cuda = np.zeros_like(r)
u_cuda = np.zeros_like(g)
v_cuda = np.zeros_like(b)

start = time.perf_counter()
rgb_to_yuv_cuda(
        drv.Out(y_cuda), drv.Out(u_cuda), drv.Out(v_cuda), 
        drv.In(r), drv.In(g), drv.In(b),
        block=(1,1,1), grid=(HEIGHT,WIDTH))
end = time.perf_counter()
print(end-start)

yuv_im = np.stack([y_cuda, u_cuda, v_cuda], axis=-1)

cv2_imshow(yuv_im)

cv2_imshow(y_cuda)
cv2_imshow(u_cuda)
cv2_imshow(v_cuda)

## **SEQUENTIAL**

def rgb_to_yuv_sequential(y, u, v, r, g, b):
  for i in range(HEIGHT):
    for j in range(WIDTH):
      y[i][j] +=  0.257 * r[i][j] + 0.504 * g[i][j] + 0.098 * b[i][j] +  16
      u[i][j] += -0.148 * r[i][j] - 0.291 * g[i][j] + 0.439 * b[i][j] + 128
      v[i][j] +=  0.439 * r[i][j] - 0.368 * g[i][j] - 0.071 * b[i][j] + 128

y_sequential = np.zeros_like(r)
u_sequential = np.zeros_like(g)
v_sequential = np.zeros_like(b)

start = time.perf_counter()
rgb_to_yuv_sequential(y_sequential, u_sequential, v_sequential, r, g, b)
end = time.perf_counter()
print(end-start)
