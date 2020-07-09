nvcc -arch=compute_70 -O2 curing.cu -lpthread --extended-lambda -o curing -DCUDA_REGISTER
