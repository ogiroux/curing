# curing

Build with `./make.sh`. You need to have CUDA 10.2 or better, Linux 5.3 or better.

Create the 1GB test file with `./create_file.sh`. ***Remember to delete it later!***

Run with `sudo ./curing`. You need elevated permissions because of the kernel feature under test.

It should take seconds to run, print a 32-bit checksum of the random file from the CPU and GPU.
