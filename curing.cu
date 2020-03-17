#include <atomic>
#include <chrono>
#include <thread>
#include <iostream>
#include <numeric>
#include <cstring>
#include <assert.h>
#include <unistd.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <liburing.h>
#include "syscall.h"

#include <cuda/std/atomic>

# define check(ans) { assert_((ans), __FILE__, __LINE__); }
inline void assert_(cudaError_t code, const char *file, int line) {
  if (code == cudaSuccess)
    return;
  std::cerr << "check failed: " << cudaGetErrorString(code) << " : " << file << '@' << line << std::endl;
//  abort();
}

static const int ioring_size = 16;

template <class F>
__global__ void run_inner(F* f)
{
    (*f)();
}

template <class F>
void run(F f)
{
    F* _f;
    check(cudaMallocManaged(&_f, sizeof(F)));
    new (_f) F(std::move(f));
    run_inner<<<1,1>>>(_f);
    check(cudaDeviceSynchronize());
    check(cudaFree(_f));
}

struct app_sq_ring {
    unsigned *head;
    unsigned *tail;
    unsigned *ring_mask;
    unsigned *flags;
    unsigned *array;
};

struct app_cq_ring {
    unsigned *head;
    unsigned *tail;
    unsigned *ring_mask;
    io_uring_cqe *array;
};

__managed__ uint32_t sum;
__managed__ uint32_t buff[1<<18];
__managed__ iovec vec;

int main()
{
    io_uring_params p;
    std::memset(&p, 0, sizeof(p));

    p.flags |= IORING_SETUP_SQPOLL;
    p.sq_thread_idle = 1000000;

    int ioring_fd = __sys_io_uring_setup(ioring_size, &p);
    if(ioring_fd < 0) {
        std::cout << "Error in setup : " << errno << std::endl;
        abort();
    }

    check(cudaGetLastError());
    auto const sqptr = (__u8*)mmap(0, p.sq_off.array + p.sq_entries * sizeof(__u32),
                                PROT_READ|PROT_WRITE, MAP_SHARED|MAP_POPULATE,
                                ioring_fd, IORING_OFF_SQ_RING);
    assert(sqptr != MAP_FAILED);
    check(cudaHostRegister(sqptr, p.sq_off.array + p.sq_entries * sizeof(__u32), cudaHostRegisterDefault));
    app_sq_ring request = {
        (unsigned *)(sqptr + p.sq_off.head),
        (unsigned *)(sqptr + p.sq_off.tail),
        (unsigned *)(sqptr + p.sq_off.ring_mask),
        (unsigned *)(sqptr + p.sq_off.flags),
        (unsigned *)(sqptr + p.sq_off.array)
    };

    auto const sqentriesptr = (io_uring_sqe*)mmap(0, p.sq_entries * sizeof(io_uring_sqe),
                                PROT_READ|PROT_WRITE, MAP_SHARED|MAP_POPULATE,
                                ioring_fd, IORING_OFF_SQES);
    assert(sqentriesptr != MAP_FAILED);
    check(cudaHostRegister(sqentriesptr, p.sq_entries * sizeof(io_uring_sqe), cudaHostRegisterDefault));

    auto const cqptr = (__u8*)mmap(0, p.cq_off.cqes + p.cq_entries * sizeof(io_uring_cqe),
                        PROT_READ|PROT_WRITE, MAP_SHARED|MAP_POPULATE, ioring_fd,
                        IORING_OFF_CQ_RING);
    assert(cqptr != MAP_FAILED);
    check(cudaHostRegister(cqptr, p.cq_off.cqes + p.cq_entries * sizeof(io_uring_cqe), cudaHostRegisterDefault));
    app_cq_ring response = {
        (unsigned *)(cqptr + p.cq_off.head),
        (unsigned *)(cqptr + p.cq_off.tail),
        (unsigned *)(cqptr + p.cq_off.ring_mask),
        (io_uring_cqe *)(cqptr + p.cq_off.cqes)
    };

    int _infd = open("myfile", O_RDONLY);
    assert(_infd > 0);

    int infd = __sys_io_uring_register(ioring_fd,  IORING_REGISTER_FILES, &_infd, 1);
    assert(_infd > 0);

    cuda::std::atomic_bool done = {false};
    std::thread helper([&](){
        while(!done) {
            if (reinterpret_cast<cuda::std::atomic_uint&>(*request.flags) & IORING_SQ_NEED_WAKEUP)
                __sys_io_uring_enter(ioring_fd, 1, 0, IORING_ENTER_SQ_WAKEUP, NULL);
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    });

    sum = 0;
    vec = iovec{ buff, 1<<20 };

    std::memset(sqentriesptr+0, 0, sizeof(io_uring_sqe));
    sqentriesptr[0].fd = infd;
    sqentriesptr[0].opcode = IORING_OP_READV;
    sqentriesptr[0].addr = (uint64_t)&vec;
    sqentriesptr[0].len = 1;
    sqentriesptr[0].flags = IOSQE_FIXED_FILE;

    auto inner = [infd, sqentriesptr, request, response] __host__ __device__ () {

        for(int i = 0; i < (1<<10); ++i)
        {
            // uring request
            {
                sqentriesptr[0].off = i<<20;
                unsigned const tail = reinterpret_cast<cuda::std::atomic_uint&>(*request.tail);
                while(1)
                {
                    unsigned const head = reinterpret_cast<cuda::std::atomic_uint&>(*request.head);
                    if(head != ((tail + 1) & *request.ring_mask))
                        break;
                }
                request.array[tail & *request.ring_mask] = 0;
                reinterpret_cast<cuda::std::atomic_uint&>(*request.tail) = (tail + 1);
            }
            // uring response
            {
                unsigned const head = reinterpret_cast<cuda::std::atomic_uint&>(*response.head);
                while(1)
                {
                    unsigned const tail = reinterpret_cast<cuda::std::atomic_uint&>(*response.tail);
                    if(head != tail)
                        break;
                }
                if(response.array[head & *response.ring_mask].res < 0) {
                    assert(0);
                }
                reinterpret_cast<cuda::std::atomic_uint&>(*response.head) = (head + 1);
            }
            // summation
            for(auto b : buff)
                sum += b;
        }
    };

    sum = 0;
    inner();
    std::cout << "CPU sum: " << std::dec << sum << std::endl;

    sum = 0;
    run(inner);
    std::cout << "GPU sum: " << std::dec << sum << std::endl;

    done = true;
    helper.join();

    check(cudaHostUnregister(sqptr));
    check(cudaHostUnregister(sqentriesptr));
    check(cudaHostUnregister(cqptr));
    __sys_io_uring_register(ioring_fd,  IORING_UNREGISTER_FILES, &_infd, 1);

    close(infd);
    close(ioring_fd);

    return 0;
}
