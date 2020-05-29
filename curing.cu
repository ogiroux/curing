#include <map>
#include <vector>
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
#include <cuda/std/semaphore>

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

namespace cuda
{
    struct app_sq_ring {
        unsigned *head;
        unsigned *tail;
        unsigned *ring_mask;
        unsigned *flags;
        unsigned *array;
    };

    struct app_cq_ring {http://cuda-repo/release-candidates/kitpicks/cuda-r11-0/11.0.1/023/local_installers/cuda_11.0.1_450.36.04_linux_ppc64le.run
        unsigned *head;
        unsigned *tail;
        unsigned *ring_mask;
        io_uring_cqe *array;
    };
    
    __managed__ int ioring_fd;
    __managed__ app_sq_ring request;
    __managed__ app_cq_ring response;
    __managed__ io_uring_sqe* sqentriesptr;
    __managed__ iovec vec;
//    __managed__ std::binary_semaphore lock;

    struct syscall_tunnel
    {
        std::atomic_bool done = {false};

        ::std::thread helper;

        syscall_tunnel()
        {
            io_uring_params p;
            ::std::memset(&p, 0, sizeof(p));

            p.flags |= IORING_SETUP_SQPOLL;
            p.sq_thread_idle = 1000000;

            ioring_fd = __sys_io_uring_setup(ioring_size, &p);
            if(ioring_fd < 0) {
                ::std::cout << "Error in setup : " << errno << ::std::endl;
                abort();
            }

            check(cudaGetLastError());
            auto const sqptr = (__u8*)mmap(0, p.sq_off.array + p.sq_entries * sizeof(__u32),
                                        PROT_READ|PROT_WRITE, MAP_SHARED|MAP_POPULATE,
                                        ioring_fd, IORING_OFF_SQ_RING);
            assert(sqptr != MAP_FAILED);
        //    check(cudaHostRegister(sqptr, p.sq_off.array + p.sq_entries * sizeof(__u32), cudaHostRegisterDefault));
            request = app_sq_ring{
                (unsigned *)(sqptr + p.sq_off.head),
                (unsigned *)(sqptr + p.sq_off.tail),
                (unsigned *)(sqptr + p.sq_off.ring_mask),
                (unsigned *)(sqptr + p.sq_off.flags),
                (unsigned *)(sqptr + p.sq_off.array)
            };

            sqentriesptr = (io_uring_sqe*)mmap(0, p.sq_entries * sizeof(io_uring_sqe),
                                        PROT_READ|PROT_WRITE, MAP_SHARED|MAP_POPULATE,
                                        ioring_fd, IORING_OFF_SQES);
            assert(sqentriesptr != MAP_FAILED);
        //    check(cudaHostRegister(sqentriesptr, p.sq_entries * sizeof(io_uring_sqe), cudaHostRegisterDefault));

            auto const cqptr = (__u8*)mmap(0, p.cq_off.cqes + p.cq_entries * sizeof(io_uring_cqe),
                                PROT_READ|PROT_WRITE, MAP_SHARED|MAP_POPULATE, ioring_fd,
                                IORING_OFF_CQ_RING);
            assert(cqptr != MAP_FAILED);
        //    check(cudaHostRegister(cqptr, p.cq_off.cqes + p.cq_entries * sizeof(io_uring_cqe), cudaHostRegisterDefault));
            response = app_cq_ring{
                (unsigned *)(cqptr + p.cq_off.head),
                (unsigned *)(cqptr + p.cq_off.tail),
                (unsigned *)(cqptr + p.cq_off.ring_mask),
                (io_uring_cqe *)(cqptr + p.cq_off.cqes)
            };

            helper = ::std::thread([&](){
                while(!done) {
                    if (reinterpret_cast<std::atomic_uint&>(*request.flags) & IORING_SQ_NEED_WAKEUP)
                        __sys_io_uring_enter(ioring_fd, 1, 0, IORING_ENTER_SQ_WAKEUP, NULL);
                    ::std::this_thread::sleep_for(::std::chrono::milliseconds(10));
                }
            });
        }

        ~syscall_tunnel()
        {
            done = true;
            helper.join();
            close(ioring_fd);
        }

        ::std::map<int, int> fds;
        ::std::vector<int> registry;
    };

    syscall_tunnel t;

    struct FILE
    {
        int fd;
        size_t read_off;
        size_t write_off;
    };

    int open(char const* name, int opt)
    {
        int const _infd = ::open(name, opt);
        assert(_infd > 0);

        if(!t.registry.empty())
            __sys_io_uring_register(ioring_fd,  IORING_UNREGISTER_FILES, 0, 0);

        int const infd = t.registry.size();
        t.registry.push_back(_infd);

        int const ret = __sys_io_uring_register(ioring_fd,  IORING_REGISTER_FILES, t.registry.data(), t.registry.size());
        assert(ret == 0);

        t.fds[infd] = _infd;
        return infd;
    }

    void close(int infd)
    {
        int _infd = t.fds[infd];
        //__sys_io_uring_register(ioring_fd,  IORING_UNREGISTER_FILES, &_infd, 1);
        ::close(_infd);
    }

    __host__ __device__ size_t fread(void * ptr, size_t size, size_t count, FILE * stream) {
//        lock.acquire();
        // uring request
        {
            ::std::memset(sqentriesptr+0, 0, sizeof(io_uring_sqe));
            sqentriesptr[0].fd = stream->fd;
            sqentriesptr[0].opcode = IORING_OP_READV;
            vec = iovec{ ptr, size * count };
            sqentriesptr[0].addr = (uint64_t)&vec;
            sqentriesptr[0].len = 1;
            sqentriesptr[0].flags = IOSQE_FIXED_FILE;
            sqentriesptr[0].off = stream->read_off;
            unsigned const tail = reinterpret_cast<std::atomic_uint&>(*request.tail);
            while(1)
            {
                unsigned const head = reinterpret_cast<std::atomic_uint&>(*request.head);
                if(head != ((tail + 1) ))
                    break;
            }
            request.array[tail & *request.ring_mask] = 0;
            reinterpret_cast<std::atomic_uint&>(*request.tail) = (tail + 1);
        }
        // uring response
        {
            unsigned const head = reinterpret_cast<std::atomic_uint&>(*response.head);
            while(1)
            {
                unsigned const tail = reinterpret_cast<std::atomic_uint&>(*response.tail);
                if(head != tail)
                    break;
            }
            if(response.array[head & *response.ring_mask].res < 0) {
                assert(0);
            }
            reinterpret_cast<std::atomic_uint&>(*response.head) = (head + 1);
        }
        stream->read_off += size * count;
//        lock.release();
        return count;
    };



    __host__ size_t fwrite(void const * ptr, size_t size, size_t count, FILE * stream) {
//        lock.acquire();
        // uring request
        {
            ::std::memset(sqentriesptr+0, 0, sizeof(io_uring_sqe));
            sqentriesptr[0].fd = stream->fd;
            sqentriesptr[0].opcode = IORING_OP_WRITEV;
            vec = iovec{ (void*)ptr, size * count };
            sqentriesptr[0].addr = (uint64_t)&vec;
            sqentriesptr[0].len = 1;
            sqentriesptr[0].flags = IOSQE_FIXED_FILE;
            sqentriesptr[0].off = stream->write_off;
            unsigned const tail = reinterpret_cast<std::atomic_uint&>(*request.tail);
            while(1)
            {
                unsigned const head = reinterpret_cast<std::atomic_uint&>(*request.head);
                if(head != ((tail + 1) ))
                    break;
            }
            request.array[tail & *request.ring_mask] = 0;
            reinterpret_cast<std::atomic_uint&>(*request.tail) = (tail + 1);
        }
        // uring response
        {
            unsigned const head = reinterpret_cast<std::atomic_uint&>(*response.head);
            while(1)
            {
                unsigned const tail = reinterpret_cast<std::atomic_uint&>(*response.tail);
                if(head != tail)
                    break;
            }
            if(response.array[head & *response.ring_mask].res < 0) {
                ::std::cout << response.array[head & *response.ring_mask].res << ::std::endl;
                assert(0);
            }
            reinterpret_cast<std::atomic_uint&>(*response.head) = (head + 1);
        }
        stream->write_off += size * count;
//        lock.release();
        return count;
    };





}

__managed__ uint32_t sum;
__managed__ uint32_t buff[1<<18];

int main()
{
    int infd0 = cuda::open("myfile", O_RDONLY);
    int infd1 = cuda::open("/dev/tty", O_WRONLY);
    auto inner = [infd0, infd1] __host__ __device__ () {
        cuda::FILE file0 = { infd0, 0 , 0 };
        for(int i = 0; i < (1<<10); ++i)
        {
            cuda::fread(buff, 1, 1<<20, &file0);
            for(auto b : buff)
                sum += b;
        }
        cuda::FILE file1 = { infd1, 0 , 0 };
        cuda::fwrite("It worked!\n", 1, 11, &file1);
    };

    sum = 0;
    inner();
    std::cout << "CPU sum: " << std::dec << sum << std::endl;

//    sum = 0;
//    run(inner);
//    std::cout << "GPU sum: " << std::dec << sum << std::endl;

//    check(cudaHostUnregister(sqptr));
//    check(cudaHostUnregister(sqentriesptr));
//    check(cudaHostUnregister(cqptr));

    cuda::close(infd0);
    cuda::close(infd1);
    return 0;
}
