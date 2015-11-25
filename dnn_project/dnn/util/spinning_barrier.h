#include <atomic>


namespace dnn {

class SpinningBarrier {
public:
    SpinningBarrier (unsigned int _n) : n (_n), bar(0), passed(0) {}

    void wait ()
    {
        char passed_old = passed.load(std::memory_order_relaxed);
        
        if(passed_old < 0) {
            throw dnnInterrupt();
        }

        if(bar.fetch_add(1) == (n - 1)) {
            // The last thread, faced barrier.
            bar = 0;
            // Synchronize and store in one operation.
            passed.store(passed_old ^ 1, std::memory_order_release);
        } else {
            // Not the last thread. Wait others.
            while(true) {
                char passed_new = passed.load(std::memory_order_relaxed);
                if((passed_new != passed_old)||(passed_new<0)) {
                    break;
                }
            }
            // Need to synchronize cache with other threads, passed barrier.
            // std::atomic_thread_fence(std::memory_order_acquire);
        }
    }
    void fail() {
        passed.store(-1);
    }
protected:
    /* Number of synchronized threads. */
    const unsigned int n;
    
    std::atomic<int> bar; // Counter of threads, faced barrier.
    std::atomic<char> passed ; // Number of barriers, passed by all threads.
};

}