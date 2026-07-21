
#include "perf.h"

// initialize staic members
size_t perf::mem_peak = 0;
size_t perf::mem_current = 0;
int perf::sampling_interval = 200;
bool perf::have_statm = false;
char perf::statm[40] = " ";
struct itimerval perf::new_it;
struct itimerval perf::old;
struct sigaction perf::sa;
struct sigaction perf::old_sa;

perf::perf()
{
    int fd;
    sprintf(statm, "/proc/%d/statm", (int)getpid());
    if ((fd = open(statm, O_RDONLY)) != -1)
    {
        have_statm = true;
        close(fd);
    }

    if (sampling_interval > 0)
    {
        /* setup timer to sample memory usage */
        sa.sa_handler = &perf::sample_mem_usage;
        sigemptyset(&sa.sa_mask);
        /*sigfillset (&sa.sa_mask);*/
        sa.sa_flags = SA_RESTART;
        if (sigaction(TimerSignal, &sa, &old_sa))
            perror("sigaction 0");
        new_it.it_value.tv_sec = sampling_interval / 1000;
        new_it.it_value.tv_usec = (sampling_interval % 1000) * 1000;
        new_it.it_interval = new_it.it_value;
        if (setitimer(TimerType, &new_it, &old))
            perror("setitimer 0");
    }
}
perf::~perf()
{
}
void perf::sample_mem_usage(int dummy)
{
    int fd;
    struct rusage RU;
    size_t mem;
    static bool locked = false;

    if (locked)
        return;
    locked = true;

    /* TODO: configure checks for different systems */

    /* first, try /proc/pid/statm for Linux systems */
    if (have_statm && (fd = open(statm, O_RDONLY)) != -1)
    {
        int rsspages;
        static char buffer[256];
        char *p = buffer;
        /* see linux-2.6.15/Documentation/filesystems/proc.txt */
        rsspages = read(fd, buffer, sizeof(buffer) - 1);
        close(fd);
        buffer[rsspages] = '\0';

        strtol(p, &p, 10);            /* first number () */
        rsspages = strtol(p, &p, 10); /* second number */

        mem = (size_t)rsspages * (size_t)getpagesize();
    }
    else
    {
        /* next, try getrusage() */
        if (getrusage(RUSAGE_SELF, &RU))
            cout << "perf::sample_mem_usage calling getrusage fail" << endl;
        else
            mem = RU.ru_maxrss * (size_t)1024;
        /*mem = RU.ru_maxrss * getpagesize();*/
    }

    if (mem > mem_peak)
        mem_peak = mem;
    mem_current = mem;
    locked = false;
}
size_t perf::MemoryUsage(size_t *current_min, size_t *current_avg, size_t *current_max,
                         size_t *peak_min, size_t *peak_avg, size_t *peak_max,
                         int nprocs)
{
    sample_mem_usage(0);

    double a[2][3], b[2][3];
    a[0][0] = a[0][1] = a[0][2] = mem_current;
    a[1][0] = a[1][1] = a[1][2] = mem_peak;
    MPI_Allreduce(a, b, 6, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    b[0][1] /= nprocs;
    b[1][1] /= nprocs;

    if (current_min != NULL)
        *current_min = (size_t)(b[0][0] + 0.5);
    if (current_avg != NULL)
        *current_avg = (size_t)(b[0][1] + 0.5);
    if (current_max != NULL)
        *current_max = (size_t)(b[0][2] + 0.5);

    if (peak_min != NULL)
        *peak_min = (size_t)(b[1][0] + 0.5);
    if (peak_avg != NULL)
        *peak_avg = (size_t)(b[1][1] + 0.5);
    if (peak_max != NULL)
        *peak_max = (size_t)(b[1][2] + 0.5);

    return (size_t)b[0][2]; /* return max(mem_current) */
}
