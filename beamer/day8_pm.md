详细安排：
0. C++快速入门（25分钟）
   - C++与C的主要区别（5分钟）
       * 命名空间、输入输出（cout/cin）
       * 引用 vs 指针
       * bool类型
   - 面向对象基础（5分钟）
       * class定义（成员变量、成员函数）
       * 构造函数/析构函数
   - C++11特性速览（15分钟）
       * auto类型推导
       * 范围for循环
       * Lambda表达式（重点，因为线程中常用）
       * 智能指针（简要提及，避免内存泄漏）
1. 导论：CPU并行框架中的C++线程（10分钟）
   - 回顾OpenMP/MPI（强调OpenMP是编译器指令，MPI是进程间通信）
   - C++线程库：直接操作系统线程，更底层、更灵活
   - 适用场景：需要精细控制线程行为、任务并行、事件驱动
2. C++线程基础（40分钟）
   - std::thread 创建线程
       * 线程函数（普通函数、Lambda表达式、成员函数）
       * join() 和 detach()
   - 传递参数：值传递、引用传递（std::ref）的陷阱
   - 实战：多线程打印（每个线程打印不同字符串）
3. 共享数据同步（45分钟）
   - 竞态条件演示：两个线程同时累加一个全局变量（结果错误）
   - 互斥锁 std::mutex
       * lock()/unlock() 手动管理
       * RAII: std::lock_guard（重点推荐）
   - 死锁问题
       * 什么是死锁（四个必要条件）
       * 避免死锁：按固定顺序加锁，或使用std::lock()同时锁定多个互斥量
   - 实战：用互斥锁修复累加器
4. 生产者-消费者模型（35分钟）
   - 为什么需要条件变量？避免忙等待（busy-wait）
   - std::condition_variable
       * wait(lock, predicate) 和 notify_one()/notify_all()
   - 线程安全队列设计（简化版）
       * 队列使用std::queue
       * 互斥锁保护整个队列
       * 条件变量通知数据就绪
   - 实战：生产者生成数字，消费者打印数字
5. 总结与框架对比（15分钟）
   - 回顾C++线程核心：thread, mutex, condition_variable
   - 对比：
       OpenMP - 适合循环并行，简单
       MPI - 分布式内存，跨节点
       C++线程 - 共享内存，精细控制
   - 进阶话题提及（学生可自学）：
       * std::async / std::future（异步任务）
       * 原子操作（std::atomic）
       * 无锁编程
6. 异步任务模型（20分钟，如果时间允许）
   - std::async 创建异步任务
   - std::future 获取结果
   - 示例：异步计算平方，主线程等待结果
注意：如果前面部分超时，则异步任务模型部分可以跳过，或者仅用5分钟介绍概念，不写代码。
时间分配要灵活，重点确保前三部分（C++入门、线程基础、互斥锁）讲透，因为这是后续并行编程的基础。
教材内容取舍：
  - 重点：第2章（Managing threads）、第3.2节（mutex）、第4.1节（condition_variable）
  - 略讲：第4.2节（future）只做概念介绍
  - 跳过：原子操作、无锁编程、线程池等高级内容
教学资源：
  - 提供简单的C++语法速查表（1页PDF）
  - 所有代码示例使用C++11标准，确保兼容性
  - 强调C++并发编程中的常见错误（如忘记join、数据竞争）

---

### **C++并行程序设计**（3小时优化版）  
**主题**： *从C到C++：共享内存并行编程实战*  
**核心调整**：  
- 增加30分钟C++快速入门（聚焦并发编程必备特性）  
- 压缩高阶内容（保留生产者-消费者模型，精简异步任务部分）  
- 强化C vs C++并行实现对比  

---

### **课程框架**（180分钟）  

#### **0. C++快速入门（关键30分钟）**  
1. **C++与C的核心差异**（10分钟）  
   - 引用 vs 指针（`int& ref = var;`）  
   - 函数/运算符重载（`operator<<`）  
   - RAII范式：构造函数/析构函数自动资源管理  
   ```cpp
   // C++资源自动释放示例
   class FileHandler {
   public:
     FileHandler(const char* fname) { fp = fopen(fname, "r"); }
     ~FileHandler() { if(fp) fclose(fp); } // 自动释放
   private:
     FILE* fp;
   };
   ```

2. **并发编程必备特性**（20分钟）  
   - Lambda表达式（线程函数核心）  
   ```cpp
   // C风格函数指针 vs C++ Lambda
   void(*c_func)(void) = thread_func;  // C
   auto cpp_func = [](int id){         // C++
     std::cout << "Thread " << id; 
   };
   ```
   - 模板基础：`std::vector<T>`容器使用  
   - 智能指针简介：`std::unique_ptr`避免内存泄漏  

#### **I. CPU并行框架全景与C++线程基础**（40分钟）  
1. **并行框架对比**（10分钟）  
   | **特性**        | OpenMP         | MPI            | C++ Thread     |  
   |----------------|----------------|----------------|----------------|  
   | 编程模型        | 编译器指令      | 消息传递        | 原生API        |  
   | 内存模型        | 共享内存        | 分布式内存      | 共享内存       |  
   | 适用场景        | 规则循环并行    | 跨节点计算      | 复杂任务调度   |  

2. **线程生命周期管理**（30分钟）  
   - `std::thread`创建/等待（对比`pthread_create`）  
   ```cpp
   // C++线程创建 vs C的pthread
   pthread_t c_thread;  // C
   pthread_create(&c_thread, NULL, func, NULL);
   
   std::thread cpp_thread([]{  // C++
     std::cout << "Hello from thread!"; 
   }); 
   cpp_thread.join();  // 等待结束
   ```
   - 参数传递陷阱：值传递 vs `std::ref`引用传递  
   - **实战**：多线程数组求和（C版本用pthread，C++版本对比）  

#### **II. 共享数据同步实战**（50分钟）  
1. **从C互斥锁到C++ RAII锁**（20分钟）  
   - C的`pthread_mutex_lock` vs C++ `std::lock_guard`  
   ```cpp
   // C手动锁管理
   pthread_mutex_lock(&mutex);
   counter++; 
   pthread_mutex_unlock(&mutex);

   // C++自动锁管理 (RAII)
   {
     std::lock_guard<std::mutex> lock(mtx); 
     counter++;  // 自动解锁
   }
   ```
   - 死锁规避：`std::lock()`多锁原子获取  

2. **生产者-消费者模型**（30分钟）  
   - C++条件变量核心操作：  
     ```cpp
     std::unique_lock<std::mutex> lck(mtx);
     cv.wait(lck, []{ return !queue.empty(); }); // 条件等待
     cv.notify_all(); // 唤醒所有
     ```
   - **案例**：线程安全队列（对比OpenMP的`#pragma omp critical`）  

#### **III. 综合应用与框架选型**（30分钟）  
1. **蒙特卡洛π计算实现**（20分钟）  
   - C++线程版动态任务分配：  
     ```cpp
     auto worker = [&](int start, int end) {
       for(int i=start; i<end; ++i) {
         double x = rand(), y = rand();
         if(x*x + y*y < 1) local_count++;
       }
     };
     ```
   - 对比OpenMP静态调度：`#pragma omp parallel for reduction(+)`  

2. **框架选型指南**（10分钟）  
   ```mermaid
   graph LR
   A[并行任务] --> B{数据分布}
   B -->|规则数据| C[OpenMP]
   B -->|跨节点数据| D[MPI]
   B -->|复杂依赖| E[C++线程]
   E -->|需GPU加速| F[CUDA Stream]
   ```

#### **附：时间分配备用方案**  
| **模块**               | 基础版 | 扩展版 |  
|------------------------|--------|--------|  
| C++快速入门            | 30min  | 30min  |  
| 线程基础               | 40min  | 35min  |  
| 数据同步               | 50min  | 45min  |  
| 生产者-消费者          | 30min  | 35min  |  
| 综合应用               | 30min  | 35min  |  

> **教学建议**：  
> 1. 所有代码示例提供C与C++双版本对比  
> 2. 关键概念用C语言实现类比（如mutex→pthread_mutex_t）  
> 3. 实时演示：CLion/Qt Creator调试多线程程序  

此调整确保零C++基础的学生能平滑过渡，同时保持HPC并行核心内容的深度覆盖，符合短学期课程的紧凑性要求。