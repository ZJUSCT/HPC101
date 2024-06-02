# 导言：Hello HPC World

第一次听到 HPC 这个词，你是不是觉得很陌生？没关系，我第一次听到时也是一头雾水。几乎没有学生在学习这门课程之前了解过 HPC，这是一个相对陌生的领域。在这门课程中，我们会一步步地带你走进 HPC 的世界，带你了解这个领域的基础知识和技术，带你感受到 HPC 的魅力。

!!! quote inline end ""

    ![Image title](index.assets/frontier.webp)

    最快的超级计算机 Frontier 每秒能提供 $10^{18}$ 次浮点运算，相当于 $10^6$ 张 4090 显卡。

## HPC 是什么？

HPC 是 High Performance Compute 的缩写，意思是高性能计算，通常是指通过**聚合计算能力**来提供比传统计算机或服务器更强大的计算性能，处理海量多维数据集（大数据），并以极高的速度解决复杂问题。HPC 系统的运行速度通常要比最快的商用台式机、笔记本电脑或服务器系统快一百万倍以上。

标准计算系统主要使用**串行计算**来解决问题，它将工作负载分成一系列任务，然后在同一处理器上依次执行这些任务。相比之下，HPC 则利用以下技术来提高计算速度：

- **大规模并行计算**。并行计算是在多个计算机服务器或处理器上同时运行多个任务。大规模并行计算则是使用数万到数百万个处理器或处理器核心的并行计算。
- **计算机集群**（也称为 HPC 集群）。HPC 集群由多个联网的高速计算机服务器组成，并有一个集中式调度器来管理并行计算工作负载。这些计算机被称为节点，可能会使用高性能多核 CPU，如今更有可能使用 GPU（图形处理单元），它们非常适合处理严格的数学计算、机器学习模型和图形密集型任务。单个 HPC 集群可能包括 100,000 个或更多节点。
- **高性能组件**：HPC 集群中的所有其他计算资源（网络、内存、存储和文件系统）都是高速、高吞吐量、低延迟组件，可以与节点同步，优化集群的计算能力和性能。

下图展示了一个典型的 HPC 集群架构：

![hpc-architecture](index.assets/hpc-architecture-explained-diagram.png)

HPC 提供的超高性能，使得它在很多领域都有着广泛的应用。随着 AI 技术的发展，**HPC 应用也已成为 AI 应用的代名词**，尤其是机器学习和深度学习应用。HPC 应用正在推动以下领域的持续创新：

- 科学研究。HPC 在科学研究中的应用包括气候建模、天气预报、宇宙学、地震学、生物学、物理学、化学、材料科学和地球科学。这些学科需要处理大量数据、进行复杂的模拟和建模，以及进行大规模的数据分析。
- 医疗保健、基因组学和生命科学。人类基因组测序的首次尝试耗时长达 13 年；而如今，HPC 系统可以在不到一天的时间内完成这项工作。在医疗保健和生命科学领域，HPC 的其他应用还包括药物发现和设计、癌症快速诊断和分子建模。
- 金融服务。除了自动交易和欺诈检测，HPC 还支持蒙特卡罗模拟和其他风险分析方法的应用。
- 政府和国防。在这一领域，两个日益增长的 HPC 用例是天气预报和气候建模，这两个用例都涉及处理大量的历史气象数据和气候相关数据点每日数百万次的变化。其他政府和国防应用包括能源研究和情报工作。
- 能源。能源相关 HPC 应用包括地震数据处理、油藏模拟和建模、地理空间分析、风场模拟和地形测绘。

## 在这个短学期课程中，你会学到什么？

在这门课中，你能够学习到的知识和技能大致如下：

<div class="grid cards" markdown style="grid-template-columns: 1fr">

- :construction_site:{ .lg .middle } **前置知识：计算机体系结构**

    ---

    要学习并行程序设计，首先要理解现有的硬件（如 CPU、GPU 等）是如何工作的、为并行做了哪些优化。并行化的目的之一，是为了最大化利用现有的硬件资源，对大量数据进行快速操作。对于 CPU，我们会了解指令集、存储器层次结构和局部性原理，知道 Cache 一致性问题的原因和解决办法，掌握 SIMD（Single Instruction Multiple Data） 并行方法。对于 GPU（以 NVIDIA 为主），我们会了解流处理器（Stream Processor，SP）、CUDA Core（甚至是 Tensor Core）的组织方式，掌握 CUDA 编程的基本方法。上面只提及了最重要的知识，你应该能体会到内容非常之多且陌生，这些内容大概会用 4 天的时间完成讲授。在这门短学期课程中，我们只需要对这些知识有基本的了解即可。一定程度上，对硬件的理解越深入，并行编程技术也会越好，所以这是值得在日后深入学习的模块。

- :computer:{ .lg .middle } **核心技术：并行编程技术**

    ---

    在本课程中，我们将学习三种经典的并行编程技术：OpenMP/MPI 和 CUDA。前者两者主要用于 CPU ，后者用于 GPU （当然二者可以结合使用，不过课程中不会涉及）。表面上，学习一些 API 就能写并行程序了，但性能并不会得到很大的提升。基于对现有硬件的理解进行针对性的优化是非常重要的一环。

    在基本的 API 之外，我们还将学习并行算法，深入体会不同算法的特点，以及如何根据硬件特性进行选择。并行编程的难点在于数据的并行化和任务的并行化，这需要对算法有深入的理解。因此你也需要经常阅读经典例程、他人的优化思路和效果，学习并积累经验。对 CUDA 来说，多看例程尤为重要。CUDA 提供了多层次的 API，我们使用的主要是 CUDART（CUDA RunTime API）。它与具体的 CUDA 硬件无关，但你仍然需要考虑数据会如何被存储在具体的硬件上、如何优化存取、运算和线程调度。比如，在 2080Ti 上调优过的程序，在 A100 上的性能表现可能会大打折扣。

- :penguin:{ .lg .middle } **辅助技术：Linux 相关**

    ---

    本课程全程使用 Linux 环境。如果你对 Linux 系统有过一定了解和经验，或是曾经用服务器搭建过网站并持续使用等，那么你的实验流程会十分舒适。不过没有基础也不必担心，集群使用 Slurm 和 Lmod 进行管理，配置环境和运行任务都非常方便。我们也会在课程中讲解如何使用这些工具，以及如何在集群上提交任务。如果出现问题，你可以随时询问运维人员，他们会很乐意帮助你解决问题。

- :robot:{ .lg .middle } **附加技术：科学计算、AI 相关**

    ---

    在第 2 个 Lab，你会接触科学计算中关键的工具：NumPy。在最后一个 Lab，你将会揭开 AI 的面纱，自己编写一个 CNN（Convolutional Neural Network，卷积神经网络），并进行多卡训练。课程会花约 2 天时间讲授 AI 基础。这次实验，或许是你第一次真正理解那些神秘的 AI 究竟是怎么用实际的代码构建起来的。

</div>

## 修读建议

!!! quote inline end ""

    ![comment](index.assets/comment.jpg)

阅读完上面的介绍，相信你已经初步了解了自己将要学习哪些内容。再回顾一下计算机系的培养方案，你会看见这些内容贯穿了整个计算机专业的学习。因此，有同学在 98 上评论“没有 ICS（计算机系统概论）的话，这门课就是最有用的”，这句话是相当正确的。HPC101 是一门非常有挑战性的课程，知识量大且难，注重实践操作，但也是一门非常有趣的课程。如果你对计算机体系结构、并行编程、AI 等方面有浓厚的兴趣，那么这门课程将会是你的一次深度学习之旅。

或许在进入计算机专业以来，你已经多次听到「CS 人的基本素养就是自学能力」这样的言论。这句话在 HPC101 中将会得到最好的体现。HPC 领域的知识广阔，我们难以在 14 * 2 个课时里向你讲授所有内容。事实上，这些课时只够上导论，能够让你对相关知识有一个基本的了解就已经达到了目的。因此，你注定要在课后花费一定时间和精力去深入学习，在 Lab 中尝试应用并真正理解。我们会提供一些资源和参考资料，帮助你更好地学习。希望你能够在课程中坚持**自发地学习**，收获满满的知识和技能。我们也非常鼓励你在学习过程中多多提问，多多交流，这样你会学得更快更好。

## 资源推荐

### 基础

!!! tip "这些书都不厚，你可以读完它们的！"

- [《Linux 实战宝典》](https://zh.singlelogin.re/book/25214582/048350/linux%E5%AE%9E%E6%88%98%E5%AE%9D%E5%85%B8.html?dsource=recommend)
    - 为什么不推荐鸟哥之类的书？因为他们基于 RHEL 类系统（RHEL、CentOS）等，目标是通过运维技术相关的考试，不适合我们。这本书对我们所需要的内容的讲解更加清晰易懂，符合我们的需求。
- [《并行程序设计导论》](https://book.douban.com/subject/20374756/)（An Introduction to Parallel Programming）
    - 很薄，介绍了 OpenMPI 和 OpenMP（Pthreads 我们不学，也基本不会接触到）。讲解的内容浅显易懂，适合快速入门。
- Python 编程三剑客：从上至下，书籍难度逐渐递增。如果你对自己的编程能力不太自信，选择第一本书。如果你编程能力好，但没接触过 Python，可以选择第二本。如果你对 Python 有一定了解，非常推荐你直接从第三本开始做一些简单的项目。
    - [《Python 编程 : 从入门到实践（第3版）》](https://book.douban.com/subject/36365320/)（Python Crash Course）
        - 项目推荐：数据可视化、Web 应用程序
    - [《Python 编程快速上手：让繁琐工作自动化》](https://book.douban.com/subject/35387685/)（Automate the Boring Stuff with Python, 2nd Edition: Practical Programming for Total Beginners）
        - 项目推荐：从 Web 抓取信息，处理 CSV 文件和 JSON 数据，保持时间、计划任务和启动程序
    - [《Python 极客项目编程》](https://book.douban.com/subject/27050630/)（Python Playground: Geeky Projects for the Curious Programmer）
        - 项目推荐：模拟生命、图片之乐
- [《Programming Massively Parallel Processors: A Hands-on Approach, 4th Edition》](https://zh.singlelogin.re/book/25394855/1ffed7/programming-massively-parallel-processors-a-handson-approach-4th-edition.html)
    - CUDA 编程目前仍在活跃发展中，因此没有合适的中文书籍。要想学习到最新的 CUDA 技术，只能看英文书籍，而且要看最新版，超过 5 年的书籍均可能过时。
- [《Parallel and High Performance Programming with Python: Unlock parallel and concurrent programming in Python》](https://zh.singlelogin.re/book/24327324/8c7dc6/parallel-and-high-performance-programming-with-python-unlock-parallel-and-concurrent-programming-in.html)
    - All in Python。是的，这本书介绍了如何在 Python 中使用 MPI、CUDA 等并行编程技术，但很可惜我们基本都是用 C/C++ 写相关代码。如果你有兴趣可以读一读。

### 进阶

!!! tip "下面的书籍/文档体量都比较大，就不适合学习了，作为手册需要时查阅即可）"

- [《Debian 管理员手册》](https://debian-handbook.info/browse/zh-CN/stable/index.html)（The Debian Administrator's Handbook）
    - 真正理解 Debian。
- [《The Linux Programming Interface》](https://book.douban.com/subject/4292217/)
    - 在不看源码的情况下，真正理解 Linux。
- [《Python 数据科学手册》](https://book.douban.com/subject/27667378/)（Python Data Science Handbook: Essential Tools for Working with Data）  
    - NumPy、Pandas、Matplotlib，这些是你今后在很多课程中都会离不开的数据处理工具。
- [《CUDA C++ Programming Guide》](https://docs.nvidia.com/cuda/pdf/CUDA_C_Programming_Guide.pdf)
    - 是的，学习 CUDA 的最好最权威的资料就是 NVIDIA 的官方文档。

## 参考资料

- [IBM：什么是 HPC？高性能计算简介](https://www.ibm.com/cn-zh/topics/hpc)
- [Introduction to HPC: What are HPC & HPC Clusters?](https://www.weka.io/learn/hpc/what-are-hpc-and-hpc-clusters/)
