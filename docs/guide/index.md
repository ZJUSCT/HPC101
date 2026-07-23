# 登录平台

课程集群通过统一的 Web 平台使用，有两个入口：

- 校网：**<https://platform.s.zjusct.io>**
- 公网（校外可访问）：**<https://clusters.zju.edu.cn>**

!!! warning "注意"

    由于平台鉴权使用的 ZJU Git 依赖校网环境，故若未给账户设置密码，仍需使用VPN/校网环境登录平台并设置密码。

在平台上你可以：

- 创建和管理 **DevPod**（开发容器，通过 SSH 连接）
- 向各分区**提交计算任务**（类 Slurm 的 `hpc` 命令行）
- **提交实验报告**（平台从你的家目录收取文件树）

!!! warning "重要使用须知"

    - 集群为全体学员共享平台，请共同维护高效运行环境，严禁资源滥用。若发现异常占用（如恶意脚本、持续满负荷任务），将追溯责任人并限制账户。

    - 集群网络资源有限，严禁进行非授权的网络活动。

    - 请勿存放与课程无关的大文件（如影视、游戏、个人备份等）。

    - 重要数据及时本地备份。

## 登录

打开平台，点击 **Sign in with GitLab**，用你的 [ZJU Git](https://git.zju.edu.cn/) 账号授权即可。

平台账户名为 **h + 学号**。

!!! example

    学号 3260100000 的同学，平台账户名为 `h3260100000`。

## 创建 DevPod

DevPod 是你在集群上的开发环境：一个常驻的容器，带有课程实验所需的完整工具链。

1. 登录平台后，点击 **创建第一个**（或右上角 **New +**）。
1. 任取一个名称（如 `mypod`），选择实验对应的环境模板（如做 x86 任务选 x86 模板，做 RISC-V 任务选 RISC-V 模板，具体以实验文档指引为准）。
1. 等待状态从 `Pending` 变为 `Running`。

!!! tip

    每人可同时运行的 DevPod 数量有限。可按需暂停/删除不使用的 DevPod。

## 通过 SSH 连接 DevPod

### 配置 SSH 公钥

集群仅支持 SSH 密钥登录。请在 [ZJU Git](https://git.zju.edu.cn/) 中添加你的 SSH 公钥，集群会从这里动态获取。你可以通过 `https://git.zju.edu.cn/学号.keys` 查看已添加的公钥。如果没有配置过 SSH 密钥对，请参考 [Lab 0 Linux Crash Course](../lab/Lab0-LinuxCrashCourse/index.md) 中的配置方法。

### 连接

DevPod 运行后，可使用下面的命令连接：

```shell
ssh h学号+DevPod名+hpc101@clusters.zju.edu.cn -p 443
```

!!! example

    用户 `h3260100000` 连接自己名为 `mypod` 的 DevPod：

    ```shell
    ssh h3260100000+mypod+hpc101@clusters.zju.edu.cn -p 443
    ```

!!! warning "警告"

    如果这一步你看到了输入密码的提示，说明 SSH 密钥登录配置有误，需要重新检查。

    **请不要在登录集群时输入自己的密码**，这是因为登录入口配置了蜜罐，输入的密码会被明文记录在日志中。

`clusters.zju.edu.cn` 支持 SSH 的端口有：22、80、443。如果默认端口（22）无法连接，请尝试使用另外两个端口。受限于学校网络安全策略，目前已知下列情况下 22 端口无法使用，请使用 80 或 443 端口：

- **校外访问**（注意放假后校外访问时需要更改端口）
- RVPN（zju-connect）访问

!!! tip "推荐：配置 ssh config"

    如果使用终端命令行或者 Visual Studio Code 连接集群，可以通过配置 `ssh config` 来获得更加优雅的登录方式。

    对于 Linux 和 macOS，这一配置位于 `~/.ssh/config`；对于 Windows 则位于 `C:\Users\<用户名>\.ssh\config`。例如：

    ```text title="~/.ssh/config"
    Host mypod
        User h3260100000+mypod+hpc101
        HostName clusters.zju.edu.cn
        Port 443
    ```

    此后可以使用

    ```shell
    ssh mypod
    ```

    或在 Visual Studio Code 建立远程 SSH 连接时选择 `mypod` 连接到你的 DevPod。

## 家目录

你的家目录 `~` 在**所有 DevPod 和所有计算任务之间共享**，且路径相同：在 DevPod 里准备的代码和脚本，计算任务可以直接使用；换一个 DevPod，家目录里的东西还在。

与之相对，容器内家目录**之外**的位置（如 `/`，`/tmp`）是每个容器私有的临时空间，容器回收后即消失，且有较小的容量限额。请把所有需要保留的文件放在家目录下。

同时，由于家目录使用的 `nfs` 要求较低的网络延迟，不同物理地域机器的 DevPod、计算容器的家目录不同。

| 分区 ↓ \ DevPod 预设 → | x86-5418Y | riscv-K1 | arm64-920B | arm64-910B |
| --- | :-: | :-: | :-: | :-: |
| `lab2`  | ✅ 相同 | ✅ 相同 | ❌ 不同 | ❌ 不同 |
| `lab2rv`  | ✅ 相同 | ✅ 相同 | ❌ 不同 | ❌ 不同 |
| `lab3`  | ✅ 相同 | ✅ 相同 | ❌ 不同 | ❌ 不同 |
| `lab4`  | ❌ 不同 | ❌ 不同 | ✅ 相同 | ❌ 不同 |
| `lab4g10`  | ✅ 相同 | ✅ 相同 | ❌ 不同 | ❌ 不同 |
| `lab4g5`  | ✅ 相同 | ✅ 相同 | ❌ 不同 | ❌ 不同 |
| `lab5`  | ✅ 相同 | ✅ 相同 | ❌ 不同 | ❌ 不同 |
| `lab3p5`  | ❌ 不同 | ❌ 不同| ❌ 不同 | ✅ 相同 |

!!! example

    做 lab4 时，代码要放在 **arm64-920B** 预设的 DevPod 家目录里再 `hpc submit -p lab4`；在 x86 DevPod 里准备的文件，lab4 的任务是看不到的。




## 常见问题

!!! question "是否能看到如下的 SSH Banner？"

    ```shell
    $ ssh h学号+DevPod名+hpc101@clusters.zju.edu.cn -p 443
    Wed, 22 Jul 2026 13:39:59 +0000
    h学号+DevPod名+hpc101
    ***.***.***.***:****
    ```

    ??? note "不能看见 SSH Banner"

        未能连接到 `clusters.zju.edu.cn`。

        - 校网环境：检查 `clusters.zju.edu.cn` 的 DNS 解析是否正常。
        - 校外环境：尝试 80 或 443 端口，检查防火墙设置。

        如果使用的是 MobaXTerm、XShell 等软件，需要在其选项里找到关于协商 SSH 版本的选项，并指定为 SSHv2。

    ??? note "能看见 SSH Banner"

        根据 SSH Banner 后的错误信息检查：

        | 错误信息 | 原因 |
        | --- | --- |
        | `Permission denied (publickey).` | 用户名或 SSH 公钥未配置正确，检查 ZJU Git 上的公钥 |
        | `Unknown host "..."` | 缺少 `+hpc101` 后缀 |
        | `Connection closed by ***.***.***.*** port ***` | 内部错误，请联系管理员 |
        | `* Failed to connect to remote host: ssh: handshake failed: ssh: unable to authenticate, attempted methods [none publickey], no supported methods remain` | DevPod 不存在或不处于 Running 状态     |

### 集群 SSH 代理原理

集群使用 [OpenNG](https://github.com/mrhaoxx/OpenNG) 提供的 SSH 代理功能：代理通过 TCP 头识别 SSH 连接并按用户名中的路由信息（`用户+目标`）转发，详见 [:simple-github: 源码](https://github.com/mrhaoxx/OpenNG/blob/f59461d12c48a9410967c7f4dd5a5ae1df251eef/tcp/detect.go#L116)。

## 下一步

- [提交计算任务](./job.md)
- [提交实验报告](./submit.md)
