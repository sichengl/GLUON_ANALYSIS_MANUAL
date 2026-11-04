# frontier_stream_d 代码讲解进度

## 用户要求与讲解方式

- 详细带读 `2pt_production/frontier/final_production_frontier/8gpu/frontier_stream_d` 的代码。
- 用户会随时插问。先答插问，再返回主线；插问本身不推进主线进度。
- 分段讲解，结合代码行号、变量含义、数组形状和实际数值例子。
- 每轮结束更新本文件，回复末尾说明停在哪里、下次从哪里接。
- 本任务是代码讲解；不提交计算作业，不修改生产代码。

## 当前书签

- 已讲：目录中四个文件与两个公共工具文件的分工；计算的整体流程；cfg/icfg、3072 个源位置、8 个进程的关系。
- 已逐段讲：单节点提交脚本第 1–35 行，包括 Slurm 资源声明、工作目录、依赖检查、运行副本与日志链接。
- 下一段：单节点提交脚本第 36–54 行，环境加载、读取 tuning 路径、QUDA 环境变量、缓存目录、GPU 通信库；随后第 56–77 行运行参数和 srun。
- 尚未详细讲：setup、公共类、Python 主程序、smearing 算法、收缩公式、输出及 Flux。
- 待答插问：无。

## 主线目录

1. 总览；单节点 `.sh` 第 1–77 行。
2. `sepq_2pt_setup_frontier_final_8gcd.py`，结合 `TwoPtParams` 默认值：格点、构型、源坐标、求解器、smearing、动量表、输出时间与路径。
3. 主 Python 第 1–101 行：导入、MPI 错误退出、参数解析、初始化、gamma、容器、文件名与全体一致的跳过判定。
4. 主 Python 第 103–153 行：构型循环、读 gauge、Coulomb gauge fixing、检查 theta、HYP、加载 Dirac。
5. 主 Python 第 155–225 行与 `coulomb_smearing.py`：源点循环、相位、正负 boost、传播子求解、sink smearing。
6. 主 Python 第 229–299 行：数组轴和 einsum 索引、pion、diquark 与 proton 的两项收缩、动量选择、源相位与空间平均。
7. 主 Python 第 301–407 行：gather、时间 roll、proton 反周期符号、forward/backward、parts、HDF5 元数据与合并、重启行为。
8. `_flux.sh` 第 1–100 行：多节点构型分配、Flux 内层调度、GPU 绑定、时间限制、日志与退出状态。
9. 串起一例从提交参数到最终数据轴的完整运行。

## 文件定位

目标目录：`/Users/mcp3270/Desktop/GLUON_ANALYSIS_MANUAL/2pt_production/frontier/final_production_frontier/8gpu/frontier_stream_d`

- 单节点提交脚本：`2pt_coulomb_boosted_sepq_3072srcs_frontier_final_8gcd.sh`，77 行。
- 多节点提交脚本：`2pt_coulomb_boosted_sepq_3072srcs_frontier_final_8gcd_flux.sh`，100 行。
- 参数文件：`sepq_2pt_setup_frontier_final_8gcd.py`，125 行。
- 主计算程序：`2pt_coulomb_boosted_sepq_3072srcs_frontier_final_8gcd.py`，407 行。
- 公共参数类和工具：`/Users/mcp3270/Desktop/GLUON_ANALYSIS_MANUAL/2pt_production/tools/pt2_comm_tools.py`。
- 公共 smearing：`/Users/mcp3270/Desktop/GLUON_ANALYSIS_MANUAL/2pt_production/tools/coulomb_smearing.py`。

## 后续讲解需保持的准确性

- 按可执行代码解释，不把旧注释当作当前运行事实。当前 setup 的 `tol=1e-8`；Flux 的 18.2 h 注释明确对应 `tol=1e-10` 的测试，不能当作当前保证。
- `delta_params` 是沿用名称，当前 setup 使用 Frontier 路径及 `[1,1,1,8]` 分解。
- `icfg` 是从 0 开始的数组下标；`cfg=204+6*icfg`。Python 的循环局部 `i_cfg` 与这两个量不同。
- 3072 是每构型的源中心位置数：24 个时间位置，每个 128 个空间位置；不代表 3072 次标量 RHS 求解。
- 8 ranks 共同处理一个构型的时间分片，局部时间长度 12。Flux 把不同构型分配到不同节点，每个构型仍用 8 ranks。
- 空间源在每个 t_src 内按 q 相位加权平均，t_src 轴在最终文件保留。
- GEVP 的 6×6 来自两种 smearing × 三种 gamma；proton 另外保留 4×4 Dirac 指标，未直接做 parity 投影。
- 六种 kinematic case 的 `forward`（q=0）和每个文件内 `pt2_forward`（正时间方向）不是同一个分类维度。
- `.sh` 显式传 `--n`，缺省 N=1；直接启动 Python 时类中 n_cfg 的默认值为 20。
- 检查点以完整 t_src 为单位；跳过判断是存在性检查。恢复已完成 t_src 不代表恢复求解器内部状态。
- `sys.excepthook` 在导入之后才注册；讲错误处理时不要夸大覆盖范围。
- 集群环境中的设置与性能注释仅是此代码的约定；若要评估当前 Frontier 官方要求或外部库实现，需要另行核实。

## 插问记录

暂无。
