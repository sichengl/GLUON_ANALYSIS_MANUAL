# frontier_stream_d 代码讲解进度

## 讲解约定
- 详细、分段带读目标目录的代码，结合行号、指标、数组形状和具体数值。
- 用户随时插问；先答插问，原主线书签保持，随后再接回主线。
- 不提交计算作业，不修改生产代码。

## 主线书签
- 已讲：四个入口/参数文件与两个公共工具文件的分工；cfg 与 icfg；3072 个源中心；8 个 MPI ranks 的时间分解。
- 已逐段讲：单节点 .sh 第 1–35 行，Slurm 声明、工作目录、依赖检查、运行副本、日志链接。
- 下一段：单节点 .sh 第 36–54 行（运行环境与 QUDA 设置），随后第 56–77 行（参数与 srun）。

## 当前插问：contract 部分
- 用户要求：解释 contract 的部分。
- 本次讲解范围：主 Python 第 229–299 行，并引用第 169–181 行的相位定义。
- 已讲内容：einsum 显式输出规则；even-odd 坐标与传播子的 sink/source 自旋颜色轴；每 4 个局部时间片分块；pion 的 open 自旋指标、动量投影和 source gamma；diquark 中间张量；proton 两个 Wick 项、动量投影与 source gamma；q 相位和空间平均；最终保留的 Dirac 指标和 6×6 算符矩阵。
- 公式关键：pion Q_li = sum_x,j,k,a,b phi_p S1*_(ji,ba) (G5 Gamma_sink)_jk S2_(kl,ba)；随后 sum_li (Gamma_bar_source G5)_li Q_li。
- 公式关键：D_ijabf = sum_d,e,g,h eps_def (C Gamma_sink)_gh S_(hj,eb) S_(gi,da)。R_ijlk = sum_a,b,c,f eps_abc [D_ijabf S_(lk,fc) - D_kjcbf S_(li,fa)]。
- proton 的 l 是外部 sink Dirac 指标，k 是外部 source Dirac 指标；原注释把二者一起称作 sink spins 不够准确。
- einsum 字母在每次调用中独立命名；末尾 qzt 的 z 是 pz 列表轴，与前面 wtzyx 的空间 z 无关。
- 不把 contract 当作会自动推导 Wick 收缩的工具；表达式中的两项、减号、共轭和 gamma 都是代码显式指定的。
- 主线仍停在单节点 .sh 第 35 行；继续追问 contract 时沿本专题展开。

## 后续主线顺序
1. 单节点启动脚本余下部分。
2. setup 与 TwoPtParams：格点、构型、源偏移、求解器、smearing、动量表和路径。
3. 主 Python 第 1–101 行：MPI 错误处理、参数、初始化、容器和文件判定。
4. 第 103–153 行：gauge、Coulomb fixing、theta、HYP、Dirac gauge 加载。
5. 第 155–225 行与 coulomb_smearing.py：源、boost、求解和 sink smearing。
6. 第 229–299 行：contract 已在插问中展开，主线到这里可先回顾，再依用户需要补充。
7. 第 301–407 行：gather、roll、反周期符号、forward/backward、parts、HDF5、重启。
8. Flux 多节点调度；串起一次完整运行。

## 文件位置
目标目录：/Users/mcp3270/Desktop/GLUON_ANALYSIS_MANUAL/2pt_production/frontier/final_production_frontier/8gpu/frontier_stream_d
- 2pt_coulomb_boosted_sepq_3072srcs_frontier_final_8gcd.sh
- 2pt_coulomb_boosted_sepq_3072srcs_frontier_final_8gcd_flux.sh
- sepq_2pt_setup_frontier_final_8gcd.py
- 2pt_coulomb_boosted_sepq_3072srcs_frontier_final_8gcd.py
公共工具：/Users/mcp3270/Desktop/GLUON_ANALYSIS_MANUAL/2pt_production/tools/pt2_comm_tools.py 和 coulomb_smearing.py。

## 准确性提醒
- 当前 setup tol=1e-8；Flux 性能注释的 18.2 h 对应 tol=1e-10 测试，不能当作当前性能保证。
- delta_params 为沿用名称，本目录的 grid 与路径属于 Frontier。
- 3072 是源中心位置数，不等于标量 RHS 求解次数。
- 空间源加权平均；最终保留 t_src 轴。
- 6×6 来自两种 smearing × 三种 gamma；proton 另外保留 4×4 Dirac 指标，尚未 parity 投影。
- kinematic case forward（q=0）与时间方向 pt2_forward 是两个不同维度。
- 检查点粒度为完整 t_src，跳过依据为文件存在性。

## 插问续：从原始 contract 推导当前优化版
- 用户要求：从原始的大 contract 一步步变到更快的优化版本。
- 原始代码定位：frontier/frontier_tests/2pt_off_forward_production_2gpu_frontier_3gamma_tunning.py 第 178–204 行；pion 一个完整表达式，proton 两个 Wick 项并分别求 parity_p / parity_m。
- 中间版 1：frontier/frontier_tests/2pt_off_forward_production_4gpu_frontier_3gamma_tuning_diquark.py 第 180–207 行；先提出 diquark，在 source gamma 循环外复用。
- 中间版 2：fermilab/fermilab_tests/2pt_off_forward_production_4gpu_fermilab_3gamma_tunning_diquark_opendirac.py 第 188–209 行；保留外部 l,k，推迟宇称投影。
- 中间版 3：fermilab/fermilab_tests/2pt_off_forward_production_4gpu_fermilab_3gamma_diquark_opendirac_optimizedcontract.py 第 188–213 行；局部颜色收缩、空间动量投影、最后 source gamma，两个 Wick 项仍各做一次投影。
- 当前版第 242–252 行：两个 Wick 项先在格点上相减，再做一次动量投影；第 267–299 行才乘 source gamma 和源相位；第 238 行按时间块处理。
- 推导顺序：原式 → 提取 D → 开放 l,k → 推迟 source gamma，形成 R1/R2 与 H → 合并两个 Wick 项后投影 → 99 动量复用与 q 源权重 → 时间分块；pion 用相同思路提出 source gamma。
- 关键指标映射：D_ijabf 的定义中 i→k、a→c，得到第二项的 D_kjcbf，展开为 eps_def A_gh S_hj^eb S_gk^dc；它是同一个 D 数组在该 einsum 中的轴标签，不要求新的 diquark 计算。
- 所有等价关系都要求同一组传播子、相位、gamma 和投影矩阵。不同生产版本的 smearing/phase 等参数不因此被视为相同。
- 当前保存 proton 的开放 Dirac 矩阵 M_lk；原版的 scalar 投影为 sum_kl P_kl M_lk。物理 backward 额外符号和 roll 不属于本节代数变换。
- 性能边界：旧版固定 sink gamma 有 3 source gamma × 2 parity × 2 Wick = 12 次大 contract 调用，但不声称优化后恰好快 12 倍。opt_einsum 单次调用也会优化；显式缓存 D/H 才实现这里的跨调用复用。
- 显存举例（complex128、32^3 空间）：D 12 时间片 2592 MiB，4 时间片 864 MiB；proton_site 分别 1536/512 MiB；H 的完整 (4,4,4,4,99,12) 数组约 4.64 MiB。不是总显存峰值。
- 数值核对：NumPy complex128，小格点完整 Ns=4/Nc=3，3 sink 矩阵、3 source 矩阵、2 投影、5 动量；直接原式和完整分解后投影最大绝对误差 1.510e-14，最大按参考最大幅值归一化误差 1.697e-15。另通过 D 提取、两项合并、时间分块及 pion source gamma 推迟的等价检查。这不是 Frontier GPU 性能测试。
- 本次讲解完成到当前优化版的代数重构；主线 .sh 第 35 行书签保持不变。

## 插问续：pion contract 中两个额外 G5 的来源
- 用户贴出 v0 pion 完整 contraction，询问 G5 从哪里来。
- 当前讲解：两个额外 G5 来自 Wilson-clover Dirac 算符的 gamma5 厄米性，不是优化引入，也不等同于插值算符选择 Gamma=gamma5。
- D†=gamma5 D gamma5 ⇒ S†=gamma5 S gamma5 ⇒ S1(y,x)=gamma5 [S1(x,y)]† gamma5（最后的 dagger 为自旋颜色矩阵 dagger）。
- 先定义 trace 本体 T=Tr[Gamma_sink S2(x,y) Gamma_bar_source S1(y,x)]，代入并利用 trace 循环性得 T=Tr[S1† (gamma5 Gamma_sink) S2 (Gamma_bar_source gamma5)]；这正是代码的四因子结构。
- Gamma_bar_source=gamma4 Gamma_source† gamma4 来自 source 算符的 Dirac 共轭；两个 gamma5 则来自反向传播子的改写，两者来源不同。
- .conj() 与 ji/kl/li 的指标连接共同实现 trace 中 S1† 的自旋颜色转置，不应额外对整个9维数组调用 .T。
- 符号必须明确：对于 O_sink=bar(q1) Gamma_sink q2、O_source†=bar(q2) Gamma_bar_source q1，标准 Grassmann 收缩为 C_conn=-<T>。贴出的代码计算 +T，没有显式外部 Wick 负号。不能声称 gamma5 厄米性自动消去该负号，也不能未检查完整算符/数据约定便宣称代码错误。
- 带 smearing 时 dagger 还作用于两端 smearing 核并交换端点；核不作用 Dirac 自旋，因此这里 gamma5 插入的来源不变。
- 暂停在 gamma5 厄米性到 pion einsum 的推导；原主线 .sh 第35行和优化推导书签不变。
