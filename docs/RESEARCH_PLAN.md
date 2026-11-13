# 量子启发的超图结构学习与重建：理论、算法、实验及代码实现规范

**英文工作标题**：Quantum-Interference-Guided Hypergraph Structure Learning and Reconstruction  
**文件定位**：研究方案 + 理论推导 + 可直接交由代码大模型执行的研发规范（Research & Implementation Specification）  
**版本**：v1.0 · 2026-11-12  
**研究性质**：经典计算机上模拟的 *quantum-inspired* 方法；**不预设量子计算加速、真实超边恢复优势或实验成功**。

> **阅读说明与证据等级**
>
> - **[文献事实]**：来自已核对的论文官方页面、期刊网页或数据集发布页，见第 21 节的链接。
> - **[已推导]**：本文件给出明确假设和推导、可用数值单元测试校验的数学关系。
> - **[研究假设]**：尚未证明，必须通过实验或进一步理论工作检验。
> - **[工程建议]**：提供首版可复现实现方案，不属于已发表方法或理论定理。
> - **[禁止主张]**：没有证据前，不能使用“量子优势已被证明”“从普通图恢复真实不可分解多体交互”等表述。

---

## 0. 执行摘要：两项目标、一个共同机制、三个决策门

### 0.1 两个彼此区分的研究目标

**目标 A — Hypergraph Reconstruction（核心科学验证）**  
已知某个普通图是潜在真实超图的投影，输入投影图及协议允许的节点属性；使用**独立的训练超图或训练超边标注**学习结构先验，预测测试网络中真实超边集合，并与真实超边逐条比较。核心评价：精确超边 F1/Jaccard、候选召回、基数分层结果、泛化与开销。

**目标 B — Hypergraph Structure Learning（应用/泛化验证）**  
输入没有可信真实超边标注的普通属性图；构建潜在关联矩阵，将其送入固定的 HGNN/AllSet 等下游模型。核心评价：节点分类 Accuracy/Macro-F1、鲁棒性、计算量、结构稳定性。**目标 B 的准确率提升不能证明真实超边恢复。**

### 0.2 共同核心机制

```text
输入 G=(V,A,X) ——预处理、归一化、数据泄漏检查
         ├── 经典候选群组生成器  C={e1,...,eM0}
         ├── 经典结构特征 z_C(e)
         └── Hermitian Hamiltonian H_Q=f(A,X)
                          ↓
              U(t)=exp(-it H_Q)
                          ↓
      相干/非相干群组级动态响应 z_Q(e)
                          ↓
          同候选、同评分器的机制辨别实验
                          ↓
       ┌─────────────────────────────────┐
       │ Gate 1：z_Q 有独立判别价值吗？  │
       └─────────────────────────────────┘
                          ↓ 是
         候选评分、结构约束、超边筛选
                          ↓
       ┌─────────────────────────────────┐
       │ Gate 2：真实超边恢复改善吗？    │
       └─────────────────────────────────┘
                          ↓ 是
        可微成员调整、结构学习与 HGNN 验证
                          ↓
       ┌─────────────────────────────────┐
       │ Gate 3：收益足以抵偿成本吗？    │
       └─────────────────────────────────┘
```

**最重要的研究原则**：先证明所定义的量子群组响应能帮助区分真实超边与匹配的非真实候选，再决定是否开发复杂的端到端超图生成器。若经典图谱、共同邻居或波动力学能够以更低开销获得同等表现，则不应为了保留“量子”而继续堆叠模型。

### 0.3 建议的最小成果与扩展成果

| 层级 | 研究输出 | 必须解决的问题 |
|---|---|---|
| P0 | 数学定义、单元测试、极限反例 | 响应指标是否正确、是否退化？ |
| P1 | 量子 vs 经典的候选超边判别实验 | 量子机制是否提供增量信息？ |
| P2 | 真实超图重建基线和完整方法 | 能否改善精确超边恢复？ |
| P3 | 无超边标注的软关联学习 | 能否学习稳定而紧凑的潜在超图？ |
| P4 | 固定下游网络的节点分类/鲁棒性 | 是否有应用价值且成本合理？ |

---

## 1. 文献现状：超图构建的研究问题及理论脉络

### 1.1 三个不同任务，避免混淆

1. **Construction / Structure Learning**：由节点特征、普通图或一个不准确的初始超图，生成或优化超边；通常由下游任务监督，结构未必是真实事件。
2. **Reconstruction**：一个真实超图经观测/投影后部分信息丢失，尝试恢复原始超边；需要定义观测模型、先验和可恢复性条件。
3. **Generation**：从一批超图学习分布 $p_\theta(\mathcal H)$，生成具有类似统计性质的新超图；不等同于逐条恢复某个隐藏真实超图。

### 1.2 代表性工作及可复用理论思想

| 论文 | 出处 | 原方法的核心 | 对本研究的具体启发 | 证据边界 |
|---|---|---|---|---|
| Zhou et al., *Learning with Hypergraphs* | NeurIPS 2006 | 超图 Laplacian 与谱学习 | 明确关联矩阵、归一化、结构传播的数学定义 | 不是端到端生成超边的方法 |
| Feng et al., *Hypergraph Neural Networks* (HGNN) | AAAI 2019 | 利用构造的超图进行卷积 | 目标 B 的固定下游结构编码器 | 下游分类不等于真实超边恢复 |
| Jiang et al., *Dynamic Hypergraph Neural Networks* (DHGNN) | IJCAI 2019 | 每层随节点表示更新超图，局部/全局构建 | 动态结构不是固定 kNN 的简单复制 | 核心为任务导向动态构建 |
| Cai et al., *Hypergraph Structure Learning for Hypergraph Neural Networks* (HSL) | IJCAI 2022 | 超边采样与关联节点采样、对比一致性 | 同时考虑增删关系与结构去噪 | 假设存在可优化的原始超图 |
| Zhou et al., *Totally Dynamic Hypergraph Neural Networks* (TDHNN) | IJCAI 2023 | 学习超边特征分布、节点关联及自适应候选数量 | 超边数量 $M$ 不应机械固定为 $N$ | 自适应策略不等于最优 $M$ 的普适定理 |
| Young, Petri & Peixoto, *Hypergraph reconstruction from network data* | Communications Physics 2021 | 贝叶斯生成模型、结构先验与 MDL | 要对“为什么选这组超边”提供统计解释 | 后验取决于模型假设，欠定问题仍存在 |
| Wang & Kleinberg, *From Graphs to Hypergraphs: Hypergraph Projection and its Reconstruction* | ICLR 2024 | 投影信息损失、可恢复结构条件、SHyRe 候选采样/分类 | 目标 A 的主要基线及理论约束 | 监督重建需要领域先验；不保证任意图可恢复 |
| Bresler, Guo & Polyanskiy, *Thresholds for Reconstruction...* | COLT 2024 | 随机 $d$-uniform 超图投影的恢复阈值 | 理论需明确生成模型、密度与可辨识条件 | 非任意真实网络的通用结论 |
| Bresler et al., *Partial and Exact Recovery...* | COLT 2025 | 精确/部分恢复及边重数可用性的理论界限 | 同一图投影可能不可恢复；边重数必须单独设实验 | 成立于其随机模型设定 |
| Lee, Lee & Shin, *MARIOH* | ICDE 2025 | 利用边重数和结构特征进行监督重建 | 必须区分 binary projection 与 multiplicity projection | 不能让不同方法使用不同输入信息量 |
| Duta & Liò, *SPHINX* | ICML 2025 | 任务驱动、可微的离散潜在超图推断 | 目标 B 的结构预测基线 | 改善任务不意味着恢复真实交互 |
| Sim et al., *Learning Multi-Scale Hypergraph...* (MuHL) | ICML 2026 | 自适应、多尺度高阶关联 | 测试多尺度、不同超边基数 | 主要实验证据为疾病分类及解释性 |
| Wen & Yu, *HyperPLR* | ICLR 2025 | 投影—学习—重建的超图生成 | 生成模型可作为后期参考 | 它生成新超图，不是严格对应的真实超边重建 |
| Chien et al., *You Are AllSet* | ICLR 2022 | 可学习的节点↔超边多重集合聚合 | 目标 B 的另一固定下游网络 | 传播架构不是待测量子构建机制 |

正式代码实现前，应按第 21 节核对官方页面与可用代码；**不能凭论文标题就假设代码、数据格式或完整复现实验已经验证。**

### 1.3 对现有研究的批判性总结

- 从固定 kNN/聚类向动态、可微、任务相关结构转变，是确实存在的研究趋势，但**静态方法仍是必要且强有力的基线**。
- 超边**权重**学习与超边**成员关系**学习不同；仅调整 $w_e$ 不能发现未出现过的节点—超边关联。
- “一群节点很相似”只是几何或统计群组关系，不必然对应真实不可分解的多体事件。
- 由分类标签优化的超图可能具有标签偏置；目标 A 必须用真实事件超边检验。
- 过大的、重复的、相互嵌套的超边会引入模型选择问题，但嵌套和重叠也可能是真实结构，不能一律清除。
- 图投影是多对一映射；学习算法可能通过额外训练先验改善**统计推断**，不能逆转纯输入的信息缺失。

---

## 2. 研究动机：量子游走提供的潜在价值与不能解决的问题

### 2.1 与经典扩散的数学差别

设 $H_Q=H_Q^\dagger$；本研究首版限定 $H_Q$ **实对称**，从而简化证明和数值校验。连续时间量子游走为

$$
U(t)=\exp(-\mathrm i tH_Q),\qquad U(t)^\dagger U(t)=I.
$$

若 $H_Q=V\Lambda V^\top$，则

$$
U(t)=V\operatorname{diag}(e^{-\mathrm i t\lambda_r})V^\top.
$$

相对地，热核为 $K_{\mathrm{heat}}(\tau)=e^{-\tau L}$。量子演化保留不同模态的相位、存在振幅的相长/相消干涉；经典热扩散按谱衰减。它们对关系的排序可能不同；但经典算法也能计算全谱及任意可定义的滤波器，因此**“谱响应不同”不能直接推出更高恢复率或量子计算加速**。

对单位初态 $|j\rangle$，转移概率

$$
P_{ij}(t)=|\langle i|U(t)|j\rangle|^2
=\sum_{r,s}v_{ir}v_{jr}v_{is}v_{js}\cos((\lambda_r-\lambda_s)t).
$$

这个公式表明相位差而非单纯局部距离参与概率计算。若使用完整幺正传播的列向量作为复空间嵌入，任意两个不同初始基态仍正交：

$$
\|U(t)e_i-U(t)e_j\|_2=\sqrt 2\quad(i\ne j).
$$

**因此不可直接声称量子演化将相似节点在完整振幅空间拉近。** 必须定义非平凡的测量、局部投影或群组级统计。

### 2.2 四个痛点：能否缓解？

| 现有难题 | 量子方案能做的事 | 不能主张的事 |
|---|---|---|
| 超边不等于真实多体交互 | 提供由相干传播产生的**候选证据特征**，并用真实事件检测其识别力 | 不能从单粒子干涉证明真实三体/四体相互作用 |
| 标签驱动结构偏置 | 首阶段量子特征独立于节点分类标签计算；目标 A 使用独立结构监督 | 量子模块本身不能消除特征/结构/训练先验偏置 |
| 超边数量 $M$ 难定 | 为候选排序、复杂度选择提供可能的信息 | 量子谱维数/概率峰数不等于真实超边数 |
| 投影后信息缺失 | 在有可恢复证据或领域先验时改善统计决策 | 不能在输入完全相同时区分不同真实超图 |

### 2.3 信息论限制（已成立的判断）

设真实超图 $\mathcal H^\star$，观测为 $Y=(A,X)$；量子启发计算为 $Z_Q=f_Q(Y)$。若不另引入包含真实结构信息的独立数据，数据处理不等式给出

$$
I(\mathcal H^\star;Z_Q)\le I(\mathcal H^\star;Y).
$$

若模型使用其他已标注的训练超图 $D_{\rm train}$ 学习先验，必须将它明确写成**额外的训练信息**：$f_Q(Y;D_{\rm train})$，不能说算法“仅凭同一普通图”获得了新真实信息。

**不可辨识反例**：$\mathcal H_1=\{\{1,2,3\}\}$ 与 $\mathcal H_2=\{\{1,2\},\{1,3\},\{2,3\}\}$ 的二值 clique projection 都是三角形 $K_3$。同样的 $A,X$ 经任何确定性/独立随机算法都无法在两个世界中同时保证准确恢复。

### 2.4 物理概念边界

- **本方案用单粒子量子游走的相干叠加构造群组统计量**，不是多粒子纠缠实验，也不是直接测量真实物理多体耦合。
- 用经典 CPU/GPU 模拟 $e^{-itH_Q}$ 属于 **quantum-inspired**；不能声称 quantum speedup。
- 如果后续要推断真实多体动力学，需要多点时间序列、干预或物理生成模型等额外观测，不属于首版任务。
- 所有主结果必须与**相同信息量和计算预算**下的经典热核、随机游走、共同邻居、谱特征以及经典波核比较。

---

## 3. 形式化问题定义与数据协议

### 3.1 统一记号

| 符号 | 类型/维度 | 含义 |
|---|---|---|
| $N$ | 整数 | 节点数量 |
| $A$ | $\{0,1\}^{N\times N}$ | 无向无自环二值投影图（主实验） |
| $W$ | $\mathbb R_{\ge0}^{N\times N}$ | 可选边重数/加权投影；**独立实验** |
| $X$ | $\mathbb R^{N\times d}$ 或 `None` | 节点特征，缺省时仅结构输入 |
| $\mathcal E^\star$ | 节点集合的集合或多重集 | 真实超边，目标 A 的评分/训练监督来源 |
| $B^\star$ | $\{0,1\}^{N\times M^\star}$ | 真实关联矩阵，不作为待重建测试图的输入 |
| $\mathcal C$ | $\{e_1,\ldots,e_{M_0}\}$ | 候选集合；所有方法共用 |
| $B_0$ | $\{0,1\}^{N\times M_0}$ | 候选关联矩阵 |
| $H_Q$ | $\mathbb R^{N\times N}$ | 实对称 Hamiltonian |
| $U(t)$ | $\mathbb C^{N\times N}$（概念） | 幺正演化；大规模不显式存储 |
| $z_Q(e)$ | $\mathbb R^{d_Q}$ | 群组级量子响应特征 |
| $z_C(e)$ | $\mathbb R^{d_C}$ | 经典结构/属性基线特征 |
| $g_e$ | $[0,1]$ | 候选超边保留门控 |
| $B$ | $[0,1]^{N\times M_0}$ | 软关联；后期可微扩展 |
| $\widehat{\mathcal E}$ | 节点子集集合 | 预测的离散超边 |

### 3.2 二值投影及信息损失

若真实超边为 $\mathcal E^\star$，主协议采用

$$
A_{ij}=\mathbf 1[\exists e\in\mathcal E^\star:\ i,j\in e],\quad i\neq j;
\qquad A_{ii}=0.
$$

实验可另行采用边重数

$$
W_{ij}=\sum_{e\in\mathcal E^\star}\mathbf 1[i,j\in e],\quad i\neq j.
$$

如果真实数据保留重复事件且为多重集，应当事先声明是计算**独特成员集合**还是**重复事件次数**；二值投影不能恢复事件重数。**禁止**把二值投影基线与额外获得 $W$ 的方法放在不加区分的同一结果表中。

### 3.3 目标 A 的训练条件

必须同时实现两套协议：

- **A1：跨图/跨时间监督重建（首选主协议）**。训练阶段使用其他训练网络/时间段的真实超边，拟合候选分类器；测试网络只暴露 $A,X$，隐藏测试真实超边。注意训练网络与测试网络的节点编号、身份和时间重叠问题。
- **A2：同图候选分层标注（诊断协议）**。可在同一投影图上将候选标签分为 train/val/test，研究候选判别机制，但由于所有候选共享同一张含测试交互的投影图，**不得将 A2 的性能当作严格的新图泛化或完全无泄漏的真实重建证据**。

若可用真实图数量有限，优先实施**明确时间先后**、不跨越边界的样本划分，并报告训练时可见信息；不要将同一原始事件拆到训练与测试后同时用于普通图构建。

### 3.4 目标 B 的训练条件

不存在真实超边标签；允许分类训练集的 $y_{\rm train}$ 参与下游监督，但不可使用 $y_{\rm val},y_{\rm test}$ 进行任何结构选择。分清：

- **B1：纯结构构建**（结构损失 + 可选自监督；不使用类别标签），用于检验构建本身；
- **B2：任务联合构建**（只用训练标签），用于评价下游性能，明确其任务偏置。

在 transductive 设置下允许读取所有节点特征和全图拓扑，但必须在论文及代码中声明；inductive 设置禁止提前使用测试节点/测试图的不可用信息。

---

## 4. 量子 Hamiltonian 的选择与可计算性

### 4.1 主版本：拓扑-only Hamiltonian，先避免特征混杂

主科学实验首先采用

$$
L_A=I-D^{-1/2}AD^{-1/2},\qquad H_Q=\tfrac12L_A.
$$

此处对非孤立节点采用对称归一化；孤立节点可置 $L_{ii}=0$ 且其余行列为 0，并在测试中覆盖该情况。对无向非负权图，$L_A$ 实对称，谱处于 $[0,2]$，故 $H_Q$ 谱处于 $[0,1]$。$H_Q$ 是 Hermitian，确保演化幺正。

**为什么先用拓扑-only？** 目标 A 本质是“投影结构可恢复性”问题；如果一开始把强特征编码器纳入 Hamiltonian，量子模块的结构价值将难以隔离。必须先在相同 $A$ 输入下与经典基线比较。

### 4.2 扩展版本：联合节点特征

利用训练协议允许的节点特征构建非负、对称、稀疏的特征相似图 $A_X$（如 mutual-kNN、余弦相似度截断；**不使用测试标签**），对应 $L_X$；定义

$$
H_Q(\lambda)=\tfrac12[(1-\lambda)L_A+\lambda L_X],\qquad\lambda\in[0,1].
$$

两个 Laplacian 按同一约定归一化，则仍实对称且谱在 $[0,1]$。需要分别报告 $\lambda=0$ 与 $\lambda>0$，证明收益是否来自特征相似度而非量子机制。

### 4.3 其他 Hamiltonian：只作为消融，不先混合

可单独比较 $H=A/\|A\|_2$（谱缩放）、$H=L_A/2$、$H=\text{normalized adjacency}$。为保证跨图时间可比，使用**固定且可复现**的谱尺度定义，而不能在测试阶段选择最有利的缩放。可学习 Hermitian 参数化放在 P3，不是首版必需组件。

### 4.4 不能忽视的归一化与相位问题

- 归一化 Laplacian 的符号结构及节点度会影响干涉；候选比较应控制度分布。
- 候选相干初态中的“所有节点同相位”依赖选定的规范基与相位约定，不是对任意局域相位基变换都不变的物理量。若对 Hamiltonian 作 $H\mapsto DHD^\dagger$ 的 gauge 变换，**初态也必须对应变换**才能维持相同物理预测。本文仅在固定的实节点基下定义评分。
- 量子相干可正可负；不能只奖励高测量概率或正干涉。

---

## 5. 核心候选理论：群组级量子相干响应（拟研究，尚非超边恢复定理）

### 5.1 候选初态与投影算子

给定候选节点集合 $e\subseteq V$、$k=|e|\ge2$。在首版固定规范基下定义同相位、归一化初态：

$$
|s_e\rangle=\frac1{\sqrt k}\sum_{i\in e}|i\rangle,\qquad
\Pi_e=\sum_{v\in e}|v\rangle\langle v|.
$$

演化态为 $|\psi_e(t)\rangle=U(t)|s_e\rangle$。定义**相干内部保留概率**：

$$
\boxed{R_{\rm coh}(e,t)=\|\Pi_eU(t)|s_e\rangle\|_2^2.}
$$

定义对应的**非相干初态混合**

$$
\rho_{\rm inc}(e)=\frac1k\sum_{i\in e}|i\rangle\langle i|,
$$

以及非相干内部保留概率：

$$
\boxed{R_{\rm inc}(e,t)=\frac1k\sum_{i\in e}\|\Pi_eU(t)|i\rangle\|_2^2.}
$$

定义**相干对照差异**

$$
\boxed{C(e,t)=R_{\rm coh}(e,t)-R_{\rm inc}(e,t).}
$$

三者均为可计算统计量，$R_{\rm coh},R_{\rm inc}\in[0,1]$，$C\in[-1,1]$。**$C$ 的正负均有意义；绝不默认真实超边具有更大的正 $C$。**

### 5.2 交叉干涉公式：严格代数展开

将初态代入得

$$
R_{\rm coh}(e,t)=\frac1k\sum_{v\in e}\left|\sum_{i\in e}U_{vi}(t)\right|^2
$$

$$
=\frac1k\sum_{v\in e}\sum_{i\in e}|U_{vi}|^2
+\frac2k\sum_{i<j;\ i,j\in e}\operatorname{Re}\left[\sum_{v\in e}U_{vi}U_{vj}^*\right].
$$

第一项等于 $R_{\rm inc}$，因此

$$
\boxed{C(e,t)=\frac2k\sum_{i<j;\ i,j\in e}\operatorname{Re}\left[\sum_{v\in e}U_{vi}(t)U_{vj}(t)^*\right].}
\tag{1}
$$

等价地，令 $K_e=[U(t)^\dagger\Pi_eU(t)]_{e,e}$，则

$$
R_{\rm coh}=\frac1k\mathbf1_k^\top K_e\mathbf1_k,\quad
R_{\rm inc}=\frac1k\operatorname{tr}(K_e),\quad
C=\frac1k\sum_{i\ne j}(K_e)_{ij}.
$$

复数情况下最后一式取实部（总值本身为实）。**重要批判**：式 (1) 是传播后的成对振幅交叉项之和，**本身并不证明不可约的三体/四体交互**；仅仅把输入集合扩大到 $k>2$，不意味着算子包含真实 $k$ 体物理项。

### 5.3 随机相位无偏参考（推导与实现依据）

令 $\xi_i\in\{-1,+1\}$ 独立等概率，定义

$$
|s_e^{\xi}\rangle=\frac1{\sqrt k}\sum_{i\in e}\xi_i|i\rangle.
$$

由于 $\mathbb E[\xi_i\xi_j]=\mathbf1[i=j]$，

$$
\boxed{\mathbb E_{\xi}\|\Pi_eU(t)|s_e^\xi\rangle\|^2=R_{\rm inc}(e,t).}
\tag{2}
$$

故可用 $R_{\rm coh}-\frac1R\sum_{r=1}^R R_\xi^{(r)}$ 无偏估计 $C$（固定 $H,e,t$ 条件下），并记录采样方差/置信区间。小图测试应**穷举所有 $2^k$ 个符号模式**验证 (2)；大图才使用 Monte Carlo。

### 5.4 小时间展开：明确指出一种“量子增益”解释的失败条件

首版限定 $H_Q\in\mathbb R^{N\times N}$ 实对称，$|s_e\rangle$ 取实向量。将 $H_Q$ 按 $e,e^c$ 分块，并记 $H_{e^c,e}$ 为从群组内部到外部的耦合矩阵。由于演化概率为 $t$ 的偶函数，

$$
R_{\rm coh}(e,t)=1-t^2\|H_{e^c,e}s_e\|_2^2+O(t^4),
$$

$$
R_{\rm inc}(e,t)=1-\frac{t^2}{k}\sum_{i\in e}\|H_{e^c,i}\|_2^2+O(t^4).
$$

相减并展开平方：

$$
\boxed{C(e,t)=-\frac{2t^2}{k}\sum_{i<j;\ i,j\in e}
\langle H_{e^c,i},H_{e^c,j}\rangle+O(t^4).}
\tag{3}
$$

**关键结论**：在 $t\to0$ 时，$C$ 的首个非零项是“候选节点通往外部的耦合列向量之间的内积”。对于来自非负普通图的 Laplacian，跨边块的非零元素通常同号，因而式 (3) 中交叉内积非负，短时间 $C$ 的主要项**非正**。它与加权共同外部邻居等经典统计存在直接关系。

这意味着：

1. **不能**将 $C>0$ 当作真实超边规则；
2. 短时间 $C$ 可能只是经典共同邻居的变换；
3. 需要检验中等时间的相位作用，并与足够强的经典谱/波核比较；
4. 候选规模、外部边界、节点度必须作为匹配变量或评分器输入。

### 5.5 必须接受的退化反例

- $t=0$：$R_{\rm coh}=R_{\rm inc}=1$，$C=0$。
- $e=V$：$\Pi_e=I$，两个保留概率都为 1，$C=0$。
- $e$ 若是独立连通分量（Hamiltonian 在 $e$ 与 $e^c$ 间无耦合）：两个保留概率均为 1，$C=0$。
- 具有相同投影和属性的不同真实超图：任何上述统计完全相同。
- 部分对称图或特定时间可能使不同候选响应一致，甚至使群组的全部判别特征退化。

**因此完整特征 $z_Q$ 必须联合使用 $R_{\rm coh}$、$R_{\rm inc}$、$C$、多时间变化及经典边界统计，而不是只根据一个差异标量筛选超边。**

### 5.6 多尺度特征与时间平均

对固定时间网格 $\mathcal T=\{t_1,\ldots,t_K\}$：

$$
z_Q(e)=\operatorname{Concat}_{t\in\mathcal T}
\left[R_{\rm coh}(e,t),R_{\rm inc}(e,t),C(e,t)\right].
$$

可扩展为

$$
\overline C(e,T)=\frac1T\int_0^T C(e,t)\,dt,
$$

并对 $R_{\rm coh},R_{\rm inc}$ 采用相同的积分方式。**首版建议固定少量 $t$，暂不学习时间。** 可试 $t\in\{0.25,0.5,1,2,4,8\}$（因主 Hamiltonian 谱在 $[0,1]$；此处是实验候选网格，不是验证后的最佳时间）。时间超参数必须只由训练/验证确定。

对完整量子转移矩阵，$P(t)=|U(t)|^{\circ2}$ 是双随机矩阵，有限时间平均亦然；其无限时间平均的代数性质已经有成熟研究。**但是**双随机 $P$ 不等于通常不受行列和限制的超图关联矩阵，因而不应直接设 $B=P$。

### 5.7 置换不变性（可证明，带先决条件）

若节点置换矩阵 $Q$ 满足 $H_Q(QAQ^\top,QX)=QH_Q(A,X)Q^\top$（这里第一个 $H_Q(\cdot)$ 表示 Hamiltonian 构造函数），则

$$
U'(t)=QU(t)Q^\top,\quad |s_{Qe}\rangle=Q|s_e\rangle,\quad \Pi_{Qe}=Q\Pi_eQ^\top.
$$

因此 $R_{\rm coh}(Qe,t)=R_{\rm coh}(e,t)$，$R_{\rm inc}$ 与 $C$ 同样不变。**注意**：完整流程还需保证候选生成、特征排序及 tie-breaking 置换等变，否则整体模型并不自动不变。

### 5.8 有限时间扰动上界（可证明，但不是量子鲁棒性优势）

给定实对称 $H,H'=H+\Delta$，由 Duhamel 公式

$$
\|e^{-itH'}-e^{-itH}\|_2\le |t|\|\Delta\|_2.
$$

固定候选 $e$、固定初态和投影，

$$
|R'_{\rm coh}-R_{\rm coh}|\le2|t|\|\Delta\|_2,
$$

非相干项同理，故

$$
\boxed{|C'(e,t)-C(e,t)|\le4|t|\|\Delta\|_2.}
\tag{4}
$$

这只是保守 Lipschitz 型界；若 $A\mapsto H$ 的归一化受节点度变化影响，须另外分析该映射。不应据此声称量子方法比经典扩散更稳定。

### 5.9 核心创新是否成立的最严格检验

**反事实实验**：在相同候选集合、相同经典协变量、相同评分器、相同随机种子及类似计算预算下，加入 $z_Q$ 是否带来可重复的验证/测试增益？

**机制消融**：仅 $R_{\rm inc}$、仅 $R_{\rm coh}$、仅 $C$、全部 $z_Q$；短时间二阶系数替代 $C$；经典波/谱核替代 $U(t)$；相位随机化初态替代统一初态；不同 $H$ 和 $t$。

若 $z_Q$ 的优势在这些控制下消失，该特征可以作为负面研究发现，但**不能写成已证明的量子高阶识别优势**。

---
## 6. 候选超边生成：让“超边判别”与“候选覆盖”分离

### 6.1 搜索空间与必要条件

对 $N$ 个节点，大小至少为 2 的潜在子集数为 $2^N-N-1$；直接枚举不可行。

**[已成立的必要条件]** 在**无噪声二值 clique projection** 下，任何真实超边 $e$ 的节点两两相连，即 $e$ 在 $A$ 中必须构成 clique。原因是任意 $i,j\in e$ 必须满足 $A_{ij}=1$。因此主目标 A 的候选应首先来自投影图的 clique 或其子集。**这一必要条件不是充分条件**：虚假三角团也可能是 clique。

在有删边噪声的实验、节点属性结构学习（目标 B）或不同观测机制下，不应将 clique 条件作为不可违背的硬约束。

### 6.2 候选生成器的最小实现

以 $A$ 为输入，产生下述候选池并做规范化去重：

| 生成渠道 | 首版方法 | 注意事项 |
|---|---|---|
| C1 — Maximal-clique guided | 受计算预算限制的极大团枚举，并按规模 $k$ 抽取子团 | 大图极大团数可指数增长；必须有时间/数量限制 |
| C2 — Local clique expansion | 从节点、普通边或三角团出发递归加入共同邻居 | 只在满足 clique 条件时扩展；实现随机种子固定 |
| C3 — Structural sampling | 对同一 clique 使用不同 $k$ 和采样预算产生多个候选 | 不得依赖测试超边标签选择具体节点 |
| C4 — Feature-driven | 特征 kNN、谱聚类群组等 | 目标 B 直接允许；目标 A 严格无噪声协议中应另标为“非 clique 候选”并区分评价 |
| C5 — Random matching control | 采样度分布和超边规模匹配的随机群组 | 只作为显著性或困难度对照，不应掺入主要恢复候选覆盖率计算 |

固定全局预算 $M_0$、每个 $k$ 的预算和随机种子。**禁止采用“最终真实超边数量”决定候选容量或测试阈值**。可给 $M_0$ 一个与 $N$ 成比例的工程默认值，但这并不代表理论最优 $M$。

首版优先关注 $k\in\{3,4,5\}$，同时记录 size-2 超边；依照数据实际分布，随后扩展到 $k\le8$ 或更大。此范围是计算预算，不是超边普适尺度。

### 6.3 候选召回是模型性能的上界

记 $\mathcal C$ 为无重复候选集合，$\mathcal E^\star$ 为无重复真实超边，则

$$
\mathrm{CandidateRecall}=\frac{|\mathcal C\cap\mathcal E^\star|}{|\mathcal E^\star|}.
$$

**若评分器只能从候选集合中选择超边，则精确超边 Recall 不可能超过 CandidateRecall。**

必须报告预算—覆盖曲线 $\mathrm{Recall@Budget}(M_0)$、不同超边大小下的候选召回，以及“oracle scorer”上界；否则难以区分量子特征失败还是候选生成失败。

### 6.4 困难负样本与避免平凡实验

- **真实正例**：测试协议允许作为评分的 $e\in\mathcal E^\star$；训练使用的正例只来自训练标注部分。
- **困难负例**：投影图中完全为 clique、但不属于真实超边的集合；与正例在规模、度、聚类系数和候选来源上尽可能匹配。
- **过于容易的负例**：任意非 clique 集合；对于无噪声精确投影，它们不可能是真实超边，主要评价不应只靠这类负例。
- **潜在遗漏标注**：真实事件数据不一定完备；把未记录的候选视为负例是数据集建模选择，应报告此限制，尤其是跨时间信息泄漏风险。

训练/验证/测试不允许根据测试 $\mathcal E^\star$ 调整负样本抽样、阈值或候选筛选。

---

## 7. 经典对照特征及“量子究竟新增了什么”的设计

### 7.1 经典候选特征 $z_C(e)$

必须包含足以解释量子短时间展开的基本统计，否则量子特征的优势可能来自不公平的基线：

- 基数 $|e|$、候选来源、节点度的均值/最大值/方差。
- 群组内部/外部边数、加权 cut、conductance 或相近的边界度量。
- 节点共享外部邻居、外部邻接列内积；针对式 (3) 的**精确二阶短时间系数**。
- 三角形/局部 clique 统计、邻域结构重叠、简单图谱特征。
- 若有 $X$：节点特征相似度、特征方差与候选特征质心统计；**不使用测试标签**。
- 在无噪声主协议中所有有效候选可均为 clique，此时“内部边密度”恒为 1，不应将其视为有判别价值的特征。

### 7.2 强制纳入的非量子基线

| 编号 | 输入/评分 | 检验目的 |
|---|---|---|
| C0 | 仅候选基数/度/边界等 $z_C$ | 强经典结构基线 |
| C1 | 常规图随机游走/PPR 或 $e^{-\tau L}$ 热核的群组响应 | 比较耗散扩散 |
| C2 | 经典波核 $\cos(t\sqrt L)$ 及合理归一化后的群组特征 | 排除一般波传播效应 |
| C3 | 多阶 $A^r$、Laplacian 多项式/谱滤波器 | 排除仅为高阶 walk 统计造成的收益 |
| C4 | 短时间 $C(e,t)/t^2$ 对应的外部列内积 | 验证式 (3) 的局部等价限制 |
| Q1 | $P(t)=|U(t)|^{\circ2}$ 的 pairwise 聚合特征 | 判别群组特征是否优于简单量子相似性 |
| Q2 | $R_{\rm inc}$、$R_{\rm coh}$、$C$ 分别加入 $z_C$ | 隔离相干项作用 |
| Q3 | 完整多时间群组特征 $z_Q$ + $z_C$ | 拟议完整量子特征模型 |
| Q4 | 随机相位/相位打乱的群组响应 | 相干对照和不变性诊断 |

在经典波核对照中，须说明其并非幺正量子演化；归一化输入态与输出统计量要保持可比较，不能直接把未归一化波核输出与概率指标混比。

### 7.3 相同候选、相同评分器、相同可见信息

最初用 `StandardScaler + LogisticRegression` 或参数很小的 MLP。每条候选只计算一次真实标签。`fit` 仅在训练 split 上；所有方法用同一候选集合、相同标签、相同交叉验证折和类似模型容量。相较只比较 downstream accuracy，这种设计可以直接归因于特征模块。

### 7.4 增量信息如何度量

主统计量可以是 $\Delta\mathrm{AP}=\mathrm{AP}(z_C,z_Q)-\mathrm{AP}(z_C)$，辅以 AUROC、Log-loss、Brier score、分层 AP。要求配对 bootstrap 或按图/时间块重新抽样的置信区间，并报告多种网络及候选规模的差异；**不能仅凭一次随机划分上胜出宣布有效**。

还应比较残差化后的 $z_Q$，或在经典特征已含精确二阶系数的前提下进行增量预测，防止短时间共同邻居信号被误认作量子独特贡献。

---

## 8. 目标 A：真实超图重建算法（核心主线）

### 8.1 首版：固定候选 + 结构监督的评分与选择

对每个候选 $e\in\mathcal C$：

$$
z(e)=[z_C(e)\Vert z_Q(e)],\qquad p_e=\sigma(f_\theta(z(e))).
$$

若有真实候选标注 $y_e=\mathbf1[e\in\mathcal E^\star]$，以结构监督训练

$$
\mathcal L_{\rm cand}=-\sum_{e\in\mathcal C_{\rm train}}\omega_{y_e}
\big[y_e\log p_e+(1-y_e)\log(1-p_e)\big].
$$

其中类别权重 $\omega_{y_e}$ 只从训练集确定。首版不要让评分器根据测试真实超边数选择 top-$M^\star$；使用训练/验证集确定阈值，或按验证得到的复杂度惩罚选择集合。

**训练/输出顺序**：固定候选 → 离线提取经典/量子特征 → 候选分类器 → 阈值/预算/结构选择 → 预测超边集合 → 对真实超边评估。

### 8.2 结构一致性：解释观测，但不假装具有唯一性

对于候选集合的软保留门控 $g_e\in[0,1]$ 与软成员关系 $b_{ie}\in[0,1]$，提出一个**可选的 noisy-OR 代理投影**（不是客观唯一真实观测机制）：

$$
\widehat A_{ij}=1-\prod_{e=1}^{M_0}(1-g_eb_{ie}b_{je}),\quad i\ne j.
\tag{5}
$$

对二值 $A$ 采用采样的 BCE：

$$
\mathcal L_{\rm proj}=-\sum_{(i,j)\in\mathcal S_{\rm pair}}w_{ij}
\big[A_{ij}\log(\widehat A_{ij}+\epsilon)+(1-A_{ij})\log(1-\widehat A_{ij}+\epsilon)\big].
$$

若所有候选都为 clique，且 $b_{ie}=\mathbf1[i\in e]$，则任何候选不会产生原图不存在的边；此时负边惩罚可能恒为 0，投影一致性主要退化成**边覆盖损失**。

**重要反例**：用所有二元边作为候选就能精确重建 $A$，却未恢复真实三元超边；反之，用一条大的 clique 覆盖也可能重建 $A$。所以式 (5) 单独优化不能识别真实超边，必须结合候选监督、可验证先验与复杂度模型。

### 8.3 结构复杂度与超边数量选择

首选简单可解释的模型选择：

$$
\mathcal L_{\rm select}=
\mathcal L_{\rm cand}
+\lambda_{\rm proj}\mathcal L_{\rm proj}
+\lambda_M\sum_e g_e
+\lambda_I\sum_{i,e}g_eb_{ie}
+\lambda_R\mathcal L_{\rm redundant}.
\tag{6}
$$

这里各项代表候选准确性、投影解释、超边总数、总关联数、重复惩罚；**不是已经证明有效的最终公式**。要针对 $\lambda$ 做消融，并检查是否把真实嵌套超边误删。

也可在后续研究 MDL：

$$
\min_B L(B)+L(A\mid B),
$$

但必须实现可计算的具体编码方案，不能仅以“MDL”命名普通 L1 正则。对于具有真实结构监督的目标 A，验证集超边指标是选择超参数的核心依据；不能用测试真值。

### 8.4 重复超边及重叠保护

软 Jaccard：

$$
J_{ab}=\frac{\sum_i b_{ia}b_{ib}}{\sum_i(b_{ia}+b_{ib}-b_{ia}b_{ib})+\epsilon}.
$$

首版仅删除**规范化后完全相同**的候选；后续可考虑 $J_{ab}>\tau_R$ 的合并或惩罚，但应设置对照实验——真实超图可能存在高度重叠和严格嵌套的合法超边。

### 8.5 目标 A 的两个可发表层级

- **A-MVP**：给现有经典候选加入有数学定义的量子群组特征，检验在真实超边恢复上是否产生一致改善，重点是**机制证据**。
- **A-Advanced**：将群组量子特征与候选选择、结构一致性、软成员优化统一训练，并证明真正带来性能/规模/结构改善，重点是**新推断方法**。

若 A-MVP 没有稳定的增益，不进入 A-Advanced。

---

## 9. 目标 B：没有真实超边时的结构学习与下游验证

### 9.1 总体设计

目标 B 的输入是普通图/属性，**没有真实超边供构建阶段使用**。同样生成候选 $\mathcal C$，计算 $z_C,z_Q$，但不能把目标 A 在不同分布上拟合的真实超边概率直接解释成普适成员概率。

建议比较：

1. **B0：kNN/聚类超图 + 固定 HGNN**。
2. **B1：经典结构特征 + 稀疏候选选择 + 相同 HGNN**。
3. **B2：经典+量子特征 + 结构约束选择 + 相同 HGNN**。
4. **B3：使用训练节点标签联合训练（明确标注 Task-driven）**。
5. **B4：在相同输入上使用不构图的强 GNN 基线**（防止“增加超图”本身也没必要）。

**固定下游架构再比较构图器**，与“改变构图器同时改变 GNN”相区别。

### 9.2 可实施的无监督结构目标

对于初始候选 $B_0$，学习 $g_e\in(0,1)$：

$$
\widetilde B=B_0\operatorname{diag}(g).
$$

先通过式 (5) 的投影一致性、总超边数/总关联数惩罚，选出可解释原图的紧凑候选超图。$g_e$ 可由 $z(e)$ 经 MLP 预测而非每个测试网络独立学习静态参数；首版必须同时运行**不含量子特征**的完全相同选择器。

注意：无监督投影一致性对真实高阶结构不可辨识；目标 B 中的结构是**模型定义的潜在结构**，不保证真实事件意义。

### 9.3 允许任务联合训练时的损失

给定固定 HGNN 参数 $\phi$，仅对训练节点定义交叉熵：

$$
\mathcal L_B=
\mathcal L_{\rm train-label}(\phi,B)
+\alpha\mathcal L_{\rm proj}
+\beta\mathcal L_{\rm complexity}
+\gamma\mathcal L_{\rm stability}.
$$

结构损失的具体形式必须实现后校验其退化现象。$\mathcal L_{\rm stability}$ 可约束在轻微边/特征扰动下同一节点集合的软关联变化，但注意跨扰动候选超边需要可对齐，否则不可直接对矩阵列做 MSE。

### 9.4 下游传播算子

若 $B\in\mathbb R_{\ge0}^{N\times M}$，给定非负超边权重 $W_E$：

$$
D_E=\operatorname{diag}(B^\top\mathbf1_N),\qquad
D_V=\operatorname{diag}(BW_E\mathbf1_M),
$$

$$
\Theta_B=D_V^{-1/2}BW_ED_E^{-1}B^\top D_V^{-1/2}.
$$

零度需要稳定处理。示例 HGNN：

$$
Z^{\ell+1}=\sigma(\Theta_BZ^\ell W^\ell).
$$

请注意：采用软 $B$ 的以上算子是自然扩展，但要与选择的具体 HGNN 实现所采用的超边权重、度矩阵及自环处理保持一致。**量子超图构建的效果必须在相同传播模型和训练协议下比较。**

---

## 10. 进阶方向：允许调整成员关系，而非仅筛选固定候选

### 10.1 首版不要端到端改变初态

首版固定离散候选成员，先建立可检验的 $z_Q$；否则若 $B$ 改变初态、初态又改变 $z_Q$、$z_Q$ 再指导 $B$，容易形成循环优化和无意义的极值。

### 10.2 第二版可微成员与软初态

若 Gate 1、Gate 2 都通过，则可以对候选支持集 $S_e$ 上的节点定义

$$
b_{ie}=\sigma(a_{ie})\mathbf1[i\in S_e],
\quad |s_e(B)\rangle=\frac{b_{:e}}{\|b_{:e}\|_2+\epsilon}.
$$

考虑空候选时需专门定义行为，不能仅在分母加 $\epsilon$ 而默认它仍为归一化态。更稳妥的约束是门控为 0 的候选跳过量子响应计算，非空候选的向量严格归一化。

软投影观测量可设为 $\Pi_e(B)=\operatorname{diag}(b_{:e})$，但它**不是正交投影**（除非成员为 0/1）。此时

$$
R_{\rm soft}(e,t)=\langle s_e(B)|U^\dagger\Pi_e(B)U|s_e(B)\rangle
$$

仍是非负加权观测量，却不再具有与硬投影完全相同的解释；性质推导和数值测试必须同步更新。另一种实现是仅训练时使用 soft $B$，评价时离散化后重新计算硬群组响应。**禁止把硬投影的证明原样套到软对角权重上。**

### 10.3 重新生成候选与超边数量

固定候选池 + 稀疏门控主要解决**候选选择**，不能产生从未提出的新超边。若要真正学习新成员关系，可采用：

- 在候选 clique 的允许支持范围内增加/删减成员；
- 基于当前节点嵌入定期重新生成候选池；
- 新提议候选与旧候选合并、精确去重；
- 训练后用独立结构准则决定最终 $M_{\rm eff}$，而不是直接设置 $M=N$。

但候选重新生成可能涉及不可微组合操作，可采用交替训练或显式两阶段策略，并报告提议刷新次数及计算开销。

---

## 11. 理论分析任务清单：哪些应证明，哪些只能假设？

| 编号 | 命题 | 状态 | 需要的工作 |
|---|---|---|---|
| T1 | $U(t)$ 幺正，$P(t)$ 双随机 | 标准数学性质 | 明确 $H$ Hermitian；实现数值验证 |
| T2 | 群组干涉恒等式 (1) | 已推导 | 手推、穷举小图验证 |
| T3 | 随机相位期望公式 (2) | 已推导 | 穷举符号 / 蒙特卡罗验证 |
| T4 | 小时间展开 (3) | 已推导（首版实对称 $H$） | 验证 $O(t^4)$ 误差随 $t$ 的规律 |
| T5 | 置换不变性 | 依赖组件等变性，可证明 | 图与候选共同置换测试 |
| T6 | 有限时间扰动界 (4) | 标准推导 | 数值扰动及保守界说明 |
| T7 | 同一普通图投影存在不同超图 | 已知不可辨识性 | 构造反例；准确区分与领域先验的关系 |
| T8 | 真实超边与假 clique 的 $z_Q$ 存在可区分分布 | **研究假设** | 在明确生成模型上提出并验证/反驳 |
| T9 | 中等时间相干项优于经典结构统计 | **研究假设** | 公平的强经典基线及理论上界分析 |
| T10 | 量子构图改善真实恢复和任务性能 | **研究假设** | 跨数据、跨随机种子、配对统计检验 |

### 11.1 进一步可尝试的条件性理论结论

在某个有明确结构参数的随机超图生成模型上（例如指定 $d$-uniform HSBM、超边概率、群组重叠机制、属性噪声，并显式给出投影过程），定义正例 $e^+$ 与在度/基数上匹配的假 clique $e^-$. 尝试分析：

$$
\Delta_C(t)=\mathbb E[C(e^+,t)]-\mathbb E[C(e^-,t)].
$$

接着求二者的方差、样本复杂度和可分离区域。**若 $\Delta_C(t)=0$ 或在噪声下不可稳定区分，必须如实报告；不能把计算出的非零值事先写成定理。**

### 11.2 与真正不可约高阶关系的距离

$C(e,t)$ 的主体是相干振幅之间的成对交叉项；这是一种**集合条件化的动态响应特征**。如拟证明它识别无法由 pairwise 机制解释的“genuine higher-order interaction”，还需要额外的可观测量、生成假设、独立结构真值或三体项等证据。首版论文应优先使用 *latent hyperedge discrimination / structure inference*，避免过度宣称 *true multi-body interaction detection*。

---

## 12. 数值计算：不构造稠密 $U(t)$ 的实现原则

### 12.1 小图正确性参考实现

在 $N\lesssim500$ 的原型规模（具体取决于计算资源）上，可以使用 `scipy.linalg.expm` 得到完整 $U(t)$ 作为 **reference oracle**，仅用于单元测试和复杂度小的验证。

### 12.2 稀疏矩阵指数作用

对于候选 $e$，真正需要的是

$$
Y(t)=e^{-itH_Q}S,
$$

其中 $S$ 的列是多个候选初态（相干/随机相位）或必要的基向量。使用 `scipy.sparse.linalg.expm_multiply`，而不是构造完整 $N\times N$ 稠密矩阵指数。

`expm_multiply((-1j * t) * H, S)` 返回 $N\times b$ 矩阵；之后只读取每个候选对应的内部节点响应。对多候选分批处理；不可把所有候选和所有时间点的 $N\times M_0$ 稠密中间量一次性驻留内存。

### 12.3 非相干参考的两种方式

- **精确方式**：对候选 $e$ 的 $k$ 个节点基态分别演化、汇总群组内部概率；适合小候选或验证 oracle。
- **随机相位方式**：每个候选演化 $R$ 个 Rademacher 随机初态，得到 $R_{\rm inc}$ 无偏估计。按预算选取 $R$，保存估计方差，避免因不同候选方差不同而产生排序偏差。

如果每个候选群组很小、节点基态在候选之间大量复用，**预计算重复锚点演化**也可能比随机相位更快，需做 profiler 对比；不要无条件认为随机相位总是更便宜。

### 12.4 复杂度实情

- 稠密全谱分解一般 $O(N^3)$ 时间、$O(N^2)$ 内存。
- 稀疏矩阵指数对向量的作用不需要同等的稠密分解，但实际时间取决于非零元数量、谱范数、时间长度、误差容限、候选初态数及数值实现。
- 每条超边的实际代价还包括候选生成、向量批量传播、评分及结构选择。必须报告**端到端时间和峰值内存**，不能只报告 GNN 训练时间。
- 阶段 P3 若尝试端到端对 $H$ 与 $B$ 求导，需切换到支持自动微分的近似（如 Chebyshev/Krylov 或小图 `torch.matrix_exp`），并用有限差分/高精度 oracle 检查梯度。

## 13. 实验数据集：来源、真实超边语义、优先级与限制

**只选能独立核对来源的真实数据；以下统计为发布页给定的原始版本，不等于经过本项目筛选后的规模。** 下载后必须在 `data_manifest.json` 中记录版本、URL、下载时间、原始文件 hash、处理脚本 hash、真实超边定义与数据许可证。

### 13.1 目标 A：有真实超边记录的数据集（重建）

| 优先级 | 数据集 | 原始公开规模/语义 | 作用 | 源地址 |
|---|---|---|---|---|
| A-Pilot 1 | **email-Enron（Cornell temporal hypergraph）** | 143 节点；10,883 个带时间事件、1,542 个独特 simplex（发布页限制单事件最多 25 节点）。一次邮件的发件人及收件人形成群组 | 首个真实事件重建；易于复核 | https://www.cs.cornell.edu/~arb/data/email-Enron/ |
| A-Pilot 2 | **contact-primary-school（labeled static version）** | 242 节点；12,704 条发布页定义的超边；平均规模约 2.4，最大 5 | 小规模、高重叠、接触类型 | https://www.cs.cornell.edu/~arb/data/contact-primary-school-labeled/ |
| A-Pilot 3 | **contact-high-school（labeled static version）** | 327 节点；7,818 条发布页定义的超边；平均规模约 2.3，最大 5 | 另一接触场景；检查跨数据泛化 | https://www.cs.cornell.edu/~arb/data/contact-high-school-labeled/ |
| A-Extension | **coauth-DBLP（temporal hypergraph）** | 原始限制版约 1,924,991 节点、3,700,067 条带时间 simplex；每篇论文的作者集合为一条事件超边 | 真正的合作事件；规模与分布挑战 | https://www.cs.cornell.edu/~arb/data/coauth-DBLP/ |
| A-Extension | **email-Eu、tags 数据** | 同一 Cornell higher-order 数据集集合，具体规模以逐个数据发布页为准 | 跨领域鲁棒性 | https://www.cs.cornell.edu/~arb/data/ |

**数据版本警告 1**：Cornell 的 `email-Enron` temporal hypergraph（143 节点）与 SNAP 的 `email-Enron` 普通图（36,692 节点）不是同一套超图数据，不能混用来声称同一个 ground truth。 

**数据版本警告 2**：`contact-high-school` 的 temporal version 按 20 秒窗口把成对接触的 maximal cliques 定义为 simplex；它们是**基于接触测量规则生成的操作性群组**，不等同于额外独立证明的真实不可约多人接触。需要注明实验使用 static labeled 版本还是 temporal 版本，不要将不同文件的超边数量混写。

**数据版本警告 3**：`coauth-DBLP` 太大，首版不能直接做全部候选团枚举和矩阵指数。只允许先做公开且可复现的**完整事件采样**或时间分片；不要先截取节点再把超边交集当作原来的真实事件。若使用诱导子超图，明确采取“仅保留所有成员均在子图中的完整事件”，报告这会改变边基数分布，并避免宣称子样本与原始分布相同。

**数据版本警告 4**：重复同成员事件和独特超边的统计不同。目标 A-MVP 主指标使用**去重后的节点集合**；另开带权/重数协议研究重复事件，避免在 binary projection 下宣称恢复重数。

### 13.2 目标 B：没有真实超边标注的普通属性图

| 数据组 | 推荐数据集 | 用途及关键注意事项 | 公开实现入口 |
|---|---|---|---|
| 同配 citation 图 | Cora、CiteSeer、PubMed | 标准节点分类，**不等同于真实超边重建**；遵守固定划分 | https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.Planetoid.html |
| WebKB 异配图 | Cornell、Texas、Wisconsin | 检查异配性与小数据不稳定；多划分重复报告 | https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.WebKB.html |
| 其他异配图 | Actor、Chameleon、Squirrel | 控制特殊版本、重复节点和划分问题；注明 PyG 数据加载器/版本 | https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.Actor.html |
| 可选新版异配 benchmark | Roman-empire、Amazon-ratings 等 | 扩展泛化性；以正式数据源/加载器核对为准 | https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.HeterophilousGraphDataset.html |

建议先从 **Cora / CiteSeer** 实现端到端构建，再在模型算力与候选数量可承受时扩展 PubMed。Cora/其他图的节点标签**不能用于目标 A 的真实超边真值**。

### 13.3 人工合成数据：应是科学实验而不仅是跑通代码

实现固定随机种子的真实超图生成器：

```text
SyntheticHypergraphGenerator(config, seed):
    Sample n vertices and optional latent blocks/attributes
    Sample a hyperedge-size law, optionally d-uniform
    Sample hyperedge memberships from stated within-/between-block rules
    Control overlap, nesting, node degree, hyperedge count and noise
    Preserve exact hyperedge event list E_star
    Construct binary clique projection A
    Optionally produce multiplicity projection W
    Optionally perturb A (separate noisy protocol)
    Return {A, W?, X?, E_star, parameters, seed}
```

建议首批参数网格（仅作为起点，不表示已优化）：

| 因素 | 建议测试设置 | 研究问题 |
|---|---|---|
| $N$ | 60、120、240、500 | 计算可扩展性与规模偏差 |
| 基数分布 | 固定 $d=3/4$；混合 $2\dots6$ | 固定尺度与混合尺度 |
| 重叠 | 低、中、高 | 真实超边被投影混淆时的判别能力 |
| 超边密度 | 稀疏、中等、较密 | 匹配随机超图可恢复阈值的研究方向 |
| 节点属性 | 无、弱、强类相关 | 特征与拓扑贡献拆解 |
| 观测噪声 | 0、少量删边/加边、较强噪声 | 稳定性与 clique 条件是否适用 |
| 重数 | binary $A$ / multiplicity $W$ | 输入信息量的影响 |

**四个必须实现的特殊合成实验**：

- S1：投影中真实 clique 与非真实 clique 较容易区分。
- S2：超边高度重叠，产生大量虚假 maximal clique 或候选子团。
- S3：构造两个具有相同 $A,X$ 但不同 $\mathcal E^\star$ 的反例；量子/经典必须表现出同样的观测歧义。
- S4：保留相同图统计（度、候选规模、局部 clustering），改变可辨识的高阶群组生成机制；检验“量子特征”是否对混杂变量敏感。

**不要将“生成真实超图后求 clique projection”误当作保证可恢复。** S3 就是专门证明不保证的实验。

---

## 14. 目标 A：详细实验矩阵与指标

### 14.1 A0：理论及数值 sanity checks

- 幺正性：$\|U^\dagger U-I\|$ 小于规定数值容差。
- 双随机性：$|U|^2$ 每行/每列之和约等于 1。
- 群组响应在 $[0,1]$，$C$ 在 $[-1,1]$。
- $t=0$、$e=V$、孤立分量与三角形不可辨识反例。
- (1) 干涉交叉项与直接计算一致。
- (2) 穷举符号态的平均等于非相干响应。
- (3) 对 $t\to0$ 验证二阶系数，检查 $O(t^4)$ 数值误差。
- (4) 添加小对称扰动，验证上界与节点置换不变性。

任何一条失败先修复理论/代码，不能直接训练复杂模型掩盖错误。

### 14.2 A1：固定候选的判别实验（首要科学关卡）

**输入**：固定 $A$、相同 $\mathcal C$、结构监督标签（仅训练 split）；所有方法看到相同信息。  
**比较**：C0-C4、Q1-Q4（第 7 节）。  
**指标**：AUROC、AP、Brier/log-loss、Precision@Budget、不同 $k$ 的 AP、单位时间 AP 改善、3–5 个或以上预注册随机种子的配对结果。  
**报告**：候选预算、正例基率、每组真实正例和困难负例数量、置信区间、评分器训练时间与特征计算时间。

**重点分析**：真实超边 $e^+$ 与匹配的虚假 clique $e^-$ 是否具有稳定的 $z_Q$ 分布差异。画正负样本在 $C(e,t)$、$R_{\rm coh}$ 等维度上的分布，并按 $k$、节点度与边界强度分层。

### 14.3 A2：真正的超边集合重建

固定候选生成与结构选择策略，只替换输入特征。输出 $\widehat{\mathcal E}$ 后，使用下列指标：

**精确匹配**（预测超边成员集与真实成员集完全相同，先去重）：

$$
TP=|\widehat{\mathcal E}\cap\mathcal E^\star|,
\quad FP=|\widehat{\mathcal E}\setminus\mathcal E^\star|,
\quad FN=|\mathcal E^\star\setminus\widehat{\mathcal E}|.
$$

$$
\mathrm{Precision}=\frac{TP}{TP+FP},\quad
\mathrm{Recall}=\frac{TP}{TP+FN},\quad
F_1=\frac{2PR}{P+R},\quad
J=\frac{TP}{TP+FP+FN}.
$$

空预测/空真实情况必须在评估函数中明确约定，以免 0/0 被 silently 忽略。

**近似成员匹配**：在明确阈值的集合 Jaccard $J(e,\hat e)$ 上进行一对一最大权匹配（例如匈牙利算法，仅在规模可承受时使用），计算 matched-set 平均 Jaccard、成员级 Precision/Recall；**不能用宽松指标代替精确指标**。

**结构级**：超边数 $M_{\rm eff}$、基数分布、节点超度分布、超边重叠统计、孤立节点比例、投影图覆盖误差。用绝对误差/分布距离，并记录置信区间。

### 14.4 A3：公平的已发表基线

| 基线 | 适用条件 | 官方来源/说明 |
|---|---|---|
| Maximal-clique reconstruction | 二值投影且候选可枚举 | 基础上界/失败反例对照 |
| k-clique sampling / simple candidate classifier | 相同候选预算 | 自行实现，不能声称复现 SHyRe |
| SHyRe（ICLR 2024） | 有应用域结构监督 | https://proceedings.iclr.cc/paper_files/paper/2024/hash/e92b8441b702d142435f243806f9ec5a-Abstract-Conference.html |
| Bayesian reconstruction（Communications Physics 2021） | 根据其具体观测模型及先验 | https://www.nature.com/articles/s42005-021-00637-w |
| MARIOH（ICDE 2025） | **边重数可用**时的专门协议 | https://doi.org/10.1109/ICDE65448.2025.00233 |
| Structure-only MLP | 固定候选、相同数据 | 排除额外评分容量带来的收益 |

**实践要求**：若官方代码不能成功复现、输入假设不一致或计算成本过高，应标记 `not comparable / not reproduced`，保留可运行自实现基线并解释原因；**不得捏造官方基线指标**。

### 14.5 A4：泛化、鲁棒性、成本

- 跨网络/跨时间：使用完整独立测试网络评估；做 time-aware split 而非随机打散未来事件。
- 结构噪声：按协议添加/删除 $A$ 的边；记录 $\|\Delta A\|$ 与归一化 $H$ 的变化（此处在代码中使用矩阵范数）。
- 模型误差归因：按候选未覆盖、评分错误、超边选择错误拆开错误来源。
- 大小分层：$k=2$、3、4、5、$\ge6$ 分别报告；不允许一个类占绝大多数掩盖其他类。
- 计算：候选生成、量子演化、训练、结构优化、推理各自的壁钟时间和峰值内存。

---

## 15. 目标 B：详细节点分类和结构学习实验

### 15.1 固定下游模型的主实验

对每个数据集固定相同 HGNN（必要时再加 AllSet），只替换 $B$ 的构建方式：

| 构建方法 | 下游学习器 | 允许信息 | 作用 |
|---|---|---|---|
| kNN / cluster $B$ | HGNN | $A,X$ | 简单经典基线 |
| 经典群组特征 $z_C$ | 同一 HGNN | $A,X$ | 检验结构选择机制 |
| 量子 pairwise $P(t)$ | 同一 HGNN | $A,X$ | 检验仅换相似度的价值 |
| 群组 $z_Q+z_C$ | 同一 HGNN | $A,X$ | 主方法 |
| classical wave / spectral | 同一 HGNN | $A,X$ | 复杂经典动力学对照 |
| 任务联合优化 $B$ | 同一 HGNN | $A,X,y_{\rm train}$ | 额外标签驱动版本，单独列结果 |
| GCN/GAT/MLP 等 | 各自原网络 | $A,X,y_{\rm train}$ | 不构图或普通图基线 |

注意“同一 HGNN”包括相同层数、隐藏维度、训练次数、优化器、验证/早停标准及标签划分；超参数可以通过预先规定的同预算验证搜索调整，而非偏向主方法。

### 15.2 主要指标与评估流程

- Accuracy、Macro-F1、必要时 Micro-F1，给均值±标准差或跨划分置信区间。
- 对小型异配图，优先使用既定多个划分，而不是只报告一组随机种子。
- 无监督 B1 与任务驱动 B2 分开评价，避免标签偏置与结构构建效果混合解释。
- 结构输出：预测超边数、基数/重叠分布、重复超边、空超边、孤立节点与模型参数量。
- 压力测试：删边/加边、节点特征噪声、减少训练标签、不同图规模与演化时间。
- 内存和总时间，包括量子特征计算，尤其在 PubMed 上不能漏报预处理成本。

### 15.3 与目标 A 的结果如何联合解释？

| A（恢复） | B（分类） | 正确解释 |
|---|---|---|
| 改善 | 改善 | 支持两种任务均有效，但仍不能宣称普适量子加速 |
| 改善 | 不改善 | 对真实关系推断有价值；下游标签任务可能不需要此类结构 |
| 不改善 | 改善 | 量子特征可能是有用的任务归纳偏置，不证明真实高阶结构 |
| 不改善 | 不改善 | 缺少继续复杂化当前量子框架的证据 |

---

## 16. 消融与统计显著性：把“为什么有优势”变成可验证论断

### 16.1 必做消融表

| 消融轴 | 对照实验 | 预期检验的解释 |
|---|---|---|
| Hamiltonian | $L_A$、$A$、$L_A+L_X$ | 是否只是特征或缩放差异 |
| Time | 单 $t$、固定多 $t$、有限平均 | 是否只在偶然时刻有效 |
| Coherence | $R_{\rm coh}$、$R_{\rm inc}$、$C$、random-phase | 相干干涉增益 |
| Classical | heat/PPR、wave、spectral polynomial、二阶近似 | 是否为普通谱/共同邻居特征的替代品 |
| Candidate | clique-guided、feature、local expansion | 候选生成是否决定全部结果 |
| Structure | no projection、no complexity、no redundancy | 哪个约束有实质作用 |
| Supervision | 结构监督、无监督、分类监督 | 标签偏置与先验 |
| Hyperedge count | 固定 $M_0$、验证预算、门控选择 | 自适应性是否有效 |
| Noise | 投影噪声、特征噪声、分布变化 | 稳定性 |

### 16.2 统计检验协议

- 预先规定主要比较：`Q3 vs C0`、`Q3 vs C2`、`Q3 vs C3`，不得只挑最佳对照。
- 在**配对**的 graph/split/seed 上计算差异；尽可能用按图/时间块的 bootstrap CI，不要将强相关候选当作独立统计样本。
- 报告各数据集逐项结果，而非仅有平均排名；说明超参数搜索预算。
- 当 $p$ 值或多重比较被使用时，声明检验方法、校正方式与主要终点，不能仅凭未校正小 $p$ 值解释理论性质。
- 因难度/基率变化，AP 要同时报告正例基率；AUROC 不应单独作为高度不平衡候选任务的主要指标。

### 16.3 三个 Go/No-Go 判断门

**Gate 1：机制价值**  
在至少两个不同合成结构条件和两个真实数据协议上，与主要强经典基线进行配对比较；至少在明确的可解释条件下，$z_Q$ 表现出有意义且可重复的增量识别力，且没有不可接受的复杂度。若没有，停止完整模型开发，回到理论分析。

**Gate 2：重建价值**  
候选覆盖率足够且在目标 A 的真实超边集合恢复上，量子模型的收益可归因于其结构特征而非候选/标签/模型容量变化。若没有，将论文定位为量子图结构表征的负面或边界研究，不宣称超图恢复优势。

**Gate 3：应用与成本**  
固定 HGNN 的目标 B 结果具有跨划分稳定性，并且端到端总计算代价可接受；若仅在少数任务上有效，明确限定适用范围。

这些门**不预先规定虚假的成功率阈值**；在正式实验前可通过预注册方案确定最小有意义效应量，并在报告中保持不变。

---

## 17. 项目代码架构、接口与文件格式（交给代码模型）

### 17.1 目录结构

```text
quantum_hypergraph/
├── README.md
├── pyproject.toml
├── configs/
│   ├── synthetic_p0.yaml
│   ├── pilot_reconstruction.yaml
│   ├── real_reconstruction.yaml
│   └── node_classification.yaml
├── data/
│   ├── raw/                  # 数据下载后只读，含 LICENSE/README
│   ├── processed/            # 缓存与 manifest，保留版本信息
│   └── synthetic/
├── qhyper/
│   ├── graph_io.py           # 邻接、特征、超边 event list
│   ├── projection.py         # 从真实超边生成 A 和可选 W
│   ├── synthetic.py          # HSBM、重叠控制及不可辨识 toy
│   ├── hamiltonian.py        # L_A、L_X、H_Q、归一化检查
│   ├── quantum_action.py     # expm_multiply 与批处理
│   ├── group_response.py     # R_coh, R_inc, C, time grids
│   ├── classic_features.py   # degree, cut, common-neighbors, heat, wave
│   ├── candidate_proposal.py # clique/subclique 及候选预算
│   ├── scorer.py             # Logistic/MLP 评分器
│   ├── structure_selector.py # 阈值、预算、门控、去重
│   ├── soft_incidence.py     # 可选 P3
│   ├── hypergraph_network.py # 固定 HGNN / AllSet wrapper
│   ├── splits.py             # 时间/跨图/随机切分，泄漏检查
│   ├── metrics.py            # exact/soft recovery, AP, F1, runtime
│   ├── plotting.py           # 诊断图与表
│   └── experiment_logger.py # 运行环境、配置、哈希、seed
├── scripts/
│   ├── 00_verify_theory.py
│   ├── 01_generate_synthetic.py
│   ├── 02_prepare_real_data.py
│   ├── 03_propose_candidates.py
│   ├── 04_extract_features.py
│   ├── 05_train_candidate_scorer.py
│   ├── 06_evaluate_reconstruction.py
│   ├── 07_train_node_classification.py
│   └── 08_run_ablation.py
├── tests/
│   ├── test_projection.py
│   ├── test_hamiltonian.py
│   ├── test_group_response.py
│   ├── test_permutation.py
│   ├── test_splits_no_leakage.py
│   └── test_metrics.py
└── results/
    └── RUN_ID/
        ├── config_resolved.yaml
        ├── environment.json
        ├── data_manifest.json
        ├── candidate_coverage.json
        ├── feature_diagnostics.csv
        ├── reconstruction_metrics.json
        ├── classification_metrics.json
        ├── runtime_memory.json
        ├── predictions.jsonl
        └── figures/
```

### 17.2 基础数据对象接口

**GraphRecord**：

```python
{
  "graph_id": "string",
  "num_nodes": 143,
  "adjacency_csr": "scipy.sparse.csr_matrix, shape (N,N)",
  "features": "ndarray (N,d) or None",
  "hyperedges_truth": "list[tuple[int,...]] or None",
  "hyperedge_event_ids": "list[str] or None",
  "hyperedge_timestamps": "list[number] or None",
  "observation_mode": "binary | multiplicity",
  "split": "train | validation | test",
  "provenance": "source URL + file hashes + preprocessing config"
}
```

**CandidateRecord**：

```python
{
  "candidate_id": "canonical hash of sorted node IDs + graph ID",
  "node_ids": [1, 4, 11],
  "size": 3,
  "proposal_source": "local_clique | maximal_clique | feature | ...",
  "classical_features": "float vector",
  "quantum_features": "float vector",
  "label_if_training": "0 | 1 | None",
  "prediction_score": "float or None"
}
```

**严格要求**：所有超边必须按排序节点元组规范化；保持内部节点 ID 连续映射及原始 ID 的双向对应；同候选样本不能跨拆分重复出现（若因协议必要共享，必须清楚报告 transductive 设置）；测试标签不落入任何特征缓存。

### 17.3 核心函数建议签名

```python
project_hypergraph(hyperedges, n_nodes, mode="binary") -> sparse_matrix
build_hamiltonian(A, X=None, kind="norm_laplacian", lam=0.0) -> sparse_matrix
propose_candidates(A, X=None, budget=..., min_k=3, max_k=5, seed=...) -> list[tuple]
quantum_action(H, initial_states, times, batch_size=...) -> generator
coherent_group_response(H, candidate, times, exact_incoherent=False, n_phase_samples=...) -> dict
classical_candidate_features(A, candidate, X=None) -> ndarray
train_candidate_scorer(records, config) -> fitted_model
evaluate_candidate_predictions(records, scores, split) -> dict
evaluate_hyperedge_sets(predicted_edges, true_edges) -> dict
construct_soft_incidence(candidate_edges, gates, n_nodes) -> sparse_matrix
train_fixed_hgnn(A, X, y, masks, B, config) -> dict
```

### 17.4 训练/预测流程伪代码

```python
# P1/P2: 先完成目标 A 的固定候选版本
for graph in dataset:
    A, X = load_allowed_observations(graph)  # no test truth
    H = build_hamiltonian(A, X, kind="norm_laplacian", lam=0.0)
    C = propose_candidates(A, X, fixed_budget, seed=seed)

    # Same C for all methods; use ground truth ONLY for permitted labels/eval
    Zc = extract_classical_features(A, X, C)
    Zq = extract_group_quantum_features(H, C, time_grid, seed=seed)
    cache_without_test_labels(graph.id, C, Zc, Zq)

for variant in ["classical", "heat", "wave", "quantum_pair", "quantum_group"]:
    model = train_on_train_graphs_or_approved_train_split(variant)
    params = choose_on_validation_only(model)
    scores = predict_on_held_out_graphs(model, params)
    E_hat = select_hyperedges(scores, validation_selected_rule)
    evaluate_on_held_out_truth(E_hat)

# P4: downstream node classification, without any ground-truth hyperedges
for data in ordinary_graph_datasets:
    for builder in fixed_builder_list:
        B = builder(A, X, allowed_training_labels_only_if_task_driven)
        metrics = train_and_evaluate_same_hgnn(B, fixed_split, fixed_budget)
```

### 17.5 配置示例（建议初值，不是已调优结果）

```yaml
project:
  name: quantum_hypergraph
  task: reconstruction
  seed: 42
  device: cpu

observation:
  projection: binary
  allow_node_features: false
  use_test_hyperedges_as_input: false

hamiltonian:
  type: normalized_laplacian
  scale: 0.5
  feature_laplacian_weight: 0.0
  isolated_node_diagonal: 0.0

time:
  points: [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
  average_window: null

candidate:
  generators: [local_clique, maximal_clique_subsets]
  min_size: 3
  max_size: 5
  budget: 5000
  enforce_clique_in_exact_projection: true

quantum:
  coherent_state_phase: zero
  incoherent_reference: exact_if_small_else_rademacher
  phase_samples: 16
  use_full_dense_U_only_in_tests: true

scorer:
  model: logistic_regression
  standardize_on_train_only: true
  selection_threshold_from_validation: true

comparison:
  same_candidates: true
  compare_to: [classical, heat, wave, common_neighbors, quantum_pair, quantum_group]

reporting:
  exact_hyperedge_metrics: true
  by_cardinality: true
  candidate_coverage: true
  runtime_and_peak_memory: true
```

### 17.6 交给代码模型的开发顺序（必须遵循）

1. 实现数据对象、二值/加权投影，验证同一个真实超图投影可复现。
2. 实现 Hamiltonian、稀疏演化和高精度小图 oracle。
3. 实现第 5 节公式的单元测试：先验证数学，再跑数据。
4. 实现**固定候选**发生器和候选覆盖率；记录每类候选来源。
5. 实现强经典特征及其所有对照；不要只实现量子方法。
6. 离线缓存 $z_C,z_Q$，严格防止测试标签进入缓存。
7. 实现候选分类器、验证集模型选择及配对统计。
8. 对真实超图输出离散集合，评价精确和宽松恢复指标。
9. Gate 1/2 通过后，开发门控/软成员/结构约束。
10. 最后才接入固定 HGNN，进行目标 B 节点分类对比。

---
## 18. 必须实现的独立数值单元测试（可直接作为代码模型验收样例）

以下测试面向小图，使用稠密 `scipy.linalg.expm` 作为参考真值。用于数学正确性验证，不是已取得的科研实验结果。

```python
import itertools
import numpy as np
from scipy.linalg import expm


def reference_group_response(H, nodes, t):
    H = np.asarray(H, dtype=float)
    assert np.allclose(H, H.T, atol=1e-12)
    n = H.shape[0]
    idx = np.array(sorted(set(nodes)), dtype=int)
    k = len(idx)
    assert 2 <= k <= n

    U = expm(-1j * t * H)
    s = np.zeros(n, dtype=complex)
    s[idx] = 1 / np.sqrt(k)

    R_coh = np.sum(np.abs((U @ s)[idx]) ** 2).real
    R_inc = np.sum(np.abs(U[np.ix_(idx, idx)]) ** 2).real / k
    C = R_coh - R_inc
    return float(R_coh), float(R_inc), float(C), U


def test_group_identities():
    rng = np.random.default_rng(2026)
    n = 7
    A = rng.uniform(0, 1, size=(n, n))
    A = (A + A.T) / 2
    np.fill_diagonal(A, 0)
    deg = A.sum(axis=1)
    H = (np.eye(n) - A / np.sqrt(np.outer(deg, deg))) / 2

    e = np.array([0, 2, 4], dtype=int)
    k = len(e)
    for t in [0., 0.1, 0.5, 1., 2.]:
        rc, ri, c, U = reference_group_response(H, e, t)

        assert 0 <= rc <= 1 + 1e-10
        assert 0 <= ri <= 1 + 1e-10
        assert -1 - 1e-10 <= c <= 1 + 1e-10
        assert np.allclose(U.conj().T @ U, np.eye(n), atol=1e-10)
        P = np.abs(U) ** 2
        assert np.allclose(P.sum(axis=0), 1, atol=1e-10)
        assert np.allclose(P.sum(axis=1), 1, atol=1e-10)

        # Cross-interference formula (1)
        interference = 0.
        for a, b in itertools.combinations(e, 2):
            interference += (2 / k) * np.real(
                np.sum(U[e, a] * np.conjugate(U[e, b]))
            )
        assert np.allclose(c, interference, atol=1e-10)

        # Exhaustive Rademacher average formula (2)
        average = 0.
        for signs in itertools.product([-1., 1.], repeat=k):
            s = np.zeros(n, dtype=complex)
            s[e] = np.array(signs) / np.sqrt(k)
            average += np.sum(np.abs((U @ s)[e]) ** 2) / 2**k
        assert np.allclose(average, ri, atol=1e-10)

    # Short-time formula (3): limit C(t)/t^2
    outside = np.setdiff1d(np.arange(n), e)
    columns = H[np.ix_(outside, e)]
    short_coeff = -(2 / k) * sum(
        np.dot(columns[:, a], columns[:, b])
        for a, b in itertools.combinations(range(k), 2)
    )
    _, _, c_small, _ = reference_group_response(H, e, 1e-3)
    assert abs(c_small / 1e-6 - short_coeff) < 1e-5

    # Trivial e=V gives zero coherence contrast
    _, _, c_all, _ = reference_group_response(H, range(n), 2.)
    assert abs(c_all) < 1e-10

    print("All reference identities passed")


if __name__ == "__main__":
    test_group_identities()
```

### 18.1 除上述样例，还需补齐以下测试

- **置换测试**：使用随机节点置换，同时置换 $A,X,e$，应得到相同的 $R_{\rm coh}$、$R_{\rm inc}$、$C$；不仅检查固定 Hamiltonian，还要重新运行 Hamiltonian 构造器。
- **扰动上界测试**：生成实对称 $\Delta H$，比较响应变化与式 (4)；数值误差容差单独设置。
- **两个三角超图不可辨识反例**：两个不同真实超边集合生成同一 $A$，构建器与量子特征必须一致。
- **候选覆盖测试**：预测只能来自 $\mathcal C$ 时，实际精确 Recall 不得超过 CandidateRecall（考虑浮点或去重情况）。
- **不使用测试标签测试**：在任何构建/候选/特征阶段随机替换测试标签，生成结果必须不变（仅最终评价改变）。
- **软关联测试**：所有元素非负、结构算子维度正确；对全零列、全零行、完全重叠列有定义的行为。
- **数据版本测试**：不同同名数据集不能因为路径名字近似而自动合并或视为同一个真值来源。
- **随机性测试**：同一随机种子/软件环境重复运行的缓存 hash 与主要结果一致（若 GPU 有非确定算子，必须记录并界定误差）。

---

## 19. 预期的可视化、实验结果组织与论文写作结构

### 19.1 必要图表

1. **Framework**：普通图输入 → 共用候选池 → 经典/量子群组特征 → 候选评分与结构选择 → A/B 两条评估分支。
2. **Mechanism**：同一候选 $e$ 上 $R_{\rm coh}$、$R_{\rm inc}$、$C$ 随 $t$ 变化；展示相位随机化对照。
3. **Theory check**：$C(t)/t^2$ 与式 (3) 的短时间系数对比，明确“量子特征可能退化为经典统计”的局限。
4. **Candidate discriminability**：真实超边 vs 规模/度匹配假 clique 的 AP 与特征分布。
5. **Candidate budget curve**：Recall@Budget，显示候选覆盖的性能上界。
6. **Reconstruction**：精确超边 Precision/Recall/F1/Jaccard，按基数与重叠强度细分。
7. **Model ablation**：经典/量子/相干/去相干/经典波核多组消融。
8. **Robustness**：不同加边/删边、噪声、超边重叠及跨域设置。
9. **Efficiency**：候选、量子特征、训练与推理分别计时；时间和峰值内存随 $N,M_0$ 变化。
10. **Downstream B**：固定 HGNN 上不同超图构建器的分类表现，并单独给出结构统计。

### 19.2 推荐的主结果表（不可预填实验数字）

| Dataset / Split | Candidate Recall | Classical AP | Quantum AP | Heat/Wave AP | Exact Edge-F1 | Jaccard | Runtime | Peak RAM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Synthetic S1 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Synthetic S2 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Enron | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Primary school | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| High school | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

**TBD 的意思是尚未运行。禁止由代码模型根据论文或预期填入任何虚构实验结果。**

### 19.3 可能的论文贡献（须通过 Gate 后才能确认）

- **Candidate contribution 1**：提出群组条件化的相干/非相干量子动态响应，完整分析其性质、退化与经典局部结构关联。
- **Candidate contribution 2**：以候选超边判别为桥梁，将量子结构特征引入具有统计监督/复杂度约束的超图重建，而非直接将 Born 概率等同成员概率。
- **Candidate contribution 3**：建立统一候选、统一经典强基线与真实超边评价协议，并明确展示何时有效、何时无效。
- **Candidate contribution 4（选做）**：在无真实超边设置下构建紧凑、可解释、对下游 HGNN 有用的潜在超图。

应根据真实理论与实验成果删减贡献，不能先承诺四项一定成立。

### 19.4 论文框架建议

1. Introduction：潜在高阶关系重建、静态/动态构建与恢复限制；提出待检验的量子相干结构假设。
2. Related Work：超图构建与动态优化、真实超图重建、量子游走图表征。
3. Problem Definition：A/B 两目标、输入观测、候选/真实超边区别、不可辨识性。
4. Quantum Group Dynamics：Hamiltonian、相干与非相干群组响应、数学性质与短时间展开。
5. Quantum-Guided Hyperedge Inference：候选生成、结构监督、复杂度选择、软关联（若确有收益）。
6. Experimental Protocol：候选覆盖、真实事件重建、强经典基线、下游 B、运行资源。
7. Results and Ablations：按科学假设呈现，有效与失败的条件。
8. Limitations：信息论不可辨识、单粒子系统非不可约多体物理、计算成本、数据语义。
9. Conclusion：仅陈述经实验/理论支持的发现。

---

## 20. 供代码模型执行的任务说明（可直接复制）

> **角色**：你是一名熟悉 SciPy 稀疏线性代数、PyTorch、图/超图学习与可复现实验的研究工程师。请严格按照本研究规范实现，不得自行改变数学定义或偷用测试标签。  
> **第一批交付仅实现 P0 + P1**，不要一开始开发完整端到端结构学习和 HGNN。  
> **每完成一个模块必须同时交付 pytest 单元测试、简单 README、输入/输出类型、时间与内存记录**。  
> **禁止**伪造实验结果、凭空引用数据集、默默改变候选预算、在各基线使用不同负样本、将量子概率直接命名为超边真值概率。
>
> 第一步：实现二值/加权投影，构建三角形不可辨识反例、合成超图生成器，并跑通过测试。  
> 第二步：实现归一化 Laplacian Hamiltonian、reference dense expm、production sparse expm_multiply。  
> 第三步：实现 $R_{\rm coh},R_{\rm inc},C$、随机相位估计、多时间响应、式 (1)-(4) 的校验及退化测试。  
> 第四步：实现共享候选生成器、困难负例、CandidateRecall/Recall@Budget 和传统强基线 $z_C$。  
> 第五步：分别提取 $z_C,z_Q$ 并用相同 LogisticRegression 进行配对候选判别；先输出 synthetic S1/S2/S3 的结果，**不提前启动目标 B**。  
> 第六步：复核结果后，按模块化结构扩展 Enron 与学校接触超图的真实重建。  
> 第七步：只有 Gate 1/2 通过，再实现超边选择优化和 HGNN 下游 B。

**每次迭代的提交材料**：源代码、单元测试通过日志、已解析配置、环境依赖锁定、数据 manifest、方法间公平性检查报告、真实结果 CSV/JSON、可重建的随机种子列表及失败案例。遇到理论定义不清或外部数据不匹配时必须在 README 中提出假设/阻塞，不允许悄悄修改问题定义。

### 20.1 第一次对代码模型的验收标准

- 可以运行 `python scripts/00_verify_theory.py`，通过群组响应全部数学单元测试。
- 可以生成三角形两种超图同投影的不可辨识性示例，并明确输出“不能仅从 $A$ 区分”。
- 可以生成小合成数据与真实候选、统计候选覆盖率。
- 能用相同候选和划分比较 C0/C4/Q1/Q2/Q3，输出 AP/AUROC、分层统计及运行时间。
- 如没有量子增益，输出负面或不显著结论，而不是自动增加复杂结构来追求更高数值。

---

## 21. 已核对的主要文献与原始数据链接

> 以下尽量采用会议 proceedings、出版社期刊页面、大学数据发布页。网页更新及后续勘误以链接内容为准。本文件的公式 (1)–(6) 是**本研究的拟议定义/推导**，不要把它们误写成下列论文的原始公式。

### 21.1 超图构建、动态结构学习与真实重建

1. **Zhou, Huang & Schölkopf (2006)**. *Learning with Hypergraphs: Clustering, Classification, and Embedding*. NeurIPS. 官方：https://papers.nips.cc/paper_files/paper/2006/hash/dff8e9c2ac33381546d96deea9922999-Abstract.html
2. **Feng et al. (2019)**. *Hypergraph Neural Networks*. AAAI. 官方：https://ojs.aaai.org/index.php/AAAI/article/view/4235
3. **Jiang et al. (2019)**. *Dynamic Hypergraph Neural Networks*. IJCAI. 官方：https://www.ijcai.org/proceedings/2019/366
4. **Cai et al. (2022)**. *Hypergraph Structure Learning for Hypergraph Neural Networks*. IJCAI. 官方：https://www.ijcai.org/proceedings/2022/0267
5. **Zhou et al. (2023)**. *Totally Dynamic Hypergraph Neural Networks*. IJCAI. 官方：https://ijcai.org/proceedings/2023/275
6. **Young, Petri & Peixoto (2021)**. *Hypergraph reconstruction from network data*. Communications Physics 4:135. 官方：https://www.nature.com/articles/s42005-021-00637-w
7. **Wang & Kleinberg (2024)**. *From Graphs to Hypergraphs: Hypergraph Projection and its Reconstruction*. ICLR. 官方：https://proceedings.iclr.cc/paper_files/paper/2024/hash/e92b8441b702d142435f243806f9ec5a-Abstract-Conference.html
8. **Bresler, Guo & Polyanskiy (2024)**. *Thresholds for Reconstruction of Random Hypergraphs From Graph Projections*. COLT. 官方：https://proceedings.mlr.press/v247/bresler24a.html
9. **Bresler, Guo, Polyanskiy & Yao (2025)**. *Partial and Exact Recovery of a Random Hypergraph from its Graph Projection*. COLT. 官方：https://proceedings.mlr.press/v291/bresler25a.html
10. **Lee, Lee & Shin (2025)**. *MARIOH: Multiplicity-Aware Hypergraph Reconstruction*. ICDE. 官方 DOI：https://doi.org/10.1109/ICDE65448.2025.00233
11. **Duta & Liò (2025)**. *SPHINX: Structural Prediction using Hypergraph Inference Network*. ICML. 官方：https://proceedings.mlr.press/v267/duta25a.html
12. **Sim et al. (2026)**. *Learning Multi-Scale Hypergraph for High-Order Brain Connectivity Analysis*. ICML. 官方：https://proceedings.mlr.press/v306/sim26b.html
13. **Wen & Yu (2025)**. *HyperPLR: Hypergraph Generation through Projection, Learning, and Reconstruction*. ICLR. 官方：https://proceedings.iclr.cc/paper_files/paper/2025/hash/3653128ac0be4c436420fbdb03142c48-Abstract-Conference.html
14. **Chien et al. (2022)**. *You Are AllSet: A Multiset Function Framework for Hypergraph Neural Networks*. ICLR. 论文入口：https://arxiv.org/abs/2106.13264

### 21.2 量子游走与数学背景

15. **Godsil (2013)**. *Average mixing of continuous quantum walks*. Journal of Combinatorial Theory, Series A. 官方 DOI：https://doi.org/10.1016/j.jcta.2013.05.006
16. **Coutinho, Godsil, Guo & Zhan (2018)**. *A New Perspective on the Average Mixing Matrix*. Electronic Journal of Combinatorics. 官方：https://www.combinatorics.org/ojs/index.php/eljc/article/view/v25i4p14
17. **Discrete-time quantum walk on complex networks for community detection (2020)**. Physical Review Research. 官方：https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.2.023378
18. **Emms, Wilson & Hancock (2009)**. *Graph matching using the interference of discrete-time quantum walks*. Image and Vision Computing. 官方：https://www.sciencedirect.com/science/article/pii/S0262885608002369

> **校正说明**：量子干涉用于图匹配的 2009 年论文是 **Image and Vision Computing** 上的 *discrete-time quantum walks* 工作，**不是**此前口头讨论中可能误述的 “Pattern Recognition 上的 continuous-time” 论文。本文件使用已核对的出版信息。

### 21.3 真实超图数据来源

19. **Austin R. Benson, Higher-order Network Datasets**, Cornell 主页：https://www.cs.cornell.edu/~arb/data/
20. **email-Enron temporal hypergraph**：https://www.cs.cornell.edu/~arb/data/email-Enron/
21. **coauth-DBLP temporal hypergraph**：https://www.cs.cornell.edu/~arb/data/coauth-DBLP/
22. **contact-primary-school labeled**：https://www.cs.cornell.edu/~arb/data/contact-primary-school-labeled/
23. **contact-high-school labeled**：https://www.cs.cornell.edu/~arb/data/contact-high-school-labeled/
24. **contact-high-school temporal**：https://www.cs.cornell.edu/~arb/data/contact-high-school/
25. **PyTorch Geometric Planetoid**：https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.Planetoid.html

---

## 22. 最终研究决策：先回答科学问题，再扩大模型

本项目的第一原则不是“必须设计一种量子超图生成模型”，而是：

> **在明确的观测机制、候选搜索预算、超边真实标签与经典强基线下，量子相干动力学能否为超边识别提供稳定且可解释的增量结构信息？**

若答案为“在某些受控条件下是”，就针对这些条件研究机制、建立理论性质，并向目标 A 的结构重建、目标 B 的任务学习扩展；若答案为“否”或“不足以抵偿成本”，必须如实停止或重新定义问题。

**最小闭环**：真实超图 $\rightarrow$ 二值投影 $A$ $\rightarrow$ 同一候选池 $\mathcal C$ $\rightarrow$ 经典与量子群组证据 $\rightarrow$ 相同评分器 $\rightarrow$ 真实超边精确恢复对照。  
**扩展闭环**：普通属性图 $(A,X)$ $\rightarrow$ 验证过的量子引导结构选择 $\rightarrow$ 潜在超图 $B$ $\rightarrow$ 固定 HGNN $\rightarrow$ 节点任务评价。

只有两条闭环分别给出可信证据，才能将该方向发展为完整的 *Quantum-Inspired Hypergraph Structure Learning and Reconstruction* 研究。
