# 医药渠道时序 Foundation MoE 专家模型清单

## 1. 适用范围

本清单面向公司内部医药渠道时序 Foundation MoE，基于当前可获得的数据字段：

- 日期
- 医院、医院派生维度
- SKU 多级分类、SKUID、SKU 派生维度
- 销量、金额、单价

覆盖当前三类业务预测场景：

- **30d**：月内不同截止点预测当月最终销量，而非预测未来 30 天。
- **1m**：目标月医院是否进货及医院进货量预测。
- **n6m**：未来 6 个月产品、规格或 SKU 销量预测。

适配等级定义：

- **P0 / 直接适配**：当前字段即可构建，建议进入首版模型及论文主实验。
- **P1 / 直接适配**：当前字段可构建，但建议在 P0 稳定后加入。
- **P2 / 条件适配**：需要医院派生维度质量、足够历史长度、价格变化次数或历史预测日志满足条件。

“专家”在本文中包括可被 Router 激活的预测专家，以及与专家协同的输出头、校准模块和一致性模块。Token、Embedding、Router、负载均衡和微调方式属于架构机制，不作为独立业务专家重复列入。

## 2. 时间模式与多尺度专家

| ID | 专家模型 | 核心解决问题 | 当前可用输入 | 适用场景 | 适配等级 | 典型实现/来源 |
|---|---|---|---|---|---|---|
| T01 | 长期趋势专家 Long-term Trend | 捕获长期增长、下降和缓慢漂移 | 月度销量、金额、活跃医院数、SKU 历史 | n6m | P0 | MoLE、FreqMoE、Long Linear |
| T02 | 局部趋势专家 Local Trend | 捕获近期窗口斜率、加速和减速 | 日/月销量、金额、滞后和滚动统计 | 30d、1m、n6m | P0 | Short Linear、MoLE、Moirai-MoE |
| T03 | 趋势反转专家 Trend Reversal | 识别增长转下降或下降转增长 | 销量斜率、动量、变化点统计 | 30d、n6m | P1 | Pattern-specific MoE、Change-point Expert |
| T04 | 固定季节专家 Fixed Seasonality | 建模星期、月份、季度、年度稳定周期 | 日期、销量、金额 | 30d、1m、n6m | P0 | FreqMoE、Fourier/Seasonal Linear |
| T05 | 动态季节专家 Dynamic Seasonality | 处理季节强度和相位随时间变化 | 日期、滚动销量、SKU/医院 Embedding | 30d、n6m | P1 | Moirai-MoE、Dynamic Fourier Expert |
| T06 | 单周期专家 Single Periodicity | 提取占主导地位的周期 | 销量序列、日期频率特征 | 30d、n6m | P1 | FreqMoE、FFT Expert |
| T07 | 多周期专家 Multiple Periodicity | 同时建模周、月、季、年等重叠周期 | 日期、日/月销量 | 30d、n6m | P1 | FreqMoE、WaveMoE |
| T08 | 短期动态专家 Short-term Dynamics | 捕获近期惯性和短滞后依赖 | 近期销量、金额、单价 | 30d、1m | P0 | MoLE、CNN、Moirai-MoE |
| T09 | 月内节奏专家 In-month Pace | 根据月内位置和 MTD 进度预测最终月销量 | 日期、工作日序号、MTD 销量、MTD 医院数 | 30d | P0 | Pace Curve、TFT-style Covariate Expert |
| T10 | 突变专家 Abrupt Change | 捕获跳变、断崖、快速放量和局部冲击 | 销量、金额、单价变化 | 30d、1m、n6m | P1 | Dynamic TMoE、Change-point Expert |
| T11 | Motif 专家 | 识别历史中重复出现的局部子序列 | 日/月销量 Patch | 30d、n6m | P2 | Pattern-specific MoE、Matrix Profile Expert |
| T12 | 形态专家 Shape Pattern | 区分峰、谷、平台、阶跃和脉冲 | 标准化销量 Patch | 30d、n6m | P1 | CNN Expert、Pattern-specific MoE |
| T13 | 可预测残差专家 Predictable Residual | 学习主专家未解释的规律性残差 | 基础预测、历史实际、残差滞后 | 全部 | P1 | TIMERM、Residual MoE |
| T14 | 非线性残差专家 Nonlinear Residual | 捕获趋势和周期之外的非线性结构 | 基础预测、协变量、历史残差 | 全部 | P1 | MLP Residual Expert、TIMERM |
| T15 | 误差修正专家 Error Correction | 修正医院、SKU、月份和视距上的系统性误差 | 历史预测日志、实际销量 | 全部 | P2 | Residual MoE、Calibration MoE |
| T16 | 细粒度尺度专家 Fine Scale | 建模日度和短窗口微观变化 | 日度医院×SKU 销量 | 30d、1m | P0 | ScaleMoR、Short Patch Expert |
| T17 | 中尺度专家 Medium Scale | 建模周度至月度业务动态 | 周/月聚合销量、金额 | 30d、1m、n6m | P0 | ScaleMoR、FreqMoE |
| T18 | 粗粒度尺度专家 Coarse Scale | 建模季度、年度低频趋势 | 月/季聚合销量、医院覆盖 | n6m | P0 | Long Patch、Low-frequency Expert |
| T19 | 跨尺度交互专家 Cross-scale Interaction | 学习短期变化对长期趋势的影响 | 日、周、月多尺度 Patch | 30d、n6m | P1 | Multi-scale MoE、Cross-scale Attention |
| T20 | 短 Patch 专家 Short Patch | 提取近期局部形态 | 短窗口销量 Patch | 30d、1m | P0 | Moirai-MoE、Time-MoE |
| T21 | 长 Patch 专家 Long Patch | 提取跨月、跨季度长期结构 | 长窗口销量 Patch | n6m | P0 | Moirai-MoE、Time-MoE |
| T22 | 自适应 Patch 专家 Adaptive Patch | 根据稀疏度和周期动态选择窗口 | 销量序列、非零率、周期统计 | 全部 | P1 | Multi-patch MoE、Moirai-MoE 类架构 |
| T23 | 低频专家 Low Frequency | 捕获趋势、年度周期和慢变化 | 月度销量频谱 | n6m | P0 | FreqMoE、Fourier Expert |
| T24 | 中频专家 Mid Frequency | 捕获月度和季度业务周期 | 日/月销量频谱 | 30d、n6m | P1 | FreqMoE |
| T25 | 高频专家 High Frequency | 捕获日度快速变化和高频扰动 | 日度销量、金额 | 30d | P1 | FreqMoE、Wavelet Expert |
| T26 | 自适应频带专家 Adaptive Frequency | 按样本动态组合有效频带 | 销量频谱、序列长度、采样频率 | 30d、n6m | P1 | FreqMoE |
| T27 | 时频融合专家 Time-frequency Fusion | 同时保留局部发生时间和频率模式 | 日/月销量、Patch 频谱 | 30d、n6m | P1 | WaveMoE、Dual-domain MoE |

## 3. 依赖关系、实体与层级专家

| ID | 专家模型 | 核心解决问题 | 当前可用输入 | 适用场景 | 适配等级 | 典型实现/来源 |
|---|---|---|---|---|---|---|
| D01 | 短程依赖专家 Short-range | 建模相邻日期和短滞后关系 | lag-1、lag-7、短窗口销量 | 30d、1m | P0 | CNN、Short Linear、Time-MoE |
| D02 | 长程依赖专家 Long-range | 建模远距离滞后和长期记忆 | 长历史销量、金额、单价 | 1m、n6m | P0 | Attention、State Space、Moirai-MoE |
| D03 | 季节滞后专家 Seasonal Lag | 建模 lag-7、lag-12 月及去年同期 | 日期、日/月销量 | 全部 | P0 | FreqMoE、Period-specific Linear |
| D04 | 单序列专家 Intra-channel | 独立学习每个医院×SKU 序列自身动态 | 医院、SKUID、销量历史 | 1m、n6m | P0 | Channel-independent Expert |
| D05 | 跨变量专家 Cross-channel | 学习销量、金额、单价及派生特征交互 | 销量、金额、单价、派生维度 | 全部 | P0 | Dual-MoE、Cross-variable Attention |
| D06 | 动态变量选择专家 Channel Selection | 按样本选择当前最有效的变量 | 全部数值和派生字段 | 全部 | P1 | Variable Selection Network、Channel MoE |
| D07 | 医院实体专家 Hospital Entity | 学习医院个体、类型、等级、区域差异 | 医院 ID、医院派生维度 | 1m | P0 | Entity-aware MoE、Entity Embedding Expert |
| D08 | 医院群组专家 Hospital Cluster | 为统计行为相似医院共享专家 | 医院派生维度、RFM、销量模式 | 1m | P0 | Clustered MoE |
| D09 | 医院生命周期专家 Hospital Lifecycle | 区分新增、活跃、沉默、流失和恢复医院 | 首次/最近交易、频次、销量 | 1m、n6m | P0 | Lifecycle/Regime Expert |
| D10 | SKU 身份专家 SKU Identity | 学习单个 SKU 的特定需求模式 | SKUID、SKU 派生维度 | 1m、n6m | P0 | Entity-aware MoE |
| D11 | SKU 层级专家 SKU Hierarchy | 在 SKU 与多级分类间迁移和共享信息 | SKU 多级分类、SKUID | 1m、n6m | P0 | Hierarchical MoE、Grouped-series Expert |
| D12 | SKU 生命周期专家 SKU Lifecycle | 区分新品、成长期、成熟期和衰退期 | SKU 上市历史、销量趋势、价格变化 | 1m、n6m | P1 | Lifecycle/Regime Expert |
| D13 | 医院×SKU 交互专家 Entity Interaction | 建模医院属性与产品属性的匹配关系 | 医院维度、SKU 维度、历史交易 | 1m | P0 | Factorization/Interaction Expert |
| D14 | 冷启动医院专家 Hospital Cold-start | 仅依赖医院派生维度为低历史医院迁移知识 | 医院派生维度、同群医院历史 | 1m | P1 | Meta/Cluster Expert |
| D15 | 冷启动 SKU 专家 SKU Cold-start | 依赖 SKU 分类和派生属性迁移同类产品知识 | SKU 分类、SKU 派生维度 | 1m、n6m | P1 | Hierarchical Transfer Expert |
| D16 | 局部关系专家 Local Relation | 建模同区域、同等级医院或同品类 SKU 的局部关系 | 医院/SKU 派生维度 | 1m、n6m | P2 | Spatiotemporal MoE、Graph Expert |
| D17 | 全局关系专家 Global Relation | 建模跨区域、跨品类的全局共性 | 医院/SKU 层级、全局聚合 | n6m | P2 | Global Graph/Attention Expert |
| D18 | 序列群组专家 Series Cluster | 按稀疏度、趋势、季节和波动性分配专家 | 序列统计特征 | 全部 | P0 | Clustered MoE、Pattern Router |

## 4. 统计分布、需求状态与数据质量专家

| ID | 专家模型 | 核心解决问题 | 当前可用输入 | 适用场景 | 适配等级 | 典型实现/来源 |
|---|---|---|---|---|---|---|
| S01 | 平稳状态专家 Stationary | 处理均值、方差和相关结构相对稳定的序列 | 稳定性统计、销量历史 | 全部 | P1 | Statistical MoE |
| S02 | 非平稳状态专家 Non-stationary | 处理均值和分布随时间变化 | 趋势、漂移、滚动统计 | 全部 | P0 | Moirai-MoE、Dynamic TMoE |
| S03 | 局部平稳专家 Local Stationarity | 将全局非平稳序列拆为局部稳定区间 | 变化点、窗口统计 | 30d、n6m | P1 | Dynamic TMoE |
| S04 | 偏态分布专家 Skewed Distribution | 处理右偏销量和金额 | 销量、金额、对数变换 | 全部 | P0 | Distributional MoE、Log-normal/Tweedie Head |
| S05 | 厚尾分布专家 Heavy-tailed | 处理偶发大单和极端值 | 销量、金额、异常统计 | 全部 | P1 | Student-t Expert |
| S06 | 异方差专家 Heteroscedasticity | 处理销量水平越高、误差方差越大的情况 | 历史均值、波动率、销量 | 全部 | P1 | Probabilistic MoE |
| S07 | 低波动专家 Low Variance | 针对稳定医院、SKU 和时期使用低复杂度预测 | 波动率、稳定性统计 | 全部 | P1 | Regime MoE、Linear Expert |
| S08 | 高波动专家 High Variance | 针对剧烈波动序列输出更稳健预测和区间 | 波动率、异常率、销量 | 全部 | P1 | Regime MoE、Robust/Quantile Expert |
| S09 | 噪声鲁棒专家 Noisy Signal | 抑制测量误差、异常大单和随机扰动 | 销量、金额、价格、鲁棒统计 | 全部 | P0 | Robust MoE、Huber/Median Expert |
| S10 | 缺失模式专家 Missing Pattern | 区分缺记录、不完整观测和真实零销量 | 数据到达标记、空值模式、交易记录 | 全部 | P0 | Robust MoE、Missingness-aware Expert |
| S11 | 上升状态专家 Rising Regime | 针对持续增长状态学习专门规律 | 趋势、动量、活跃医院变化 | 全部 | P1 | Regime MoE |
| S12 | 下降状态专家 Falling Regime | 针对持续下降状态学习专门规律 | 趋势、动量、医院流失 | 全部 | P1 | Regime MoE |
| S13 | 稳定状态专家 Sideways Regime | 针对平稳或横盘状态使用简单稳定模型 | 趋势和波动率 | 全部 | P1 | Regime MoE、ETS/Linear Expert |
| S14 | 正常业务状态专家 Normal Regime | 学习无明显异常时期的基础生成机制 | 全部历史字段 | 全部 | P0 | Dynamic TMoE |
| S15 | 冲击状态专家 Shock Regime | 处理销量或价格突然跳变 | 销量、金额、单价变化 | 全部 | P1 | Dynamic TMoE、Change-point Expert |
| S16 | 恢复状态专家 Recovery Regime | 建模冲击后恢复速度和新水平 | 冲击标记、冲击后销量路径 | 30d、n6m | P1 | Dynamic TMoE |
| S17 | 渐进漂移专家 Gradual Drift | 处理需求关系缓慢变化 | 滚动分布、趋势、医院/SKU 结构变化 | 全部 | P1 | Drift-aware MoE |
| S18 | 突发漂移专家 Sudden Drift | 处理生成机制突然改变 | 变化点、销量和价格跳变 | 全部 | P1 | Dynamic TMoE |
| S19 | 重现漂移专家 Recurring Drift | 识别曾出现并再次返回的需求状态 | 历史状态 Embedding、季节特征 | 30d、n6m | P2 | Dynamic MoE |
| S20 | 间歇需求发生专家 Demand Occurrence | 预测医院×SKU 在目标期是否发生进货 | 医院、SKU、Recency、Frequency、历史零/非零 | 1m | P0 | Bernoulli/Hurdle Expert、Task MoE |
| S21 | 条件进货量专家 Conditional Quantity | 在发生进货条件下预测正销量 | 正销量历史、医院/SKU 特征 | 1m | P0 | Gamma、Log-normal、Quantile Expert |
| S22 | 过度离散计数专家 Negative Binomial | 处理方差显著大于均值的非负销量 | 非负销量、医院/SKU 特征 | 1m、n6m | P0 | Negative Binomial Head |
| S23 | 零膨胀专家 Zero-inflated/Hurdle | 联合处理大量零值与正销量分布 | 零/非零标签、销量、实体特征 | 1m | P0 | ZINB、Hurdle Distribution Expert |
| S24 | 多峰分布专家 Mixture Distribution | 处理正常补货、大单和异常补货形成的多峰分布 | 销量、医院/SKU 状态 | 1m、n6m | P2 | Mixture Density/Probabilistic MoE |

## 5. 业务协变量与领域专家

| ID | 专家模型 | 核心解决问题 | 当前可用输入 | 适用场景 | 适配等级 | 典型实现/来源 |
|---|---|---|---|---|---|---|
| B01 | 星期/工作日专家 Weekday | 建模工作日和周内结构 | 日期、工作日派生特征 | 30d | P0 | Calendar Covariate Expert |
| B02 | 节假日专家 Holiday | 建模节假日及节前节后差异 | 日期派生的法定节假日 | 30d | P0 | Holiday Covariate Expert |
| B03 | 月内位置专家 Month Position | 建模月初、月中、月末效应 | 日期、自然日/工作日序号 | 30d | P0 | TFT-style Covariate Expert |
| B04 | 月份/季度专家 Calendar Season | 建模月份、季度及年内位置 | 日期 | 1m、n6m | P0 | Temporal Embedding Expert |
| B05 | 价格关联专家 Price Association | 学习单价区间变化与销量变化的关联 | 单价、销量、金额、SKU | 1m、n6m | P1 | Covariate MoE、Price-response Expert |
| B06 | 价格状态专家 Price Regime | 区分稳定价、降价、涨价及新价格平台 | 单价变化点、价格区间 | n6m | P1 | Regime MoE |
| B07 | 目标—协变量交互专家 | 学习历史销量与医院/SKU/价格特征交互 | 全部字段 | 全部 | P0 | Covariate MoE |
| B08 | 多协变量交互专家 | 学习医院、SKU、价格、日历的联合影响 | 全部字段 | 全部 | P1 | Cross-attention/MLP Expert |
| B09 | 动态特征选择专家 | 按实体和状态选择有效派生特征 | 全部原始和派生字段 | 全部 | P1 | Feature-routing MoE |
| B10 | 医药渠道需求专家 | 学习医药渠道补货、医院覆盖和 SKU 生命周期模式 | 医院、SKU、销量、价格 | 全部 | P0 | Domain Expert、Retail/Channel MoE |
| B11 | 数据集专家 Dataset Specialization | 为不同客户、品牌或数据源学习隐式偏好 | 数据集/客户标识（若存在） | 全部 | P2 | Dataset-aware MoE |
| B12 | 金额一致性模块 Revenue Identity | 保证金额与销量、单价关系一致 | 销量、单价、金额 | 全部 | P0 | `amount ≈ quantity × price` 约束模块 |
| B13 | 医院层级协调模块 | 保证医院、医院群组、区域和全国汇总一致 | 医院派生层级、销量 | 1m、n6m | P1 | Hierarchical Reconciliation、MECATS |
| B14 | SKU 层级协调模块 | 保证 SKUID 与多级品类汇总一致 | SKU 多级分类、SKUID、销量 | 1m、n6m | P1 | Hierarchical Reconciliation、MinT 类方法 |

> 价格专家当前只能解释相关关系，不能仅凭销量、金额和单价宣称因果价格弹性。若未来加入促销、集采、准入、库存、断货和销售动作字段，才能进一步进行可靠的价格因果或事件增量建模。

## 6. 异构模型族专家

| ID | 专家模型 | 核心解决问题 | 适用场景 | 适配等级 | 典型实现 |
|---|---|---|---|---|---|
| M01 | 自回归专家 Autoregressive | 学习线性短滞后关系 | 全部 | P0 | AR、ARIMA、AutoReg |
| M02 | 指数平滑专家 Exponential Smoothing | 学习水平、趋势和季节平滑 | 30d、n6m | P0 | ETS、Holt-Winters |
| M03 | 间歇需求统计专家 | 提供稀疏需求强基线和可解释预测 | 1m | P0 | Croston、SBA、TSB |
| M04 | 短窗口线性专家 Short Linear | 低成本学习近期线性映射 | 30d、1m | P0 | DLinear/RLinear Short Expert |
| M05 | 长窗口线性专家 Long Linear | 学习长期趋势和季节滞后 | n6m | P0 | MoLE、Long Linear |
| M06 | 周期线性专家 Period-specific Linear | 为不同月份或周期学习不同线性规则 | 30d、n6m | P1 | MoLE、Seasonal Linear |
| M07 | MLP 专家 | 学习固定窗口上的非线性交互 | 全部 | P0 | Feed-forward Expert |
| M08 | 卷积专家 Convolution | 提取局部形态、突变和多尺度模式 | 30d、1m | P1 | 1D CNN、TCN Expert |
| M09 | 循环专家 Recurrent | 学习递归状态和序列记忆 | 30d、1m | P2 | GRU、LSTM Expert |
| M10 | 注意力专家 Attention | 学习长距离及动态依赖 | 1m、n6m | P0 | Transformer Expert |
| M11 | 状态空间专家 State Space | 高效学习长序列动态 | 30d、n6m | P1 | SSM、Mamba-style Expert |
| M12 | 频域专家 Fourier | 直接建模趋势和周期频带 | 30d、n6m | P1 | FFT/Fourier Expert、FreqMoE |
| M13 | 小波专家 Wavelet | 定位局部冲击发生的时间与尺度 | 30d、n6m | P2 | WaveMoE、Wavelet Expert |
| M14 | 统计—神经混合专家 | 融合统计归纳偏置与神经网络能力 | 全部 | P1 | Hybrid MoE |
| M15 | 线性—非线性混合专家 | 按样本决定采用简单或复杂规律 | 全部 | P0 | MoLE、Heterogeneous MoE |
| M16 | 树模型专家 Tree-based | 处理医院/SKU 高维类别和派生特征 | 1m、n6m | P0 | LightGBM、CatBoost Expert |

## 7. 预测视距、任务与概率输出专家

| ID | 专家模型/输出头 | 核心解决问题 | 适用场景 | 适配等级 | 实现建议 |
|---|---|---|---|---|---|
| H01 | 短视距专家 Short Horizon | 优化临近未来和月内局部预测 | 30d、1m | P0 | Horizon Token + Short Expert |
| H02 | 中视距专家 Medium Horizon | 平衡近期动态和季节模式 | n6m 的 M+1～M+3 | P0 | Horizon-aware MoE |
| H03 | 长视距专家 Long Horizon | 强化趋势、季节和长期依赖 | n6m 的 M+4～M+6 | P0 | Freq/Long Trend Expert |
| H04 | 近端步长专家 Near-step | 专门优化预测窗口前部 | n6m 的 M+1～M+2 | P1 | Horizon-level Routing |
| H05 | 远端步长专家 Far-step | 控制远期误差累积 | n6m 的 M+5～M+6 | P1 | Horizon-level Routing |
| H06 | 直接多步专家 Direct Forecast | 一次输出完整预测窗口，避免递归误差累积 | n6m | P0 | Direct Multi-output Head |
| H07 | 递归预测专家 Recursive Forecast | 用前一步预测生成后一步，作为对照模型 | n6m | P2 | Autoregressive Decoder |
| H08 | 多视距专家 Multi-horizon | 为不同预测距离学习不同权重 | n6m | P0 | Horizon-aware MoE |
| O01 | 进货分类头 Classification | 预测医院×SKU 是否进货 | 1m | P0 | Bernoulli Head、BCE/Focal Loss |
| O02 | 条件均值头 Mean Forecast | 输出条件期望销量 | 全部 | P0 | Regression Head |
| O03 | 条件中位数头 Median Forecast | 对偏态和异常值提供稳健点预测 | 全部 | P1 | P50 Quantile Head |
| O04 | 分位数专家 Quantile | 输出 P10/P50/P90 等业务风险区间 | 全部 | P0 | Pinball Loss、Quantile MoE |
| O05 | Student-t 分布头 | 建模厚尾预测误差 | 30d、n6m | P1 | Distributional Head |
| O06 | Negative Binomial 分布头 | 建模过度离散的非负销量 | 1m、n6m | P0 | NB/NB2 Head |
| O07 | 混合分布头 Mixture Distribution | 建模多峰和复杂条件分布 | 1m、n6m | P2 | Mixture Density Head |
| O08 | 异常检测头 Anomaly Detection | 识别偏离正常预测分布的销量 | 全部 | P1 | Residual/Probabilistic Anomaly Head |

## 8. 聚合、校准与不确定性专家

| ID | 专家模型/模块 | 核心解决问题 | 前置数据 | 适用场景 | 适配等级 | 典型实现 |
|---|---|---|---|---|---|---|
| C01 | 加权聚合专家 Weighted Sum | 按 Router 权重融合专家输出 | 各专家输出 | 全部 | P0 | Soft/Sparse MoE Aggregator |
| C02 | 注意力融合专家 Attention Fusion | 根据上下文动态融合专家表示 | 专家表示、实体和视距上下文 | 全部 | P1 | Dual-MoE Attention Fusion |
| C03 | Stacking 专家 | 用二级模型组合异构专家预测 | OOF 专家预测 | 全部 | P1 | Meta Learner、Stacking |
| C04 | 残差融合专家 Residual Fusion | 让后续专家逐级修正剩余误差 | 基础预测、残差 | 全部 | P1 | TIMERM、Residual MoE |
| C05 | 偏差校准专家 Bias Correction | 修正医院、SKU、月份和视距系统偏差 | 历史预测与实际 | 全部 | P1 | Calibration MoE |
| C06 | 方差校准专家 Variance Calibration | 修正预测区间过宽或过窄 | 概率预测、历史覆盖率 | 全部 | P2 | Probabilistic Calibration |
| C07 | 分位数校准专家 Quantile Calibration | 提高 P10/P50/P90 覆盖率可靠性 | 分位数预测、实际值 | 全部 | P1 | Conformal/Quantile Calibration |
| C08 | 数据不确定性专家 Aleatoric | 表达需求自身不可约的不确定性 | 分布参数、噪声特征 | 全部 | P1 | Distributional MoE |
| C09 | 模型不确定性专家 Epistemic | 表达低样本医院/SKU上的模型认知不足 | 多模型/多专家预测 | 1m、n6m | P2 | Ensemble MoE |
| C10 | 路由不确定性专家 Routing Uncertainty | 表达 Router 无法确定专家时的风险 | Router logits/entropy | 全部 | P2 | Bayesian/Entropy-aware Router |

## 9. 首版推荐专家池

首版不建议把所有专家同时实现。建议先用 **8 个可路由核心专家 + 4 个输出/约束模块** 建立可验证基线：

| 类型 | 首版选择 |
|---|---|
| 可路由核心专家 | 月内节奏、局部趋势、长期趋势、季节频率、间歇需求发生、条件进货量、医院/SKU 层级、价格关联 |
| 输出与约束模块 | 多视距输出、分位数/NB 分布、层级协调、残差与偏差校准 |
| 推荐路由 | 时间模式 + 医院/SKU 层级 + 预测任务/视距联合路由，Top-2 激活 |
| 推荐粒度 | 日度医院×SKUID 为底层事实粒度，同时构建医院、SKU 分类和全国聚合视图 |

## 10. 当前不纳入的专家

| 专家方向 | 不纳入原因 | 需要补充的数据 |
|---|---|---|
| 促销增量专家 | 当前没有促销活动和投入字段 | 促销类型、开始结束时间、投入、覆盖对象 |
| 政策/集采事件专家 | 当前无法识别政策事件及影响窗口 | 政策、集采、医保、准入事件及生效日期 |
| 库存/断货专家 | 零销量无法区分无需求与无库存 | 库存、补货、断货、在途数据 |
| 销售行为专家 | 无法评估拜访和销售动作影响 | 拜访、覆盖频次、销售人员及动作数据 |
| 天气专家 | 当前业务场景和字段不支持 | 天气和医院所在地理映射 |
| 宏观经济专家 | 当前没有外部宏观指标 | 宏观指标及其发布时间版本 |
| 真实动态图专家 | 缺少医院之间的真实关系和传播边 | 医院网络、转诊、区域联系或业务关系 |
| 临床 Healthcare 专家 | 目标是渠道销量，不是生命体征或临床事件 | 临床事件、患者级时间序列 |

## 11. 建模前必须确认的数据契约

1. 明确“无交易记录”“真实零销量”“数据迟到”“医院/SKU 映射缺失”的区别。
2. 所有特征必须满足预测时点可获得，禁止使用目标期金额、销量或后验医院状态。
3. 单价若由金额除以销量得到，必须避免使用目标期单价造成标签泄漏。
4. 金额、销量、单价必须满足业务一致性；优先预测销量和价格情景，再计算金额。
5. 训练、验证和测试使用 rolling/expanding walk-forward，不使用随机切分。
6. 数据版本需记录时间跨度、医院数、SKU 数、医院×SKU 序列数、时间点数、非零率、价格变化次数和标签到达延迟。
7. 内部数据只能作为工业案例时，论文主结论还应在公开层级零售或间歇需求数据集上复现。

