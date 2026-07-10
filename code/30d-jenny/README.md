# 30d-jenny 预测任务

## 任务定义

基于日度销售数据预测每个业务月 (`bizym`) 的最终月销量 `month_total_qty`。

当前核心实验是 [03-experiment-20260701.ipynb](03-experiment-20260701.ipynb)：把“月中滚动预测”整理成直接月总量回归任务。模型在每个工作日开始前，只使用当时已经可见的信息，预测该月最终总销量。

## 数据

- 默认输入：`../../data/sales_daily.csv`
- 必需字段：`bizym`, `transdate`, `qty`, `num_hosp`
- `qty` 是日销量，`num_hosp` 是当天覆盖/发生销售的医院数
- 缺失日期会补齐；缺失 `qty` / `num_hosp` 按 0 处理；负数会截断为 0

## 预测时点

每个月按工作日序号生成多条预测样本：

- `forecast_workday_seq = 1`：第 1 个工作日 00:00，使用月初前/首个工作日前可见信息
- `forecast_workday_seq = k`：第 k 个工作日 00:00，使用截至第 k-1 个工作日结束后的 MTD 信息
- 预测越靠近月末，`mtd_qty`、`mtd_num_hosp` 等累计信息越完整

也就是说，任务不是预测某一天销量，而是在月内不同 cutoff 下滚动预测同一个月的最终销量。

## 标签与评估

- 预测目标：`actual_month_total = month_total_qty`
- 输出会被约束为非负，且不低于当前已发生的 `mtd_qty`
- 主要关注月总量误差：`month_total_mape_pct`, `WAPE`, `Bias`, `RMSE`, `MAE`
- 额外按 `forecast_workday_seq` 汇总 MAPE，用来观察“第几个工作日预测已足够可靠”

## 时间切分

`03-experiment-20260701.ipynb` 当前配置：

- 训练：`202201` 到 `202512`
- 测试：`202601` 到 `202612`
- 验证集：默认关闭 (`valid_ym_range = None`)

注意：如果真实数据只到某个月，测试样本也只会覆盖已有真实月总量的月份。

## 特征口径

特征只应使用预测时点已经可见的信息，避免泄漏整月结果。主要包括：

- 日历与工作日特征：自然日、工作日序号、剩余工作日、农历/春节相关特征
- 当前月 MTD：`mtd_qty`, `mtd_num_hosp`, 当日/累计医院数、工作日均值
- 历史滞后与滚动窗口：跨月 lag、rolling 均值/标准差/总和
- 同月历史：去年/前年/三年前同业务月的月总量和 MTD 形态
- 基于历史 MTD share / 工作日均值推导的 expected month total / remaining qty

月总量、月累计占比、未来日销量等事后才能知道的字段不能作为特征。
