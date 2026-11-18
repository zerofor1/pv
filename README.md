# 上海光储充数据与仿真

当前使用上海 MP-EVData 和 2022—2025 年天气，生成 ERA5 实况条件下的“假设真实负荷”，并计算参考/预报光伏。每站 35,064 小时，统一为北京时间整点小时。合成负荷不是四年实测记录。

## 目录

```text
data_raw/
  weather/gfs/           GFS 原生格点数据、预报批次索引
  weather/era5/          ERA5 原生变量组、来源记录
  load/MP_EVData/        原始交易和公开小时工作簿
data/
  weather/gfs/           四年处理天气、预报光伏
  weather/era5/          四年处理天气、参考光伏
  load/cleaned/          交易重构、缺失补全后的各站四年负荷
  load/simulated/        四年仿真负荷、逐日来源
model/
  preprocess/           天气下载、原生数据转换、负荷清洗对齐
  generate_load/        负荷生成脚本和配置
  pv/                   光伏物理模型和配置
  run.py                统一入口
test/
  weather/              天气检查
  load/                 负荷检查与回归测试
  pv/                   光伏检查与回归测试
  integration/          目录、迁移和流程检查
  fixtures/             合成测试样例
  reports/              全部检查报告、独立测试集合
项目材料/                申报书、计划书、方法说明、演示材料
  历史备份/              深圳数据、旧 GMM、购电优化及重复中间结果
.runtime/               隐藏的本地 Python 环境和依赖
```

所有保留文件的用途见 [文件索引](项目材料/文件索引.md)。同族站点文件和原生格点数据按通配符说明，避免重复列出上千个数据文件。旧文件、空目录和字节码缓存已清理；历史版本仅在一个备份目录内。

## 使用

在项目根目录执行，日常检查不会重新抽样：

```powershell
$python = '.\.runtime\python\python.exe'
& $python -m model.run --validate-only
& $python -m unittest discover -s test -t . -p 'test_*.py' -v
```

需要重新生成时：

```powershell
& $python -m model.run                  # 重新生成负荷、光伏并检查
& $python -m model.run --rebuild-load   # 从原始数据重新清洗，再生成并检查
& $python -m model.run --load-only      # 仅负荷
& $python -m model.run --pv-only        # 仅光伏
& $python -m model.preprocess.process --weather-only  # 从原生格点离线重建天气
```

新环境可安装 `.runtime/requirements.txt`；现有版本在 `.runtime/requirements-lock.txt`。下载脚本位于 `model/preprocess`，已有原生数据支持离线处理。重新下载 ERA5 才需要本机 CDS 授权，凭据不写入项目。

## 主要数据

- `data_raw/load/MP_EVData/*.xlsx`：原始工作簿，保持原样。
- `data_raw/weather/gfs/points/*.json`：保留原生变量和单位的 1,461 个预报格点片段；`run_manifest.csv` 记录起报、截止、可用时间、格点及对应文件。它们是数据集的点提取，不是全球 GRIB 全量文件。
- `data_raw/weather/era5/points/*.npz`：温度、风、气压降水、辐射四个原生变量组及源元数据。
- `data/weather/{gfs,era5}/hourly.csv`：经过单位转换、小时对齐及互补的四年天气，逐变量保留来源。
- `data/weather/{gfs,era5}/pv_hourly.csv`：同一参考阵列下的预报/参考光伏功率和损耗。
- `data/load/cleaned/A*_hourly.csv`：原交易覆盖内重构和同期均值补全值，附标准化值与来源标记。
- `data/load/simulated/all_stations_hourly_2022_2025.csv`：四年全站仿真负荷宽表，单位 kW。
- `data/load/simulated/A*_hourly_2022_2025.csv`：各站负荷、观测库尺度的 z 值及合成标记。
- `data/load/simulated/daily_provenance.csv`：每日电量来源、形状来源、缩放、片段、季节回退等信息。

## 方法与边界

从连接/断开时间及电量按实际小时重叠重构平均功率，保留超峰记录，不强制截断。缺失小时按同站同月、星期和小时的均值逐级回退补全。补全年份不进入模板库。

模板要求完整小时网格和全天交易证据覆盖；覆盖未确认的首日、不完整末日和范围外日期均排除，原始记录不改。A1 的 2025-01-01 跨年尾部不再作为完整模板。内部无交易时段仍按零处理，采集完整性和站点辅助用电尚未核实。A7/A8 的换电记录重构不等于已核实的电网侧补能功率。

各站统一标准化后使用“日电量＋配对形状＋少量近电量单日替换”。1、3、7 天片段公平比较；按日历和相近季节选样，缩放最多 ±10%，不加白噪声或裁峰。对最终缩放曲线的边界评分，保留历史跳变。ERA5 日均温和降雨条件只有在验证中改善至少 5% 才启用；GFS 不参与这一步的假设真实负荷生成。

连续验证和独立测试日期分离，训练统计和来源隔离。每站测试集合 100 条路径，报告电量、峰值、跳变、零值、CRPS 和覆盖率；独立成绩不再调参。最终四年场景用全部有效日期建库，不用于冒充独立预测成绩。A10 年度结果仅为探索，不代表已恢复其缺失季节。

配置分别在 `model/generate_load/config.json` 和 `model/pv/config.json`。光伏为上海城市参考点的 29.7 kWp/25 kW 示例阵列，不是各站实测设备或容量建议。当前主流程仅保留负荷和光伏；购电优化已归历史备份，未采用深圳电价，也未宣称真实购电成本改进。

## 检查报告

### 上海整日抽样与逐小时 GMM 对照

在项目根目录运行。默认每个站点的每个留出日各抽 100 条场景，可通过 `--draws`、`--seed`、`--output` 指定抽样数、随机种子和输出目录。

```powershell
& ./.runtime/python/python.exe -m simu_model.generate_load.daily_baseline
& ./.runtime/python/python.exe -m unittest test.load.test_daily_baseline -v
```

- 整日抽样在同站的训练日里，按北京时间周一至周五、周六至周日分组，有放回地复制 24 小时负荷。工作日按星期划分，法定假日和调休沿用星期标签。代码中的 `sample_whole_days` 也可以传入其他目标日期。
- 逐小时 GMM 沿用备份逐站模型的候选分量数和单变量边际结构。对每个站用训练日的原始 kW 重算标准化，再用训练集 BIC 选分量数。全天 24 小时独立抽样，负值置零，截零比例在报告中可查。
- 两组共用现有片段法的日期切分。训练、预留验证和测试互斥，候选模板、GMM 参数和评分尺度只依赖训练日。只读取覆盖审核通过、24 个整点均标为 `observed` 的交易重构功率。日内爬坡每条曲线计算 23 次，单位为 kW，时距为一小时。
- [comparison.json](test/reports/load/daily_baseline/comparison.json) 包含各站的日期清单、重拟合 GMM 参数、分项误差、分布统计和小时 P10/P50/P90。 [scenarios.npz](test/reports/load/daily_baseline/scenarios.npz) 含每站的真实留出曲线、两组生成场景及整日模板来源日期。生成数组维度为「抽样编号 × 留出日期 × 24 小时」，真实曲线维度为「留出日期 × 24 小时」。

**评分依据。** 先按站点和日类型比较，设尺度 `S = max(训练小时平均功率, 0.01 kW)`。每项占 25%。W₁ 是一阶 Wasserstein 距离，可理解为将两个分布对齐所需的平均移动量。

1. **日总电量**计算日总电量分布的 W₁，再除以 `24 × S × 1 h`，检查每日负荷规模和日间波动。
2. **日峰值**计算日最大功率分布的 W₁，再除以 `S`，检查场景峰值分布。
3. **小时爬坡**计算日内绝对小时功率差的 W₁ 和 P95 的绝对偏差，两者各占一半，再除以 `S`，同时检查常见跳变和较大跳变。
4. **典型日曲线**计算 24 个小时各自 P10、P50、P90 的平均绝对偏差，再除以 `S`，检查分时负荷水平、曲线形状和离散程度。

分位数使用经验分布的逆累积分布函数，同一组曲线整体等比例复制时分位数不变。站内按真实留出日数量汇总两类日的误差。跨站对归一化误差等权平均。综合误差 `L` 是四项的平均值，展示分为 `100 / (1 + L)`，100 表示四项误差均为零。该权重和分数映射是本次基线的约定，应用时可结合分项误差和原单位统计解读。

默认实验覆盖 266 个站点日，整日抽样与逐小时 GMM 分别得到 **84.76** 和 **62.71** 分。两者的小时爬坡归一化误差分别为 **0.092** 和 **0.994**，典型日曲线误差分别为 **0.185** 和 **0.825**。A8 和 A10 的整日抽样仍有较大的峰值分布偏差。A10 两个留出窗口均在第四季度，周末只有 4 个留出日。A7/A8 的结果沿用交易重构口径。评分范围为站级日内统计保真度。

### 现有流程报告

`test/reports/load/` 保存提取摘要、覆盖审核、生成评估、独立测试路径及来源、最终核验。`test/reports/weather/` 保存互补和天气核验。`test/reports/pv/` 保存输入输出哈希和损耗核验。`test/reports/integration/` 保存本次迁移清单与检查；`--migration-strict` 用于一次性比较迁移前快照，日常运行只检查当前结构。

报告和缓存可重新生成；迁移快照与历史报告仅作追溯，不是新参数下必须重复匹配的数值目标。
