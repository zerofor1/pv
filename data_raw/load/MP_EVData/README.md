# MP-EVData 原始输入

- `charging_session.xlsx`：全部站点原始交易及设备元数据。
- `station_hourly.xlsx`：公开小时表，用于覆盖和交叉核验。A7/A8 的会话数不能直接当作 kW。

来源：Runlong Liu 等，*An AI-Augmented Dataset of Multi-Prototype Electric Vehicle Charging Load Profiles in China*，数据 DOI：https://doi.org/10.6084/m9.figshare.29882366，源仓库标注 CC BY 4.0。

原始工作簿不改写。处理代码在 `model/preprocess/process.py`，清洗结果在 `data/load/cleaned`。覆盖审核与 A5/A6 交易核查代码在 `test/load`，报告在 `test/reports/load`。当前只将采集范围内的交易重构为小时平均功率，设备口径、辅助用电及采集完整性仍待核实。
