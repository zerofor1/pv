# 光伏物理模型

配置在本目录 `config.json`，默认使用上海城市参考坐标和 29.7 kWp/25 kW 示例阵列。这是可修改工程场景，不是各站实测容量。物理计算位于 `simulate.py`，批量入口为 `run.py`。

在项目根目录运行：

```powershell
.runtime/python/python.exe -m model.pv.run
.runtime/python/python.exe -m test.pv.validate_pv
```

读取 `data/weather/gfs/hourly.csv` 和 `data/weather/era5/hourly.csv`，输出各自目录内的 `pv_hourly.csv`。源哈希和验证报告只写入 `test/reports/pv`。ERA5 为参考光伏，GFS 为预报光伏。演示数据在 `test/fixtures/pv/demo_hourly.csv`，回归测试在 `test/pv`。购电优化只保存在项目材料的历史备份，不属于当前主流程。

## 输入和输出

必需列包括 `valid_time_utc`（含时区偏移的时段起点）、`interval_end_utc`（含时区偏移的时段终点）、`ghi_w_m2`（时段平均水平总辐照，W/m²）、`air_temp_c`（时段起点气温，°C）和 `wind_speed_m_s`（时段起点风速，m/s）。每行覆盖 `[起点, 终点)` 一小时。若输入带站点坐标，两列 `site_latitude`、`site_longitude` 须与配置相符。

ERA5-Land 的逐小时辐射能量需除以 3,600 s 才能作为 `ghi_w_m2`。如果所用产品的 SSRD 从 00 UTC 累加，需先在同一累计周期内作差，01 UTC 终止的首小时直接取首步，次日 00 UTC 对应前一日的最后一步。已按小时去累计的产品可直接换算。气温从 K 减去 273.15，10 m 风速用两分量的模，并以小时起点匹配气温和风速。[ECMWF ERA5-Land 数据说明](https://confluence.ecmwf.int/spaces/CKB/pages/140385202/ERA5-Land+data+documentation)

输出保留原天气字段，包括来源、时间、预报批次和数据划分等元数据，并附加：

- 时段中点太阳高度与方位、太阳在地平线以上的采样比例 `daylight_fraction`，以及模型实际使用的 `ghi_used_w_m2`。
- `dni_w_m2`（以小时水平直射量和太阳投影反推的等效法向直射辐照）、`dhi_w_m2`（水平散射辐照）、组件平面直射、散射、反射分量及总辐照 `poa_global_w_m2`。
- `module_temp_c`、降额前直流功率 `pv_dc_gross_kw`、逆变器输入 `pv_dc_kw`、逆变器削顶功率 `pv_inverter_clipping_kw`、时段平均可用交流功率 `pv_ac_kw` 和时段电量 `pv_energy_kwh`。步长为 1 h，因此功率和电量两列数值相同，单位分别为 kW 和 kWh。
- 温度影响、入射角、积灰、失配、直流布线、逆变转换、削顶、交流布线分别输出功率差，便于逐时闭合损耗账。另输出组件总表面积、水平投影和 `pv_config_id`。

## 模型和参数

1. 按 NOAA 太阳位置近似，在每小时内的第 5、15、25、35、45、55 分钟求太阳几何并平均投影。用 Erbs 的小时晴空指数关系从 GHI 分解等效 DNI 和 DHI。低太阳高度时将直射分量置零，避免接近地平线时分母过小。六个采样点均低于地平线时，辐照和出力归零。[NOAA 太阳位置公式](https://gml.noaa.gov/grad/solcalc/solareqns.PDF) · [Erbs 等，1982](https://doi.org/10.1016/0038-092X(82)90302-4) · [pvlib Erbs 数值阈值](https://pvlib-python.readthedocs.io/en/stable/reference/generated/pvlib.irradiance.erbs.html)
2. 将直射辐照投影至组件平面，加入各向同性天空散射 `DHI × (1 + cos β) / 2` 和地面反射 `GHI × ρ × (1 − cos β) / 2`。β 为倾角，ρ 为反照率。[Sandia 各向同性天空散射模型](https://pvpmc.sandia.gov/modeling-guide/1-weather-design-inputs/plane-of-array-poa-irradiance/calculating-poa-irradiance/poa-sky-diffuse/isotropic-sky-diffuse-model/)
3. 用 Faiman 关系 `T_module = T_air + POA / (U0 + U1 × v)` 估计温度，以组件温度近似电池片温度。默认 `U0 = 25`、`U1 = 6.84`，并暂以输入的 10 m 风速近似组件附近风速。`wind_speed_multiplier` 可用于调整风速口径。[Faiman，2008](https://doi.org/10.1002/pip.813) · [Sandia Faiman 参数说明](https://pvpmc.sandia.gov/modeling-guide/2-dc-module-iv/module-temperature/faiman-module-temperature-model/)
4. 按 PVWatts v5 第 8 式计算降额前直流功率，再对直射辐照作入射角修正，逐项计算积灰、失配和线路损耗。用第 10 式的逆变器效率曲线得到交流功率，限幅后扣除交流布线损耗。小时内气象波动和动态局部遮挡未展开，待机用电计入站内辅助负荷的合计假设。[Dobos，2014，PVWatts v5 手册](https://pvwatts.nrel.gov/downloads/pvwattsv5.pdf)

当前配置采用上海代表坐标（31.2304°N、121.4737°E），66 块 450 W 组件共 29.7 kWp，组件总表面积 131.875128 m²。温度系数为 −0.0029/°C，正南安装、20° 倾角、25 kW 逆变器、96% 逆变器额定效率、0.20 反照率。直射光增加 ASHRAE 入射角修正。积灰 2%、失配 1%、直流线路 1.5% 和交流线路 1% 按顺序计算，旧版综合直流损耗设为零。交流布线后母线出力上限为 24.75 kW。组件规格参考厂家数据表，其余设备、损耗和安装参数属于工程假设。

## 检查结果

当前两套上海天气均已生成 35,064 小时出力，数值见 `data/weather/`，源文件与配置指纹见 `data/energy/input_manifest.json`。`model/pv/demo_hourly.csv` 是根据当前上海坐标重建的两个合成示例日，不代表电表实测。损耗账与组件容量一致性具有回归测试，实测发电误差仍需真实电表数据。
