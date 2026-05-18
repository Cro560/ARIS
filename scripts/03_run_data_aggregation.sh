#!/bin/bash
echo "开始执行数据聚合流程..."

# 创建输出目录
mkdir -p output

# 依次执行各个聚合脚本
echo "=== 处理主表 ==="
python3 main_table.py

echo "=== 处理入院信息 ==="
python3 admissions_table.py

echo "=== 处理诊断信息 ==="
python3 diagnoses_table.py

echo "=== 处理实验室检查 ==="
python3 labevents_table.py

echo "=== 处理用药记录 ==="
python3 prescriptions_table.py

echo "=== 处理微生物检查 ==="
python3 microbiology_table.py

echo "=== 处理出院记录 ==="
python3 discharge_table.py

echo "=== 处理出院详情 ==="
python3 discharge_detail_table.py

echo "=== 合并所有表 ==="
python3 merge_all_tables.py

echo "数据聚合流程完成！"
