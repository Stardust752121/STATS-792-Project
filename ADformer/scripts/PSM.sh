python main_OT.py --k 1 --num_proto 10 --len_map 10     --mode train --anomaly_ratio 0.6 --dataset PSM --data_path dataset/PSM --input_c 51    --output_c 51

python main_OT.py --k 1 --num_proto 10 --len_map 10     --mode test --anomaly_ratio 0.6 --dataset SWaT --data_path dataset/PSM --input_c 51    --output_c 51


## 修改版本:
python main_OT.py --k 1 --num_proto 10 --len_map 10     --mode train --anomaly_ratio 0.6 --dataset PSM --data_path C:/Users/Stardust/Desktop/UoA/2025.02/STATS792B/AD-Model/data/PSM --input_c 25    --output_c 25

python main_OT.py --k 1 --num_proto 10 --len_map 10     --mode test --anomaly_ratio 0.6 --dataset PSM --data_path C:/Users/Stardust/Desktop/UoA/2025.02/STATS792B/AD-Model/data/PSM --input_c 25    --output_c 25

## 3.Parameter Sensitivity study：
# (1) 改变K值 = 1/2/3/4/5
python main_OT.py --k 5  --num_proto 10 --len_map 10     --mode train --anomaly_ratio 0.6 --dataset PSM --data_path C:/Users/Stardust/Desktop/UoA/2025.02/STATS792B/AD-Model/data/PSM --input_c 25    --output_c 25

python main_OT.py --k 5 --num_proto 10 --len_map 10     --mode test --anomaly_ratio 0.6 --dataset PSM --data_path C:/Users/Stardust/Desktop/UoA/2025.02/STATS792B/AD-Model/data/PSM --input_c 25    --output_c 25


# (2) 改变num_proto值 = 8/10/12/16/
python main_OT.py --k 1 --num_proto 6 --len_map 10     --mode train --anomaly_ratio 0.6 --dataset PSM --data_path C:/Users/Stardust/Desktop/UoA/2025.02/STATS792B/AD-Model/data/PSM --input_c 25    --output_c 25

python main_OT.py --k 1 --num_proto 6 --len_map 10     --mode test --anomaly_ratio 0.6 --dataset PSM --data_path C:/Users/Stardust/Desktop/UoA/2025.02/STATS792B/AD-Model/data/PSM --input_c 25    --output_c 25


# (3) 改变len_map 值 = 8/10/12/16
python main_OT.py --k 3 --num_proto 8 --len_map 10     --mode train --anomaly_ratio 0.6 --dataset PSM --data_path C:/Users/Stardust/Desktop/UoA/2025.02/STATS792B/AD-Model/data/PSM --input_c 25    --output_c 25

python main_OT.py --k 3 --num_proto 8 --len_map 10     --mode train --anomaly_ratio 0.6 --dataset PSM --data_path C:/Users/Stardust/Desktop/UoA/2025.02/STATS792B/AD-Model/data/PSM --input_c 25    --output_c 25


# (4) 改变anomaly_ratio 值 = 0.6/0.7/0.8/0.9
python main_OT.py --k 1 --num_proto 8 --len_map 10     --mode train --anomaly_ratio 0.6 --dataset PSM --data_path C:/Users/Stardust/Desktop/UoA/2025.02/STATS792B/AD-Model/data/PSM --input_c 25    --output_c 25

python main_OT.py --k 3 --num_proto 8 --len_map 10     --mode train --anomaly_ratio 0.6 --dataset PSM --data_path C:/Users/Stardust/Desktop/UoA/2025.02/STATS792B/AD-Model/data/PSM --input_c 25    --output_c 25

# Best:
python main_OT.py --k 1 --num_proto 10 --len_map 12     --mode train --anomaly_ratio 0.6 --dataset PSM --data_path C:/Users/Stardust/Desktop/UoA/2025.02/STATS792B/AD-Model/data/PSM --input_c 25    --output_c 25

python main_OT.py --k 1 --num_proto 10 --len_map 12     --mode test --anomaly_ratio 1.4 --dataset PSM --data_path C:/Users/Stardust/Desktop/UoA/2025.02/STATS792B/AD-Model/data/PSM --input_c 25    --output_c 25
