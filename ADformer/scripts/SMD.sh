python main_OT.py --k 1 --num_proto 16 --len_map 16   --mode train --anomaly_ratio 0.7 --dataset SMD --data_path dataset/SMD --input_c 38    --output_c 38

python main_OT.py --k 1 --num_proto 16 --len_map 16   --mode test --anomaly_ratio 0.7 --dataset SMD --data_path dataset/SMD --input_c 38    --output_c 38


## 2.适配修改版本:
python main_OT.py --k 1 --num_proto 16 --len_map 16   --mode train --anomaly_ratio 0.7 --dataset SMD --data_path C:/Users/Stardust/Desktop/UoA/2025.02/STATS792B/AD-Model/data/SMD --input_c 38    --output_c 38

python main_OT.py --k 1 --num_proto 16 --len_map 16   --mode test --anomaly_ratio 1 --dataset SMD --data_path C:/Users/Stardust/Desktop/UoA/2025.02/STATS792B/AD-Model/data/SMD --input_c 38    --output_c 38

## 3.Parameter Sensitivity study：
# (1) 改变K值 = 1/2/3/4/5 (最终loss中的超参数，控制 similarity loss 的权重)
python main_OT.py --k 5 --num_proto 16 --len_map 16   --mode train --anomaly_ratio 0.7 --dataset SMD --data_path C:/Users/Stardust/Desktop/UoA/2025.02/STATS792B/AD-Model/data/SMD --input_c 38    --output_c 38

python main_OT.py --k 5 --num_proto 16 --len_map 16   --mode test --anomaly_ratio 0.7 --dataset SMD --data_path C:/Users/Stardust/Desktop/UoA/2025.02/STATS792B/AD-Model/data/SMD --input_c 38    --output_c 38

# (2) 改变num_proto值 = 6/8/10/12/16
python main_OT.py --k 4  --num_proto 16 --len_map 16   --mode train --anomaly_ratio 0.7 --dataset SMD --data_path C:/Users/Stardust/Desktop/UoA/2025.02/STATS792B/AD-Model/data/SMD --input_c 38    --output_c 38

python main_OT.py --k 4 --num_proto 16 --len_map 16   --mode test --anomaly_ratio 0.7 --dataset SMD --data_path C:/Users/Stardust/Desktop/UoA/2025.02/STATS792B/AD-Model/data/SMD --input_c 38    --output_c 38

# (3) 改变len_map 值 = 6/8/10/12/16


# (4) 改变anomaly_ratio 值 = 0.5/0.6/0.7/0.8/0.9


##Best:
python main_OT.py --k 4 --num_proto 16 --len_map 16   --mode train --anomaly_ratio 1.0 --dataset SMD --data_path C:/Users/Stardust/Desktop/UoA/2025.02/STATS792B/AD-Model/data/SMD --input_c 38    --output_c 38

python main_OT.py --k 4 --num_proto 16 --len_map 16   --mode test --anomaly_ratio 1.4 --dataset SMD --data_path C:/Users/Stardust/Desktop/UoA/2025.02/STATS792B/AD-Model/data/SMD --input_c 38    --output_c 38
