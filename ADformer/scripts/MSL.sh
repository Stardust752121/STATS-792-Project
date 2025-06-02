export CUDA_VISIBLE_DEVICES=1

python main_OT.py --k 3 --num_proto 12 --len_map 16   --mode train --anomaly_ratio 0.8 --dataset MSL --data_path dataset/MSL --input_c 55    --output_c 55

python main_OT.py --k 3 --num_proto 12 --len_map 16   --mode test --anomaly_ratio 0.8 --dataset MSL --data_path dataset/MSL --input_c 55 --output_c 55


## 修改版本:
python main_OT.py --k 5 --num_proto 12 --len_map 16   --mode train --anomaly_ratio 0.8 --dataset MSL --data_path C:/Users/Stardust/Desktop/UoA/2025.02/STATS792B/AD-Model/data/MSL --input_c 55    --output_c 55

python main_OT.py --k 5 --num_proto 12 --len_map 16   --mode test --anomaly_ratio 0.8 --dataset MSL --data_path C:/Users/Stardust/Desktop/UoA/2025.02/STATS792B/AD-Model/data/MSL --input_c 55 --output_c 55

# (2) 改变num_proto值 = 6/8/10/12/16
python main_OT.py --k 3 --num_proto 16 --len_map 16   --mode train --anomaly_ratio 0.8 --dataset MSL --data_path C:/Users/Stardust/Desktop/UoA/2025.02/STATS792B/AD-Model/data/MSL --input_c 55    --output_c 55

python main_OT.py --k 3 --num_proto 16 --len_map 16   --mode test --anomaly_ratio 0.8 --dataset MSL --data_path C:/Users/Stardust/Desktop/UoA/2025.02/STATS792B/AD-Model/data/MSL --input_c 55 --output_c 55

# (3) 改变len_map 值 = 6/8/10/12/16


# (4) 改变anomaly_ratio 值 = 0.5/0.6/0.7/0.8/0.9

#best：
python main_OT.py --k 3 --num_proto 12 --len_map 16   --mode train --anomaly_ratio 0.8 --dataset MSL --data_path C:/Users/Stardust/Desktop/UoA/2025.02/STATS792B/AD-Model/data/MSL --input_c 55 --output_c 55

python main_OT.py --k 3 --num_proto 12 --len_map 16   --mode test --anomaly_ratio 0.8 --dataset MSL --data_path C:/Users/Stardust/Desktop/UoA/2025.02/STATS792B/AD-Model/data/MSL --input_c 55 --output_c 55
