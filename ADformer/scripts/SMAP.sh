export CUDA_VISIBLE_DEVICES=1

python main_OT.py --k 2 --num_proto 12 --len_map 6    --mode train --anomaly_ratio 0.7 --dataset SMAP --data_path dataset/SMAP --input_c 25    --output_c 25

python main_OT.py --k 2 --num_proto 12 --len_map 6    --mode test --anomaly_ratio 0.7 --dataset SMAP --data_path dataset/SMAP --input_c 25    --output_c 25

## 修改版本:
python main_OT.py --k 2 --num_proto 12 --len_map 6    --mode train --anomaly_ratio 0.7 --dataset SMAP --data_path C:/Users/Stardust/Desktop/UoA/2025.02/STATS792B/AD-Model/data/SMAP --input_c 25    --output_c 25

python main_OT.py --k 2 --num_proto 12 --len_map 6    --mode test --anomaly_ratio 0.7 --dataset SMAP --data_path C:/Users/Stardust/Desktop/UoA/2025.02/STATS792B/AD-Model/data/SMAP --input_c 25    --output_c 25

# 3.Parameter Sensitivity study：
# (1) 改变K值 = 1/2/3/4/5 (最终loss中的超参数，控制 similarity loss 的权重)
python main_OT.py --k 5 --num_proto 12 --len_map 6    --mode train --anomaly_ratio 0.7 --dataset SMAP --data_path C:/Users/Stardust/Desktop/UoA/2025.02/STATS792B/AD-Model/data/SMAP --input_c 25    --output_c 25

python main_OT.py --k 5 --num_proto 12 --len_map 6    --mode test --anomaly_ratio 0.7 --dataset SMAP --data_path C:/Users/Stardust/Desktop/UoA/2025.02/STATS792B/AD-Model/data/SMAP --input_c 25    --output_c 25

##使用原百分位阈值设定法:
python main_OT.py --k 2 --num_proto 12 --len_map 6   --mode train --anomaly_ratio 1.5 --dataset SMAP --data_path C:/Users/Stardust/Desktop/UoA/2025.02/STATS792B/AD-Model/data/SMAP --input_c 25     --output_c 25

python main_OT.py --k 2 --num_proto 12 --len_map 6    --mode test --anomaly_ratio 1.4 --dataset SMAP --data_path C:/Users/Stardust/Desktop/UoA/2025.02/STATS792B/AD-Model/data/SMAP --input_c 25     --output_c 25

##使用SPOT动态阈值设定法:×不可行
#理论best: 2,12,6,>1.0
python main_OT.py --k 3 --num_proto 12 --len_map 16   --mode train --anomaly_ratio 1.0 --dataset SMAP --data_path C:/Users/Stardust/Desktop/UoA/2025.02/STATS792B/AD-Model/data/SMAP --input_c 25     --output_c 25

python main_OT.py --k 3 --num_proto 12 --len_map 16   --mode test  --anomaly_ratio 1.45 --dataset SMAP --data_path C:/Users/Stardust/Desktop/UoA/2025.02/STATS792B/AD-Model/data/SMAP --input_c 25     --output_c 25
# Accuracy : 0.9701, Precision : 0.9628, Recall : 0.7968, F-score : 0.8719

python main_OT.py --k 3 --num_proto 12 --len_map 6   --mode test  --anomaly_ratio 1.0 --dataset SMAP --data_path C:/Users/Stardust/Desktop/UoA/2025.02/STATS792B/AD-Model/data/SMAP --input_c 25     --output_c 25
#Accuracy : 0.9648, Precision : 0.9386, Recall : 0.7754, F-score : 0.8492
python main_OT.py --k 2 --num_proto 12 --len_map 10   --mode test  --anomaly_ratio 1.0 --dataset SMAP --data_path C:/Users/Stardust/Desktop/UoA/2025.02/STATS792B/AD-Model/data/SMAP --input_c 25     --output_c 25

python main_OT.py --k 2 --num_proto 12 --len_map 10   --mode train  --anomaly_ratio 1.0 --dataset SMAP --data_path C:/Users/Stardust/Desktop/UoA/2025.02/STATS792B/AD-Model/data/SMAP --input_c 25     --output_c 25


#实际best: 2,12,6,>1.0
python main_OT.py --k 2 --num_proto 12 --len_map 6   --mode test  --anomaly_ratio 0.7 --dataset SMAP --data_path C:/Users/Stardust/Desktop/UoA/2025.02/STATS792B/AD-Model/data/SMAP --input_c 25     --output_c 25
# Accuracy : 0.9405, Precision : 0.9416, Recall : 0.5704, F-score : 0.7104

python main_OT.py --k 2 --num_proto 12 --len_map 6   --mode test  --anomaly_ratio 0.8 --dataset SMAP --data_path C:/Users/Stardust/Desktop/UoA/2025.02/STATS792B/AD-Model/data/SMAP --input_c 25     --output_c 25
# Accuracy : 0.9475, Precision : 0.9397, Recall : 0.6300, F-score : 0.7543

python main_OT.py --k 2 --num_proto 12 --len_map 6   --mode test  --anomaly_ratio 0.9 --dataset SMAP --data_path C:/Users/Stardust/Desktop/UoA/2025.02/STATS792B/AD-Model/data/SMAP --input_c 25     --output_c 25
#Accuracy : 0.9470, Precision : 0.9319, Recall : 0.6315, F-score : 0.7528

python main_OT.py --k 2 --num_proto 12 --len_map 6   --mode test  --anomaly_ratio 1.0 --dataset SMAP --data_path C:/Users/Stardust/Desktop/UoA/2025.02/STATS792B/AD-Model/data/SMAP --input_c 25     --output_c 25
#Accuracy : 0.9782, Precision : 0.9476, Recall : 0.8785, F-score : 0.9117


