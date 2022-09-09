run_chirality:
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python train.py --learning_rate 0.001 --batch_szie 32 --name_different 5

run_BYOL_voc:
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=1 python voc_byol.py Your_Data_Path --HASH_BIT 64 --batch-size 24 --epochs 50 > ./log/chiral_100_achiral_0_64

run_cauchy_coco:
python pure_coco.py --imgdirabspath Your_Train_Data_Path --txtabspath Your_Train_Data_TxT_Path --valimgdirabspath Your_Test_Data_Path --valtxtabspath Your_Test_Data_TxT_Path --HASH_TASK --batch-size 32 --HASH_BIT 64 --epochs 50 -t -v > ./voc_log_files/mscoco/chiral_0_achiral_100_64bit


run_HashNet_voc:
python HashNet.py --bit_size 32 --learning_rate 0.01 --dataset Your_Data_path --data_path Your_Abs_Data_path > ./voc_log_files/voc2007/chiral_0_achiral_100_r10_001_32
