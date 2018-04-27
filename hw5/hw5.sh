#run python3.6
git clone https://gitlab.com/yujheli/ML_HW5_model.git
python3 test.py --model ML_HW5_model/model_dim_15.h5 --test $1 --output $2 --user2id user2id.npy --movie2id movie2id.npy
