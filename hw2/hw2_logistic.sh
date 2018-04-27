#train
#python LogisticRegression.py --train --train_data_path $3 --train_label_path $4 --test_data_path $5
python LogisticRegression.py --infer --train_data_path $3 --train_label_path $4 --test_data_path $5 --output_dir $6
