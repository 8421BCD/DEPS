# python main_no_doc_inter.py --mode 0 --cudaid 0 --seed 888 --batch_size 64 --lr 0.001 --epochs 5 --in_path ../../data/ValidQLSample_HF --test_score_path result/test_score.txt
# python main_doc_inter.py --mode 0 --cudaid 2 --seed 888 --batch_size 64 --lr 0.001 --epochs 5 --in_path ../../data/ValidQLSample_HF --test_score_path result/test_score.txt
python main_doc_inter_fine_grained.py --mode 1 --cudaid 0 --seed 888 --batch_size 64 --lr 0.001 --epochs 5 --in_path ../../data/ValidQLSample_HF --test_score_path result/test_score.txt
# python main_doc_inter_fine_grained_version2.py --mode 0 --cudaid 1 --seed 888 --batch_size 64 --lr 0.001 --epochs 5 --in_path ../../data/ValidQLSample_HF --test_score_path result/test_score.txt


