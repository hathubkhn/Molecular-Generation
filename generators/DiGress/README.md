- Env: digress_clean - đừng cài thêm gì please :)

- Train:
%cd src/

Chỉnh các config cần thiết trong configs:
    Về training: /train/train_default.yaml
    Về model: /model/discrete.yaml
    Thông số khác (pretrained weight, số molecule gen, ...): /general/general_default.yaml

Chạy python main.py --dataset={dataset name}