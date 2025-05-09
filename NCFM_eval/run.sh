CUDA_VISIBLE_DEVICES=3 python NCFM_hard_lzy.py --dataset cifar100 --data_path /root/DD-Ranking/baselines/NCFM/CIFAR100_IPC10.pt --config /root/DD-Ranking/configs/NCFM/CIFAR100/IPC10.yaml

CUDA_VISIBLE_DEVICES=2 python NCFM_hard_lzy.py --dataset cifar100 --data_path /root/DD-Ranking/baselines/NCFM/CIFAR100_IPC50.pt --config /root/DD-Ranking/configs/NCFM/CIFAR100/IPC50.yaml

CUDA_VISIBLE_DEVICES=0 python NCFM_sl.py  --data_path /root/DD-Ranking/baselines/NCFM/Tiny-ImageNet_IPC50.pt --config /root/DD-Ranking/configs/NCFM/TinyImageNet/IPC50.yaml

CUDA_VISIBLE_DEVICES=7 python NCFM_hl.py  --data_path /root/DD-Ranking/baselines/NCFM/CIFAR10_IPC1.pt --config /root/DD-Ranking/configs/NCFM/CIFAR10/IPC1.yaml

CUDA_VISIBLE_DEVICES=6 python NCFM_hl.py  --data_path /root/DD-Ranking/baselines/NCFM/CIFAR10_IPC10.pt --config /root/DD-Ranking/configs/NCFM/CIFAR10/IPC10.yaml

CUDA_VISIBLE_DEVICES=5 python NCFM_hl.py  --data_path /root/DD-Ranking/baselines/NCFM/CIFAR100_IPC1.pt --config /root/DD-Ranking/configs/NCFM/CIFAR100/IPC1.yaml

CUDA_VISIBLE_DEVICES=4 python NCFM_hl.py  --data_path /root/DD-Ranking/baselines/NCFM/TinyImageNet_IPC10.pt --config /root/DD-Ranking/configs/NCFM/TinyImageNet/IPC10.yaml

CUDA_VISIBLE_DEVICES=3 python NCFM_hl.py  --data_path /root/DD-Ranking/baselines/NCFM/TinyImageNet_IPC50.pt --config /root/DD-Ranking/configs/NCFM/TinyImageNet/IPC50.yaml




CUDA_VISIBLE_DEVICES=6 python NCFM_eval/run_NCFM_script.py  --data_path /home/tiger/Documents/NCFM/DD-Ranking/NCFM_BEST_CKPT/CIFAR10_IPC1.pt --config /home/tiger/Documents/NCFM/DD-Ranking/configs/NCFM/CIFAR10/IPC1.yaml --softlabel

CUDA_VISIBLE_DEVICES=0 python NCFM_eval/run_NCFM_script.py  --data_path /home/tiger/Documents/NCFM/DD-Ranking/NCFM_BEST_CKPT/CIFAR10_IPC1.pt --config /home/tiger/Documents/NCFM/DD-Ranking/configs/NCFM/CIFAR10/IPC1.yaml --softlabel

CUDA_VISIBLE_DEVICES=0 python NCFM_eval/run_NCFM_script.py  --data_path /home/tiger/Documents/NCFM/DD-Ranking/NCFM_BEST_CKPT/CIFAR10_IPC10.pt --config /home/tiger/Documents/NCFM/DD-Ranking/configs/NCFM/CIFAR10/IPC10.yaml --softlabel

CUDA_VISIBLE_DEVICES=1 python NCFM_eval/run_NCFM_script.py  --data_path /home/tiger/Documents/NCFM/DD-Ranking/NCFM_BEST_CKPT/CIFAR10_IPC50.pt --config /home/tiger/Documents/NCFM/DD-Ranking/configs/NCFM/CIFAR10/IPC50.yaml --softlabel

CUDA_VISIBLE_DEVICES=2 python NCFM_eval/run_NCFM_script.py  --data_path /home/tiger/Documents/NCFM/DD-Ranking/NCFM_BEST_CKPT/CIFAR100_IPC1.pt --config /home/tiger/Documents/NCFM/DD-Ranking/configs/NCFM/CIFAR100/IPC1.yaml --softlabel

CUDA_VISIBLE_DEVICES=3 python NCFM_eval/run_NCFM_script.py  --data_path /home/tiger/Documents/NCFM/DD-Ranking/NCFM_BEST_CKPT/CIFAR100_IPC10.pt --config /home/tiger/Documents/NCFM/DD-Ranking/configs/NCFM/CIFAR100/IPC10.yaml --softlabel

CUDA_VISIBLE_DEVICES=4 python NCFM_eval/run_NCFM_script.py  --data_path /home/tiger/Documents/NCFM/DD-Ranking/NCFM_BEST_CKPT/CIFAR100_IPC50.pt --config /home/tiger/Documents/NCFM/DD-Ranking/configs/NCFM/CIFAR100/IPC50.yaml --softlabel

CUDA_VISIBLE_DEVICES=5 python run_NCFM_script.py  --data_path /home/tiger/Documents/NCFM/DD-Ranking/NCFM_BEST_CKPT/TinyImageNet_IPC1.pt --config /home/tiger/Documents/NCFM/DD-Ranking/configs/NCFM/TinyImageNet/IPC1.yaml --softlabel

CUDA_VISIBLE_DEVICES=6 python run_NCFM_script.py  --data_path /home/tiger/Documents/NCFM/DD-Ranking/NCFM_BEST_CKPT/TinyImageNet_IPC10.pt --config /home/tiger/Documents/NCFM/DD-Ranking/configs/NCFM/TinyImageNet/IPC10.yaml --softlabel

CUDA_VISIBLE_DEVICES=7 python run_NCFM_script.py  --data_path /home/tiger/Documents/NCFM/DD-Ranking/NCFM_BEST_CKPT/TinyImageNet_IPC50.pt --config /home/tiger/Documents/NCFM/DD-Ranking/configs/NCFM/TinyImageNet/IPC50.yaml --softlabel
