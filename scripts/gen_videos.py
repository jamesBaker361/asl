for lr in [0.001,0.0001]:
    for frames in [2,4]:
        name=f"video_{lr}_{frames}"
        command=f"sbatch -J video --constraint=L40S --err=slurm_chip/hands/{name}.err --out=slurm_chip/hands/{name}.out "
        command+=f" runpygpu_chip_l40.sh train_video.py --lr {lr} --frames {frames} --name jlbaker361/video_{name} "
        print(command)