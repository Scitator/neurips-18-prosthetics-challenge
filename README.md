# NeurIPS 2018: AI for Prosthetics Challenge – 3rd place solution

![alt text](./poster.png)

## Attention

`git clone --recursive git@github.com:scitator/neurips-18-prosthetics-challenge.git`

## How2run

1. System requirements – redis and [Anaconda](www.anaconda.com/download)

    `sudo apt install redis-server`

2. Python requirements

    ```bash
    conda create -n opensim-rl -c kidzik opensim python=3.6.1
    activate opensim-rl
    source activate opensim-rl
    conda install -c conda-forge lapack git
    conda install pytorch torchvision -c pytorch
    pip install git+https://github.com/stanfordnmbl/osim-rl.git
    pip install git+https://github.com/pytorch/tnt.git@master
    pip install pyaml tensorboardX jpeg4py albumentations    
    ```

3. Mujoco example

    For installation issues follow [official guide](http://www.mujoco.org).

    ```bash
    redis-server --port 12000
    export GPUS=""
    CUDA_VISIBLE_DEVICES="$GPUS" PYTHONPATH=. \
        python rl/offpolicy/scripts/run_trainer.py \
        --config=experiments/mujoco/ecritic_quantile.yml
    
    CUDA_VISIBLE_DEVICES="" PYTHONPATH=. \
        python rl/offpolicy/scripts/run_samplers.py \
        --config=experiments/mujoco/ecritic_quantile.yml
    
    CUDA_VISIBLE_DEVICE="" tensorboard \
       --logdir=./experiments/logs
    ```

4. L2R example – ensemble training

    1. Redis
        ```bash
        redis-server --port 13131
        ```

    2. FC
        ```bash
        export GPUS=""
        CUDA_VISIBLE_DEVICES="$GPUS" PYTHONPATH=. \
            python catalyst/rl/offpolicy/scripts/run_trainer.py \
            --config=experiments/prosthetics/ecritic_quantile_fc.yml
        
        CUDA_VISIBLE_DEVICES="" PYTHONPATH=. \
            python catalyst/rl/offpolicy/scripts/run_samplers.py \
            --config=experiments/prosthetics/ecritic_quantile_fc.yml
        ```

    3. LAMA
        ```bash
        export GPUS=""
        CUDA_VISIBLE_DEVICES="$GPUS" PYTHONPATH=. \
            python catalyst/rl/offpolicy/scripts/run_trainer.py \
            --config=experiments/prosthetics/ecritic_quantile_lama.yml
        
        CUDA_VISIBLE_DEVICES="" PYTHONPATH=. \
            python catalyst/rl/offpolicy/scripts/run_samplers.py \
            --config=experiments/prosthetics/ecritic_quantile_lama.yml
        ```
    
    4. Monitoring
        ```bash

        CUDA_VISIBLE_DEVICE="" tensorboard \
           --logdir=./experiments/logs
        ```

5. L2R submit test
    ```bash
    bash ./submit/run.sh
    ```


## Additional links

[Medium post](https://medium.com/@dbrainio/neurips-to-make-the-most-of-it-and-get-to-the-top-cb103d5cdf00?fbclid=IwAR3uNq10PjiS65KjTeE0DElOkg1lTdiMzgkPOc-RSFnnqOK4gB_ftaFOWSg)

[Video](https://www.youtube.com/watch?v=uGL6jcSmGoA&feature=youtu.be)


## Citation
Please cite the following paper if you feel this repository useful.
```
@article{catalyst_rl,
  title={Catalyst.RL: A Distributed Framework for Reproducible RL Research},
  author = {Kolesnikov, Sergey and Hrinchuk, Oleksii},
  journal={arXiv preprint arXiv:1903.00027},
  year={2019}
}
```
