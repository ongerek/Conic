**The Reach of Conic Geometry in Deep Classifiers: code**  
This repository contains the code behind the paper  
*Ö. N. Gerek, The Reach of Conic Geometry in Deep Classifiers, submitted for publ’cat’on, 2026.*  
The paper asks where conic geometry is present in a trained classifier. A norm-linear conic head,  
   
 gk(z) = ⟨wk, z − ak⟩ + αk‖z − ak‖ε, is used as a measuring  
   
 instrument. Its class regions are bounded by quadrics of eccentricity 1/τk, the loss over an affine  
   
 final layer is a rounded polyhedral cone, and along a line through the parameters of a rectifier backbone  
   
 the loss has only kinks that each carry the share of one image. The code trains ResNet-18 and ResNet-56 on CIFAR-10 and CIFAR-100 with affine, conic, prototype  
   
 and cosine heads, measures the trained networks, and rebuilds Tables 1–6 and Figs. 1–4 of the paper.  
The commands below are written for an Ubuntu machine with one NVIDIA GPU, on which the second half of  
   
 the experiments was run (an NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 96 GB, with 32 CPU  
   
 cores). [Other machines explains what to change elsewhere.](#anchor-1 "#anchor-1")  
**Contents**  
```  
src/                    Python (PyTorch)  
   conic_head.py           the conic head of eq. (1) and its prototype limit; self-tests  
   train.py                one training run: backbones, heads, SGD, logging, resume  
   summarize.py            all run logs in one table or one CSV  
   probe.py                measurements on trained checkpoints (Sections 5, 7)  
   loss_surface.py         two-dimensional loss slices (Figs. 2, 3; Table 1)  
   crease_lines.py         twenty planes and forty refined lines (Section 6)  
   geometry.py             flatness of the learned decision boundaries (Section 7.8)  
   kink_test.py            crease test on refined windows (Table 1)  
   toy_kinks.py            Proposition 11 on a small network, computed exactly (Section 6)  
   final_layer_slice.py    the Section 5 measurements on the slices of Fig. 2  
   wrinkles.py             512 against 4096 images (Section 6)  
   crease_stats.py         the Section 6 statistics of the planes and lines  
   tau_replay.py           tau trajectories and the decay-only paths (Fig. 4, Table 3)  
   tables.py               Tables 2-6 and the numbers quoted with them  
 scripts/                bash drivers  
   selftest.sh             checks the installation (and, with "full", the whole pipeline)  
   train.sh                all 166 training runs, several at a time on one GPU  
   analyze.sh              all measurements and all numbers  
   figures.sh              Figs. 1-4 with MATLAB or GNU Octave  
   env_report.sh           records the software and hardware versions  
 matlab/                 figure scripts (MATLAB and Octave)  
   cone_figures.m  make_fig_head_slice.m  make_fig_r56.m  make_fig_tau.m  crease_control.m  
   
```  
Everything the scripts produce goes into three folders, which are not part of the repository:  
   
 `data/` (CIFAR, downloaded on first use), `runs/` (one folder per training run) and `results/`  
   
 (all measurements, tables and figures).  
**Installation**  
Ubuntu 22.04 or newer, Python 3.10 or newer, an NVIDIA driver that supports CUDA 12.8. Blackwell GPUs  
   
 (RTX 50xx, RTX PRO 6000) need the CUDA 12.8 builds of PyTorch, that is, PyTorch 2.7 or newer.  
```  
sudo apt install -y python3-venv git  
 cd reach-of-conic-geometry  
 python3 -m venv .venv  
 source .venv/bin/activate  
 pip install --upgrade pip  
 pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128  
 pip install -r requirements.txt  
 python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"  
   
```  
The last line must print `True` and the name of the GPU. In every new terminal, activate the  
   
 environment again with `source .venv/bin/activate` from the repository folder. The scripts also find  
   
 `.venv/bin/python` by themselves.  
For the figures, either MATLAB (R2020a or newer) or GNU Octave:  
```  
sudo apt install -y octave gnuplot xvfb ghostscript fonts-freefont-otf  
   
```  
To record the exact versions next to the results, run once:  
```  
bash scripts/env_report.sh  
   
```  
**Check**  
```  
bash scripts/selftest.sh  
   
```  
This checks the analytic Hessian and curvature of the head against automatic differentiation, trains  
   
 two epochs on random images, runs the probe on random checkpoints and the plane and line measurements  
   
 on a random ResNet-56. It takes 1–2 minutes on the GPU machine (3–4 on a laptop) and downloads nothing. The last line  
   
 must be `SELFTEST PASSED`.  
```  
bash scripts/selftest.sh full  
   
```  
does the same and then runs the whole pipeline of the next section in miniature, on random images and  
   
 in a temporary copy of the repository (about 40 minutes on a two-core laptop CPU, much less on a  
   
 workstation). It tests the scripts and not the science.  
**Reproducing the paper**  
Three commands, each of which can be left running unattended. Each command skips work that is already  
   
 done, so after an interruption the same command continues where it stopped; an interrupted training  
   
 run continues from its last completed epoch.  
**1. Train** (about 15 hours on the RTX PRO 6000, all 166 runs)  
```  
mkdir -p logs  
 nohup bash scripts/train.sh all > logs/train.log 2>&1 &  
   
```  
The terminal prints a line like `[1] 12345` and returns; the runs continue after the terminal is  
   
 closed. A subset of the paper needs only some of the groups, for example  
   
 `bash scripts/train.sh main sweep` for Tables 2–4 and Fig. 4. The groups are:  
  
|  |  |  |  |  |  
| - | - | - | - | - |  
| **group** |  **runs** |  **names** |  **what they are** |  **used for** |  
| `main` |  35 |  `S_*`, `CTRL_*`, `LR8_*` |  affine and conic heads, ResNet-18, CIFAR-10, three settings, 5 seeds; conic head with batch norm at lr 0.01 |  Tables 2, 3; Fig. 4a; Sections 5, 7.1–7.4, 7.8 |  
| `sweep` |  32 |  `LR_*` |  learning-rate sweep, 20 epochs |  Table 4; Fig. 4b |  
| `ablation` |  7 |  `BASE_affine`, `A_floor1_eps05`, ... |  single-seed runs |  Table 5; Section 7.1 |  
| `resnet56` |  2 |  `R56_skip`, `R56_noskip` |  ResNet-56 with and without skip connections |  Section 6; Fig. 3; Table 1 |  
| `nwd` |  20 |  `NWD_*` |  conic head, ηk excluded from weight decay |  Table 6; Section 7.7 |  
| `proto` |  10 |  `PROTO_*` |  prototype limit of the head, wk = 0 |  Table 6; Section 7.7 |  
| `cosine` |  15 |  `COS_*` |  cosine head with learned scale |  Table 6; Section 7.7 |  
| `replicate` |  35 |  `G_*` |  the runs of `main` under new names |  Table 6; Section 7.7 |  
| `c100` |  10 |  `C100_*` |  affine and conic heads on CIFAR-100, 5 seeds |  Section 7.7 |  
  
   
All runs use SGD with momentum 0.9, weight decay 5·10−4, batch 128 and a cosine schedule over  
   
 100 epochs (20 in the sweep); the conic head uses ε = 0.5 (0.1 in one single-seed run) and starts at  
   
 τk = 1.5. The full option list of every run is written to `runs/<name>/config.json`, and  
   
 `LIST=1 bash scripts/train.sh all` prints every training command without running it.  
To watch the training:  
```  
tail -f logs/train.log                   # which runs started and ended (Ctrl-C stops only the watching)  
 tail -n 1 logs/train_lane*.log           # the last epoch of each lane, with seconds per epoch  
 nvidia-smi                               # GPU load  
 python3 src/summarize.py --prefix S_ --group     # accuracy and tau so far, mean and sd over seeds  
   
```  
To stop everything: `pkill -f scripts/train.sh; pkill -f src/train.py`.  
**2. Measure** (about 3–4 hours)  
```  
nohup bash scripts/analyze.sh > logs/analyze.log 2>&1 &  
   
```  
The steps are `logs probe slices crease geometry numbers`; `bash scripts/analyze.sh probe` runs one  
   
 step. A step that needs runs which are not there is skipped with a note.  
**3. Figures** (about a minute)  
```  
bash scripts/figures.sh  
   
```  
The numbers end up in `results/*.txt`, the figures in `results/figures/`.  
**What produces what**  
  
|  |  |  |  |  
| - | - | - | - |  
| **paper** |  **training group** |  **analysis step** |  **output** |  
| Fig. 1 |  – |  `figures.sh` |  `results/figures/cone_envelope.pdf`, `cone_cells.pdf` |  
| Section 5, Fig. 2 |  `main` |  `slices`, `numbers`, `figures.sh` |  `results/final_layer.txt`, `results/figures/fig_head_slice.pdf` |  
| Section 5, recession function and band |  `main`, `replicate` |  `probe` |  `results/probe/*/probe_finallayer.csv` |  
| Section 6, small network of Proposition 11 |  – |  `numbers` |  `results/toy_kinks.txt` |  
| Section 6, Fig. 3 |  `resnet56` |  `slices`, `figures.sh` |  `results/figures/fig_r56_skip.pdf`, `fig_r56_noskip.pdf` |  
| Section 6, 512 against 4096 images |  `resnet56` |  `slices`, `numbers` |  `results/wrinkles.txt` |  
| Table 1 |  `resnet56` |  `slices`, `numbers` |  `results/kink_test.txt` |  
| Section 6, twenty planes and the 4hD(h) bound |  `resnet56` |  `crease`, `numbers` |  `results/crease_stats.txt` |  
| Section 6, synthetic control |  – |  `figures.sh` |  `results/crease_control.txt` |  
| Tables 2, 3, 4, 5 |  `main`, `sweep`, `ablation` |  `logs`, `numbers` |  `results/tables.txt` |  
| Sections 7.1, 7.2, 7.4 |  `main`, `ablation` (the cell counts use every conic run) |  `probe`, `numbers` |  `results/tables.txt` |  
| Fig. 4 |  `main`, `sweep` |  `numbers`, `figures.sh` |  `results/figures/tau_trajectories.pdf` |  
| Table 6, Section 7.7 |  `nwd`, `proto`, `cosine`, `replicate`, `c100` |  `probe`, `numbers` |  `results/tables.txt` |  
| Section 7.8 |  `main` |  `geometry` |  `results/geometry.txt` |  
  
   
The per-run and per-class values behind these summaries are in `results/logs.csv` and  
   
 `results/probe/<family>/probe_{runs,classes,finallayer}.csv`, where the family is the first part of the  
   
 run name. The columns are described at the top of `src/probe.py`.  
Not produced by these scripts are the numerical examples at the end of Section 4, the two control  
   
 values in the last row of Table 1, and four numbers of Section 7.1 (escape of the training features,  
   
 inward pull, rescaling of the features, share of the norm terms in the gap).  
**Runtimes**  
Measured on the RTX PRO 6000 Blackwell with 32 CPU cores and four runs side by side, except where marked  
   
 as an estimate.  
  
|  |  |  
| - | - |  
| **work** |  **time** |  
| one ResNet-18 run, CIFAR-10, 100 epochs |  19–23 min while four run together (12–14 s per epoch); 80 such runs took 8 h 13 min |  
| one ResNet-18 run, CIFAR-100, 100 epochs |  26 min while four run together; the 10 runs of `c100` took 1 h 16 min |  
| `main` 3.6 h, `sweep` 0.6 h, `ablation` 0.75 h, `nwd` 2.1 h, `proto` 1.0 h, `cosine` 1.5 h, `replicate` 3.6 h, `c100` 1.3 h |  from the measured rate |  
| `resnet56` |  about 0.5 h (estimate) |  
| `train.sh all` |  about 15 h; without `replicate` about 11.5 h |  
| `crease` (20 planes, 40 lines) |  about 1 h (3 min per plane on an idle GPU, up to 17 min per plane while training runs) |  
| `probe` (134 checkpoints) |  1–2 h (estimate) |  
| `slices` (4 full slices, 2 with 4096 images, 6 refined windows, 2 final-layer slices) |  about 1 h (estimate) |  
| `geometry`, `numbers`, `figures.sh` |  a few minutes |  
  
   
The first half of the experiments (`main`, `sweep`, `ablation`, `resnet56`) was trained on an Apple M4  
   
 Mac mini, at 85–90 s per epoch for ResNet-18 (about 2.4 h per 100-epoch run) and 34–38 s per epoch for  
   
 ResNet-56.  
**Other machines**  
**Another NVIDIA GPU.** The installation command above usually works; for an older driver, take the  
   
 matching command from [https://pytorch.org/get-started/locally/ (Blackwell GPUs need a CUDA 12.8 build).  
   
 Each ResNet-18 run takes a few GB of GPU memory, but more lanes help only while the GPU has spare  
   
 compute: use `LANES=2` on a 24 GB card and `LANES=1` on a card  
   
 with 12 GB or less, for example `LANES=2 nohup bash scripts/train.sh all > logs/train.log 2>&1 &`.  
   
 Times scale roughly with the throughput of the GPU.](https://pytorch.org/get-started/locally/ "https://pytorch.org/get-started/locally/")  
**Fewer CPU cores.** The data loading of one run needs about four cores. `LANES` defaults to the number  
   
 of cores divided by four (at most four), and `WORKERS` to the cores per lane minus one (at most six).  
   
 Both can be set by hand.  
**Apple Silicon (MPS).** Install PyTorch with `pip install torch torchvision` (no index URL). Training  
   
 then runs on the Apple GPU, one run at a time:  
```  
ALLOW_NO_CUDA=1 LANES=1 WORKERS=0 nohup caffeinate -i bash scripts/train.sh main > logs/train.log 2>&1 &  
   
```  
`caffeinate -i` keeps the Mac awake, and `WORKERS=0` keeps the data loading in the training process,  
   
 as `src/train.py` does by default on a Mac (the Apple GPU, not the loading, limits the speed there). At about 2.4 h per run, `main` takes about 3.5 days and all groups about two  
   
 weeks. The analysis steps work as on Linux.  
**CPU only.** The self-tests, `numbers` and `figures.sh` run on any CPU. Training on a CPU is not  
   
 practical (several days per run), and `probe`, `slices` and `crease` are slow there.  
**Windows.** Use WSL2 with Ubuntu and follow the Linux instructions.  
**No internet on the GPU machine.** Copy the folders `cifar-10-batches-py` and `cifar-100-python`  
   
 (from the archives at https://www.cs.toronto.edu/\~kriz/cifar.html) into `data/`. Both are needed even  
   
 for CIFAR-10 alone, since the probe uses each test set as outside data for the other.  
**MATLAB or Octave.** `scripts/figures.sh` uses MATLAB when the `matlab` command exists and Octave  
   
 otherwise. The figure scripts in `matlab/` can also be run by hand: copy the CSVs named at the top of  
   
 each script into one folder, change into it, and type the script name.  
**Disk.** A ResNet-18 checkpoint (`final.pt`) takes 45 MB, so the 134 checkpoints take about 6 GB, and  
   
 CIFAR-10 and CIFAR-100 take 340 MB. During training, `ckpt.pt` (90 MB per run) holds the resume state  
   
 and is deleted when the run finishes. `src/train.py --no-final` skips `final.pt` for runs that only  
   
 need their log.  
**Notes on reproducibility**  
- Seeds fix the initialization and the order of the data, but cuDNN autotuning and the parallel data  
   
 loading make GPU runs nondeterministic at the level of rounding. Repeated runs therefore agree within  
   
 the spread of the seeds, not bit for bit. The `replicate` group measures this: on the RTX PRO 6000  
   
 its mean accuracies were within 0.16 points of the runs made on the Mac, and its mean values of τ  
   
 within 0.01.  
- The runs of `main`, `sweep`, `ablation` and `resnet56` in the paper were trained on the Mac with  
   
 earlier versions of the same scripts (same model, optimizer and schedule; later versions added  
   
 resuming, logging and speed-ups). `LR_conic_bn_0.8` diverged in its first epoch there before any row  
   
 was logged; the current script logs the diverging epoch.  
- The `lr` column of `log.csv` is read after the scheduler step, so it holds the rate of the next epoch.  
 `src/tau_replay.py` does not use it.  
- Batch normalization runs in inference mode in every measurement (probe, slices, planes and lines), as  
   
 Proposition 11 assumes.  
- p-values in `results/tables.txt` are exact two-sided permutation tests over the 252 divisions of two  
   
 groups of five runs; variance ratios use F(4, 4).  
**Citation**  
If you use this code, please cite the paper (see `CITATION.cff`):  
```  
Ö. N. Gerek, "The Reach of Conic Geometry in Deep Classifiers," submitted to Neural Networks, 2026.  
   
```  
**License**  
MIT; see `LICENSE`.  
