


# Dataset Details
![Dataset Details](../figures/dataset_details.png)


## Downloading Videos
- Go to  -->  [![Hugging Face Dataset](https://img.shields.io/badge/Hugging%20Face-ual--bench-blue)](https://huggingface.co/datasets/hasnat79/ual_bench)
- files and versions
- download the tar file
- untar the file
  - For example: 
```bash
tar -xvf uag_oops.tar
```
- After downloading and setting the videos in the correct directory, you can use the data loaders modules from [src/dataloaders](../../src/dataloaders) to directly load the videos and annotations in your code. For example:

```python

from src.dataloaders.uag_oops_loader import UAGOopsLoader
uag_oops = UAGOopsLoader()
for video_id, video_info in uag_oops:
    print(video_id, video_info)
```

## Tree Structure of the directory
```bash 
├── charades_sta
│   ├── Charades_STA_test.json
│   └── videos
│       └── 3MSZA.mp4 ...
├── README.md
├── uag_funqa
│   ├── uag_funqa_dataset.json
│   └── videos
│       └── H_A_101_1433_1631.mp4 ...
├── uag_oops
│   ├── oops_uag_paper_version.json
│   └── videos
│       └── 34\ Funny\ Kid\ Nominees\ -\ FailArmy\ Hall\ Of\ Fame\ (May\ 2017)0.mp4 ...
├── uag_oops_train_instruct
│   ├── uag_oops_train_instruct_dataset.json
│   └── videos
│       └── 25\ Best\ Trampoline\ Fail\ Nominees\ -\ FailArmy\ Hall\ of\ Fame\ (July\ 2017)6.mp4  ...
└── uag_ssbd
    ├── ssbd_paper_version.json
    └── videos
        ├── v_ArmFlapping_01.mp4 ...
```

## Annotations
- Charades_STA
  - `Charades_STA_test.json` contains the test set of Charades_STA dataset.
  - the start time, end time and description are from taken from original dataset and regorganized to test in our setting.
  - You can download the videos from [hf/ual_bench](https://huggingface.co/datasets/hasnat79/ual_bench) as well.
- UAG-OOPS
  - `oops_uag_paper_version.json` contains the UAG-Oops dataset.
  - It contains start time, end time and description of the videos.
- UAG-SSBD
  - `ssbd_paper_version.json` contains the UAG-SSBD dataset.
  - It contains start time, end time and description of the videos.
  - the key 'time' contains start and end time (seconds) separated by colon.
- FunQA
  - `uag_funqa_dataset.json` contains the FunQA dataset.
  - the key 'instruction' is used as a description / language query for the videos
- UAG-OOPS Train Instruct
  - `uag_oops_train_instruct_dataset.json` contains the UAG-OOPS instruction-tune dataset.
  - format: 
```python
[
    {
        "video": "25 Best Trampoline Fail Nominees - FailArmy Hall of Fame (July 2017)11.mp4",
        "QA": [
            {
                "q": "Find the start time and end time of the query below from the video.\n          Query: man attempted to jump from trampoline into pool man tripped on trampoline and fell onto the ground",
                "a": "start_time: 1.689396, end_time: 6.0"
            }
        ]
    }, ...
]
```
    

#### note


- **[original SSBD](https://rolandgoecke.net/research/datasets/ssbd/)** has listed 75 videos. Among them 58 are available on YouTube for download. uag-ssbd dataset is a subset of SSBD dataset.
