## Port Permissions
```python
sudo chmod 666 /dev/ttyACM0
sudo chmod 666 /dev/ttyACM1
```

## Calibration & Teleoperation

```bash
lerobot-teleoperate --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=follower_arm --teleop.type=so101_leader --teleop.port=/dev/ttyACM1 --teleop.id=leader_arm
```

## Camera

```bash
flatpak run com.obsproject.Studio
```

- To list the devices
```bash
v4l2-ctl --list-devices
```

## Data Collection

```bash
lerobot-teleoperate --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=follower_arm --robot.cameras="{ front: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}}" --teleop.type=so101_leader --teleop.port=/dev/ttyACM1 --teleop.id=leader_arm --display_data=true
```

## record

```bash
lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0\
    --robot.id=follower_arm \
    --robot.cameras="{ front: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}}" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=leader_arm \
    --display_data=true \
    --dataset.repo_id=selvaa/sponge-dice \
    --dataset.num_episodes=5 \
    --dataset.single_task="Grab the orange cube"
```


## LeRobot Dataset Format

```bash
dataset_repo/
├── meta/
│   ├── info.json          # Global config (FPS, robot type, features)
│   ├── stats.json         # Pre-calculated mean/std for normalization
│   └── episodes.jsonl     # Index of every episode (start/end timestamps)
├── videos/
│   ├── camera_01/         # e.g., "laptop_webcam"
│   │   ├── episode_0.mp4
│   │   └── ...
│   └── camera_02/         # e.g., "wrist_camera"
│       └── ...
└── data/
    ├── chunk-000/
    │   ├── episode_0.parquet
    │   └── ...
    └── ...
```

- q01 (1st Percentile): The value below which 1% of your data falls. Think of this as the "practical minimum."
- q99 (99th Percentile): The value below which 99% of your data falls. Think of this as the "practical maximum."

### Why these stats?

Robots are noisy. Sometimes a sensor might glitch and report a value of 999999 for a split second, or 0.000001.

If you used min and max for normalization:
That single 999999 spike would stretch your entire scale. Your normal data (e.g., 0 to 100) would get squashed into a tiny range like 0.00001 to 0.0001, making it impossible for the neural network to learn.

By using q01 and q99:
The code effectively says: "Ignore the bottom 1% and the top 1% of values. They might be noise. Focus on the middle 98% where the real physics is happening."

### How it's used in training?

When the LeRobot training script loads your data, it often uses a Clamping strategy:

1. It looks at q99.

2. Any value in your dataset higher than q99 gets chopped (clamped) down to q99.

3. Any value lower than q01 gets raised to q01.

4. Then it normalizes the data to be between -1 and 1.

This ensures your neural network sees clean, consistent numbers, even if your hardware had a few hiccups during recording.


## Replay the episode
```bash
lerobot-replay \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0\
    --robot.id=follower_arm \
    --dataset.repo_id=selvaa/sponge-dice \
    --dataset.episode=0
```

## ACT policy evaluate
```bash
lerobot-record   --robot.type=so100_follower   --robot.port=/dev/ttyACM0   --robot.id=follower_arm   --robot.cameras='{ front: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}}'   --display_data=true   --dataset.repo_id=selvaa/eval_act_cube-pick-finetune   --dataset.num_episodes=10   --dataset.single_task="pick the cube"   --policy.path=selvaa/act_policy
```
