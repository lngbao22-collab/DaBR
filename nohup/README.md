# Background Process Scripts

This folder contains helper scripts to run DaBR training/prediction in the background using `nohup`.

## Scripts

### `run.sh` - Start a background process
Runs the main.py script in the background with specified arguments.

**Usage:**
```bash
nohup/run.sh [--dataset DATASET] [--config CONFIG_FILE] [--mode MODE] [--checkpoint_path PATH]
```

**Examples:**
```bash
# Train on WN18RR
nohup/run.sh --dataset WN18RR --config ./config/WN18RR_test.json --mode train

# Train on FB15K237
nohup/run.sh --dataset FB15K237 --config ./config/FB15K237.json --mode train

# Run prediction
nohup/run.sh --dataset WN18RR --config ./config/WN18RR_test.json --mode predict --checkpoint_path ./logs/WN18RR/checkpoints/.../DaBR.ckpt
```

**Features:**
- Prevents multiple instances from running simultaneously
- Logs output to `nohup.out`
- Saves process ID to `.pid` file for tracking
- Reports the PID when started

### `stop.sh` - Stop the background process
Gracefully stops the running background process. Will force kill if graceful shutdown doesn't work.

**Usage:**
```bash
nohup/stop.sh
```

**Features:**
- Graceful shutdown (SIGTERM) first
- Waits 2 seconds for process to exit
- Force kills (SIGKILL) if still running
- Cleans up `.pid` file

### `check.sh` - Check process status
Displays the status of the background process and recent output logs.

**Usage:**
```bash
nohup/check.sh [N]    # Show status and last N lines (default 20)
```

**Examples:**
```bash
# Show status and last 20 lines of output
nohup/check.sh

# Show status and last 50 lines of output
nohup/check.sh 50

# Show entire output
nohup/check.sh $(wc -l < nohup/nohup.out)
```

**Features:**
- Shows process status (RUNNING or NOT RUNNING)
- Displays PID and process info
- Shows tail of output log
- Auto-cleans `.pid` file if process no longer exists

## Workflow Example

```bash
# Start training in background
cd /home/bn/GBPU/DaBR
nohup/nohup/run.sh --dataset WN18RR --config ./config/WN18RR_test.json --mode train

# Check progress after a minute
nohup/nohup/check.sh

# Check with more output
nohup/nohup/check.sh 50

# Stop if needed
nohup/nohup/stop.sh
```

## Output Files

- `nohup.out` - Complete stdout/stderr output from the running process
- `.pid` - Process ID file (auto-created, auto-cleaned)
- `../logs/` - Model checkpoints and results.json as usual

## Notes

- All scripts must be run from the nohup directory or with full paths
- Output is redirected to `nohup.out`, not lost when terminal closes
- Process continues running even after terminal session ends
- Use `nohup/check.sh` frequently to monitor progress
- Process will output to `nohup.out` in real-time
