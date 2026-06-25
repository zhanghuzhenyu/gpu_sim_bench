# GPUSimBench Experiment Repository

This repository contains the paper PDF and the corresponding experiment scripts used to benchmark multiple GPU-accelerated simulators for embodied AI workloads.

The benchmark is organized around three experiment groups:

- `exp1`: free-fall cube-array throughput benchmark
- `exp2`: inclined collision distribution benchmark
- `exp3`: Franka random-action throughput benchmark

The current repository layout is simulator-centric: each simulator has its own folder, and each folder contains one or more of the three benchmark groups.

## Paper


The paper evaluates the trade-offs among:

- scalability
- physical consistency
- determinism

across the following simulators:

- Genesis
- IsaacLab
- MJX
- Madrona
- ManiSkill
- MuJoCo Playground
- MuJoCo Warp

## Repository Structure

```text
.
├── benchmark_defaults.py
├── Genesis/
├── IsaacLab/
├── MJX/
├── Madrona/
├── ManiSkill/
├── MujocoPlayground/
└── MujocoWarp/
```

Each simulator directory follows this convention when available:

- `exp1/`: free-fall cube-array benchmark
- `exp2/`: inclined collision / distribution export
- `exp3/`: Franka benchmark

## Unified Benchmark Defaults

To keep experiments aligned across simulators, shared benchmark defaults are centralized in [benchmark_defaults.py](./benchmark_defaults.py).

### `exp1`: Free-Fall Cube Array

- number of cubes per dimension: `3`
- cube edge length: `1.0`
- cube mass: `1.0`
- spacing: `2.0`
- initial center height: `10.0`
- simulation timestep: `0.01`
- environment spacing: `20.0`
- warmup steps: `30`
- timed test steps: `300`
- memory sample interval: `15`

### `exp2`: Inclined Collision

- timestep: `0.01`
- default parallel environments: `16`
- sample time: `5.0 s`
- default steps: `500`

The physical parameters for the slope, ball, cubes, friction, restitution, and array placement are stored in the per-simulator `slope_cube_config.yaml` files.

### `exp3`: Franka Benchmark

- timestep: `0.01`
- environment spacing: `4.0`
- warmup steps: `30`
- timed test steps: `300`
- memory sample interval: `15`

## Path Handling

Absolute paths have been removed from the repository scripts.

Current scripts now resolve resources using:

- the current file directory
- shared defaults from `benchmark_defaults.py`
- repository-local XML / YAML files where available

This makes the repository portable across machines, as long as the required simulator packages are installed in the local environment.

## Main Entry Scripts

### Genesis

- `exp1`: [Genesis/exp1/Cube_single_run_genesis.py](./Genesis/exp1/Cube_single_run_genesis.py)
- `exp2`: [Genesis/exp2/slope_ball_collision.py](./Genesis/exp2/slope_ball_collision.py)
- `exp3`: [Genesis/exp3/benchmark_genesis.py](./Genesis/exp3/benchmark_genesis.py)

### IsaacLab

- `exp1`: [IsaacLab/exp1/Cube_single_run_isaaclab.py](./IsaacLab/exp1/Cube_single_run_isaaclab.py)
- `exp3`: [IsaacLab/exp3/isaaclab_franka_random_benchmark.py](./IsaacLab/exp3/isaaclab_franka_random_benchmark.py)

### MJX

- `exp1`: [MJX/exp1/Cube_single_run_mjx.py](./MJX/exp1/Cube_single_run_mjx.py)
- `exp2`: [MJX/exp2/benchmark_slope_ball_mjx.py](./MJX/exp2/benchmark_slope_ball_mjx.py)

### Madrona

- `exp1`: [Madrona/exp1/Cube_single_run_madrona.py](./Madrona/exp1/Cube_single_run_madrona.py)
- `exp2`: [Madrona/exp2/record_full_distribution.py](./Madrona/exp2/record_full_distribution.py)

### ManiSkill

- `exp1`: [ManiSkill/exp1/Cube_single_run_maniskill.py](./ManiSkill/exp1/Cube_single_run_maniskill.py)
- `exp2`: [ManiSkill/exp2/export_distribution.py](./ManiSkill/exp2/export_distribution.py)
- `exp3`: [ManiSkill/exp3/benchmark_maniskill_unified.py](./ManiSkill/exp3/benchmark_maniskill_unified.py)
- alternative `exp3` benchmark: [ManiSkill/exp3/maniskill_franka_random_benchmark.py](./ManiSkill/exp3/maniskill_franka_random_benchmark.py)

### MuJoCo Playground

- `exp1`: [MujocoPlayground/exp1/Cube_single_run_playground.py](./MujocoPlayground/exp1/Cube_single_run_playground.py)
- `exp2`: [MujocoPlayground/exp2/export_distribution_playground.py](./MujocoPlayground/exp2/export_distribution_playground.py)

### MuJoCo Warp

- `exp1`: [MujocoWarp/exp1/Cube_single_run_mujocowarp.py](./MujocoWarp/exp1/Cube_single_run_mujocowarp.py)
- `exp2`: [MujocoWarp/exp2/export_distribution.py](./MujocoWarp/exp2/export_distribution.py)
- `exp3`: [MujocoWarp/exp3/benchmark_mujocowarp.py](./MujocoWarp/exp3/benchmark_mujocowarp.py)

## Typical Outputs

### Throughput Benchmarks (`exp1`, `exp3`)

Most performance scripts print a final line in the form:

```text
BENCHMARK_RESULT:{...}
```

The JSON payload typically includes:

- `num_envs`
- `fps`
- `mem_baseline`
- `mem_post_init`
- `mem_post_warmup`
- `mem_peak`
- `mem_final`

### Distribution Benchmarks (`exp2`)

Most `exp2` scripts export JSON files containing:

- simulator metadata
- timestep and sample time
- per-environment cube positions
- quaternions
- linear velocity
- angular velocity
- centroid
- max spread

## Example Commands

These examples assume that the corresponding simulator stack is already installed and importable in the current Python environment.

### Genesis

```bash
python3 Genesis/exp1/Cube_single_run_genesis.py --num_envs 64
python3 Genesis/exp2/slope_ball_collision.py --no-viewer --output-json genesis_exp2.json
python3 Genesis/exp3/benchmark_genesis.py --num_envs 64
```

### MJX

```bash
python3 MJX/exp1/Cube_single_run_mjx.py --num_envs 64
python3 MJX/exp2/benchmark_slope_ball_mjx.py --output-json mjx_exp2.json
```

### MuJoCo Warp

```bash
python3 MujocoWarp/exp1/Cube_single_run_mujocowarp.py --num_envs 64
python3 MujocoWarp/exp2/export_distribution.py MujocoWarp/exp2/slope_cube.xml --output_json mujocowarp_exp2.json
python3 MujocoWarp/exp3/benchmark_mujocowarp.py --num_envs 64
```

## Known External Dependencies

This repository standardizes paths and benchmark defaults, but some scripts still depend on simulator-specific packages or local extensions that are not included in this repository.

### Missing repository-local modules

- `ManiSkill/exp2/export_distribution.py`
  - depends on `benchmark_zhang_slope.slope_ball_impact_visualize`
- `ManiSkill/exp3/benchmark_maniskill_unified.py`
  - depends on `roboverse_maniskill.franka_unified`
- Madrona scripts depend on local compiled bindings such as:
  - `falling_cubes`
  - `slope_scene`

These are not path bugs; they are genuine external code dependencies.

## Notes

- The repository now avoids hard-coded machine-specific absolute paths.
- Default benchmark parameters are intentionally aligned across simulators as much as the current script implementations allow.
- Some simulator backends still differ in internal solver behavior, control interfaces, or scene construction details. Parameter alignment does not guarantee numerically identical trajectories.

## Recommended Next Steps

- add environment setup instructions for each simulator stack
- add a result collection script to aggregate `BENCHMARK_RESULT` outputs
- replace the remaining external ManiSkill-only imports with repository-local implementations
