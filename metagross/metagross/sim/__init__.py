"""Simulated world: the ground-truth owner (never imported by ``metagross.autonomy``).

Modules
-------
scenario   deterministic scenario generator (schema ``metagross.scenario/1``), IO, sha256, seeds
miniworld  hand-built mini scenarios for unit tests
terrain    heightmap + material raster with vectorised sampling
geometry   frames / Euler conventions (REP-103), raster helpers
objects    static + dynamic scene objects: analytic ray-cast primitives and collision footprints
hazards    GT hazard rasters (object / ditch / water / slope / dynamic-rest masks)
gridplan   grid A* with corridor clearance (solvability, SPL reference path)
vehicle    kinematic skid-steer with lag, slip, chi, terrain following; latency command queue
sensors    wheel encoders, gyro, Tier-0 synthetic depth sensor (+ GT depth / semantics)
referee    collision / ditch / water / tip-over / stuck / timeout / success; false-stop analysis
world      World: physics stepping, dynamics, lighting, SensorFrame + MissionSpec production, GT log
runner     closed-loop episode over the ipc protocol with a spawned autonomy process
batch      resumable multi-process batches + summary CSV
bench      reproducible performance measurements
"""
