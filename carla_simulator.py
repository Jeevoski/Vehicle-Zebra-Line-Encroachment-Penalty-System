"""Generate a repeatable traffic video from a local CARLA server.

The simulator is optional. It records a CARLA camera feed, adds a deterministic
zebra crossing and traffic-light panel to the image, and writes matching ROI
coordinates so the recorded video can be replayed by StripeGuard.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from queue import Empty, Queue

import cv2
import numpy as np

WIDTH = 1280
HEIGHT = 720
FPS = 20
ZEBRA_ROI = [[350, 515], [930, 515], [1010, 650], [270, 650]]
SIGNAL_ROI = [1060, 55, 1165, 180]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record a repeatable CARLA traffic scenario")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--duration", type=float, default=60.0, help="Recording duration in seconds")
    parser.add_argument("--output", default="data/carla_traffic.mp4")
    parser.add_argument("--roi-output", default="data/carla_rois.json")
    parser.add_argument("--vehicles", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def draw_simulation_overlay(image: np.ndarray, signal: str) -> np.ndarray:
    """Draw stable test geometry on top of the CARLA camera image."""
    output = image.copy()
    polygon = np.asarray(ZEBRA_ROI, dtype=np.int32)
    left, top = polygon[0]
    right, bottom = polygon[2]
    stripe_count = 8
    for index in range(stripe_count):
        x1 = left + index * (right - left) // stripe_count
        x2 = left + (index + 1) * (right - left) // stripe_count - 8
        points = np.array([[x1, top], [x2, top], [x2 + 25, bottom], [x1 + 25, bottom]], dtype=np.int32)
        cv2.fillConvexPoly(output, points, (235, 235, 235))

    x1, y1, x2, y2 = SIGNAL_ROI
    cv2.rectangle(output, (x1, y1), (x2, y2), (25, 25, 25), -1)
    bulb_x = (x1 + x2) // 2
    for index, color in enumerate(((0, 0, 255), (0, 165, 255), (0, 200, 0))):
        bulb_color = color if (signal == "RED" and index == 0) or (signal == "GREEN" and index == 2) else (55, 55, 55)
        cv2.circle(output, (bulb_x, y1 + 20 + index * 35), 13, bulb_color, -1)
    cv2.putText(output, signal, (x1 - 5, y2 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    return output


def make_camera(world: carla.World, vehicle: carla.Actor, image_queue: Queue) -> carla.Actor:
    blueprint = world.get_blueprint_library().find("sensor.camera.rgb")
    blueprint.set_attribute("image_size_x", str(WIDTH))
    blueprint.set_attribute("image_size_y", str(HEIGHT))
    blueprint.set_attribute("fov", "90")
    transform = carla.Transform(carla.Location(x=1.5, z=2.4), carla.Rotation(pitch=-5.0))
    camera = world.spawn_actor(blueprint, transform, attach_to=vehicle)
    camera.listen(image_queue.put)
    return camera


def spawn_vehicles(world: carla.World, count: int, seed: int) -> list[carla.Actor]:
    random.seed(seed)
    library = world.get_blueprint_library().filter("vehicle.*")
    blueprints = [blueprint for blueprint in library if blueprint.has_attribute("role_name")]
    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)
    vehicles: list[carla.Actor] = []
    batch = []
    for blueprint, transform in zip(random.choices(blueprints, k=count), spawn_points):
        blueprint.set_attribute("role_name", "autopilot")
        batch.append(carla.command.SpawnActor(blueprint, transform).then(carla.command.SetAutopilot(carla.command.FutureActor, True)))
    for response in world.apply_batch_sync(batch, True):
        if not response.error:
            vehicles.append(world.get_actor(response.actor_id))
    return vehicles


def write_roi_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"reference_frame_size": [WIDTH, HEIGHT], "zebra_roi": ZEBRA_ROI, "signal_rois": [SIGNAL_ROI]}, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    global carla
    args = parse_args()
    try:
        import carla
    except ImportError as exc:  # pragma: no cover - depends on optional CARLA install
        raise SystemExit(
            "CARLA Python API is not installed. Install the version matching your "
            "CARLA server, then run this script again."
        ) from exc

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_roi_file(Path(args.roi_output))

    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    world = client.get_world()
    original_settings = world.get_settings()
    actors: list[carla.Actor] = []
    camera = None
    image_queue: Queue = Queue()
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT))
    if not writer.isOpened():
        raise RuntimeError(f"Unable to open video output: {output_path}")

    try:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / FPS
        world.apply_settings(settings)
        vehicles = spawn_vehicles(world, args.vehicles, args.seed)
        actors.extend(vehicles)
        if not vehicles:
            raise RuntimeError("CARLA did not spawn any vehicles")
        camera = make_camera(world, vehicles[0], image_queue)
        actors.append(camera)

        total_frames = int(args.duration * FPS)
        for frame_index in range(total_frames):
            world.tick()
            try:
                image = image_queue.get(timeout=2.0)
            except Empty as exc:
                raise RuntimeError("Timed out waiting for a CARLA camera frame") from exc
            frame = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))[:, :, :3]
            signal = "RED" if (frame_index // (FPS * 12)) % 2 == 0 else "GREEN"
            writer.write(draw_simulation_overlay(frame, signal))
    finally:
        writer.release()
        for actor in reversed(actors):
            if actor is not None:
                actor.destroy()
        world.apply_settings(original_settings)

    print(f"Recorded {output_path}")
    print(f"ROI metadata: {args.roi_output}")
    print("Replay with: python main.py --source " + str(output_path) + " --signal-mode auto --headless --cpu-ocr --output-video violations/carla_annotated.mp4")


if __name__ == "__main__":
    main()
