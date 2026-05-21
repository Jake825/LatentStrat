# FRC Coordinate Systems

Use this reference when working with robot poses, autonomous paths, scouting coordinates, heatmaps, or AprilTag layouts.

## WPILib NWU Frame

WPILib uses the NWU convention for most robot and field geometry:

- `+X`: forward.
- `+Y`: left.
- `+Z`: up.
- Positive rotation is counter-clockwise when viewed along the positive axis.

For field layouts, use the standard always-blue field origin unless a task explicitly asks for alliance-relative coordinates.

## Always-Blue Field Frame

The common FRC field frame is:

- Origin `(0, 0, 0)`: bottom-right corner of the blue alliance wall when looking out from the blue alliance driver stations.
- `+X`: downfield toward the red alliance wall.
- `+Y`: left across the field toward the scoring table.
- `+Z`: upward from the carpet.
- Heading `0 rad`: facing downfield toward the red alliance wall.
- Heading `pi / 2 rad`: facing toward `+Y`.
- Heading `pi rad`: facing toward the blue alliance wall.
- Heading `-pi / 2 rad`: facing toward `-Y`.

Official AprilTag layouts provide tag poses in meters using this convention unless an origin transform is applied.

## Field-Relative vs Alliance-Relative

Field-relative coordinates keep the origin fixed at the blue alliance wall for both alliances. This is preferred for:

- AprilTag pose estimation.
- Shared scouting heatmaps.
- Field plots.
- Joining data from multiple alliances.

Alliance-relative coordinates put each alliance's wall at its own origin. This can be useful for strategy summaries or mirrored dashboards, but must be labeled clearly.

## Transform Patterns

For a 180-degree rotated alliance-relative view:

```python
import math


def normalize_angle(theta_rad: float) -> float:
    return math.remainder(theta_rad, 2.0 * math.pi)


def blue_field_to_red_alliance_pose(
    x_blue_m: float,
    y_blue_m: float,
    theta_blue_rad: float,
    field_length_m: float,
    field_width_m: float,
) -> tuple[float, float, float]:
    x_red_m = field_length_m - x_blue_m
    y_red_m = field_width_m - y_blue_m
    theta_red_rad = normalize_angle(theta_blue_rad + math.pi)
    return x_red_m, y_red_m, theta_red_rad


def blue_field_to_red_alliance_vector(
    vx_blue_mps: float,
    vy_blue_mps: float,
) -> tuple[float, float]:
    return -vx_blue_mps, -vy_blue_mps
```

## Practical Guardrails

- Do not mix inches and meters in one helper. Convert at the boundary.
- Do not mix degrees and radians in internal geometry code. Convert UI degrees to radians before calculations.
- Do not assume every game has identical blue/red symmetry. Check the season's field layout and WPILib documentation.
- If a dashboard expects always-blue coordinates, convert alliance-relative poses back before display.
