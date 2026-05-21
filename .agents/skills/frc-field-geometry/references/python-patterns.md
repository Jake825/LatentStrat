# Python Patterns for FRC Field Geometry

Use these patterns when writing analysis notebooks, plotting scripts, scouting visualizations, or path/pose helpers.

## Field Constants

Keep field constants grouped and in meters. Do not scatter magic numbers through analysis code.

```python
from dataclasses import dataclass
from math import pi, remainder


def in2m(inches: float) -> float:
    return inches * 0.0254


@dataclass(frozen=True)
class Field2026Rebuilt:
    """Approximate 2026 REBUILT field constants in meters.

    Verify exact values against official WPILib/FIRST sources before precision use.
    """

    length_m: float = 16.54
    width_m: float = 8.07
    fuel_diameter_m: float = in2m(5.91)
    hub_width_m: float = in2m(47.0)
    tower_low_rung_z_m: float = in2m(27.0)
    tower_mid_rung_z_m: float = in2m(45.0)
    tower_high_rung_z_m: float = in2m(63.0)

    @property
    def center(self) -> tuple[float, float]:
        return self.length_m / 2.0, self.width_m / 2.0


FIELD_2026 = Field2026Rebuilt()
```

## Pose Helpers

```python
@dataclass(frozen=True)
class Pose2d:
    x_m: float
    y_m: float
    theta_rad: float


def normalize_angle(theta_rad: float) -> float:
    return remainder(theta_rad, 2.0 * pi)


def to_red_alliance_pose(pose: Pose2d, field: Field2026Rebuilt = FIELD_2026) -> Pose2d:
    return Pose2d(
        x_m=field.length_m - pose.x_m,
        y_m=field.width_m - pose.y_m,
        theta_rad=normalize_angle(pose.theta_rad + pi),
    )
```

## Plotting and Heatmaps

Always use equal aspect ratio for field plots.

```python
import matplotlib.patches as patches
import matplotlib.pyplot as plt


def draw_field(ax, field: Field2026Rebuilt = FIELD_2026) -> None:
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(0.0, field.length_m)
    ax.set_ylim(0.0, field.width_m)
    ax.set_xlabel("X field coordinate, m")
    ax.set_ylabel("Y field coordinate, m")

    ax.add_patch(
        patches.Rectangle(
            (0.0, 0.0),
            field.length_m,
            field.width_m,
            linewidth=2,
            edgecolor="black",
            facecolor="none",
        )
    )

    center_x, center_y = field.center
    ax.axvline(center_x, color="gray", linestyle="--", linewidth=1)
    ax.axhline(center_y, color="gray", linestyle=":", linewidth=1)


def plot_shot_heatmap(df, x_col: str = "x_m", y_col: str = "y_m") -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    draw_field(ax)
    ax.hexbin(df[x_col], df[y_col], gridsize=30, cmap="YlOrRd", alpha=0.7)
    plt.show()
```

## Zone Helpers

Use named helper functions for zones. Verify boundaries against official sources before using them for rule-sensitive logic.

```python
def is_inside_rect(
    x_m: float,
    y_m: float,
    min_x_m: float,
    max_x_m: float,
    min_y_m: float,
    max_y_m: float,
) -> bool:
    return min_x_m <= x_m <= max_x_m and min_y_m <= y_m <= max_y_m


def is_blue_alliance_zone_x(
    x_m: float,
    boundary_x_m: float,
) -> bool:
    return x_m <= boundary_x_m
```

## Data Hygiene

- Name raw source columns by unit, such as `shot_x_in`, `pose_y_m`, or `heading_deg`.
- Convert once into canonical `*_m` and `*_rad` fields.
- Store the coordinate frame in metadata or column names when possible.
- For red-alliance analysis, document whether data is mirrored into alliance-relative coordinates or kept in always-blue field coordinates.

## Spatial ML Feature Representation

When turning spatial scouting data into LatentStrat features, choose a representation that matches the model task and timing boundary.

- Use canonical Cartesian columns such as `start_x_m`, `start_y_m`, and `heading_rad` when the model should learn continuous geometry.
- Use named zone categories or one-hot zone columns when the scouting source only supports coarse locations.
- Add distance and angle-to-target features when a game element is the meaningful reference point, such as `distance_to_hub_m` or `bearing_to_tower_rad`.
- Consider alliance-relative mirrored coordinates for comparing red and blue behavior from each alliance's perspective.
- Preserve raw source columns separately from normalized model features when debugging imports.

Do not mix field-relative and alliance-relative coordinates in the same feature column. Use `$latentstrat-feature-pipeline` for Parquet/tensor boundaries and `$frc-time-aware-analysis` before using spatial observations as pre-match model inputs.
