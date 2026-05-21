# 2026 REBUILT Field Geometry

Use this reference for 2026 REBUILT spatial orientation. Verify exact dimensions and tag poses against the official game manual, field dimension drawings, Team Updates, and WPILib AprilTag layouts before precision use.

## Units

- Convert inches to meters with `inches * 0.0254`.
- Use meters and radians internally.
- Treat inch values from the manual as source measurements, not as mixed-unit code constants.

## Core Field Dimensions

Use the official WPILib AprilTag field layout for exact field length and width. The draft constants use approximate dimensions:

- Field length: about `16.54 m`.
- Field width: about `8.07 m`.
- Field center: `(field_length / 2.0, field_width / 2.0)`.

WPILib exposes 2026 REBUILT layouts for both welded and AndyMark fields:

- `k2026RebuiltWelded`
- `k2026RebuiltAndymark`

Choose the layout that matches the field being analyzed.

## Important X-Axis References

In the always-blue field frame:

- Center line: `field_length / 2.0`.
- Blue alliance zone / starting line: verify from the manual or field drawings.
- Red alliance zone / starting line: symmetric counterpart if the official layout confirms symmetry.
- Neutral zone near/far lines: verify from official drawings before precision plotting.
- Hub centers: derive from official tag poses or field drawings rather than hard-coding approximate offsets.

## Important Y-Axis References

In the always-blue field frame:

- Field centerline: `field_width / 2.0`.
- Right side is near `Y = 0` from the blue alliance perspective.
- Left side is near `Y = field_width`.
- Bump and trench openings should be derived from official drawings or tag-aligned constants for precision work.

## Element Dimensions From Draft Reference

These values are useful orientation anchors. Verify against official sources before precision pathing or inspection-sensitive analysis.

- Fuel diameter: `5.91 in` (`0.150 m`).
- Hub width: `47.0 in` (`1.194 m`).
- Hub top height: `72.0 in` (`1.829 m`), including catcher.
- Hub inner opening width: `41.7 in` (`1.059 m`).
- Hub inner opening height: `56.5 in` (`1.435 m`).
- Bump width along Y: `73.0 in` (`1.854 m`).
- Bump depth along X: `44.4 in` (`1.128 m`).
- Bump height: `6.513 in` (`0.165 m`).
- Trench width along Y: `65.65 in` (`1.668 m`).
- Trench depth along X: `47.0 in` (`1.194 m`).
- Trench height: `40.25 in` (`1.022 m`).
- Tower width: `49.25 in` (`1.251 m`).
- Tower depth: `45.0 in` (`1.143 m`).
- Tower height: `78.25 in` (`1.988 m`).
- Tower rung heights: low `27.0 in`, mid `45.0 in`, high `63.0 in`.
- Depot width: `42.0 in` (`1.067 m`).
- Depot depth: `27.0 in` (`0.686 m`).
- Outpost width: `31.8 in` (`0.808 m`).
- Fuel pool width along Y: `181.9 in` (`4.620 m`).
- Fuel pool depth along X: `71.9 in` (`1.826 m`).

## AprilTag Layout Notes

- Tag family: 36h11.
- Tag printed size: `6.5 in` (`0.165 m`).
- Draft notes indicate 32 tags for the 2026 layout. Confirm count and poses from the selected WPILib field layout.
- Use official layout objects or JSON for exact tag center poses.
- Do not copy tag-to-element mappings from memory. Verify tag IDs against the official layout or field drawings before coding scoring, localization, or alignment targets.

## Source Priority

1. Official FIRST field drawings and Team Updates.
2. Official game manual.
3. WPILib official AprilTag layout for the chosen field variant.
4. Team/community constants as implementation examples only.
