# Source Links

Use these sources before relying on exact coordinates, tag poses, or rule-sensitive geometry.

## Official and Primary Sources

- WPILib Coordinate System: https://docs.wpilib.org/en/stable/docs/software/basic-programming/coordinate-system.html
- WPILib AprilTagFieldLayout API: https://github.wpilib.org/allwpilib/docs/release/java/edu/wpi/first/apriltag/AprilTagFieldLayout.html
- WPILib AprilTagFields API: https://github.wpilib.org/allwpilib/docs/release/java/edu/wpi/first/apriltag/AprilTagFields.html
- FIRST Season Materials: https://www.firstinspires.org/resources/library/frc/season-materials
- FIRST 2026 Game Manual PDF: `.agents/skills/frc-game-manual/assets/manuals/2026GameManual-TU22.pdf`
- FIRST 2026 Combined Team Updates PDF: `.agents/skills/frc-game-manual/assets/manuals/REBUILT_TeamUpdate-Combined.pdf`

## Verification Notes

- WPILib documents NWU axes and always-blue-origin field-coordinate workflows.
- WPILib `AprilTagFieldLayout` reads official layouts, exposes field length and width in meters, and supports origin transforms.
- WPILib `AprilTagFields` exposes both `k2026RebuiltWelded` and `k2026RebuiltAndymark`.
- FIRST field drawings and Team Updates should be checked before precision geometry, inspection-sensitive dimensions, or tag placement claims.
