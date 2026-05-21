# Statbotics Codebase Structure

Use this reference when asked to understand or adapt the Statbotics platform itself. This is orientation for the external Statbotics repository, not LatentStrat's local source tree. Verify the current upstream repository before making direct Statbotics code changes.

## High-Level Architecture

Statbotics is broadly organized around a Python backend and a web frontend.

```text
statbotics/
|-- backend/
|   |-- src/
|   |   |-- api/          # FastAPI routes and endpoint handlers
|   |   |-- data/
|   |   |   |-- epa/      # EPA aggregation, calculation, math, and metrics
|   |   |   `-- tba/      # TBA ingestion and raw FRC data loading
|   |   |-- db/
|   |   |   |-- models/   # Database table models
|   |   |   |-- read/     # Read queries
|   |   |   `-- write/    # Write/update operations
|   |   |-- models/       # Pydantic schemas and typed API models
|   |   `-- site/         # Site helper logic
`-- frontend/
    `-- src/
        |-- api/          # Frontend API wrappers
        |-- components/   # Reusable UI components
        |-- pages/        # Next.js routes
        `-- pagesContent/ # Page-specific content and layouts
```

## Internal Data Flow

1. TBA ingestion collects schedules, teams, matches, results, and score breakdowns.
2. EPA calculation code processes matches in time order.
3. Pre-match predictions are produced from the current team ratings.
4. Actual results update team ratings and component EPA values.
5. Updated records are written to the database.
6. API routes expose teams, events, matches, team-years, team-events, and team-matches to public consumers.

## Development Notes

- EPA math is sensitive to ordering, mean reversion, and update formulas. Preserve chronological processing.
- Backend changes should respect typed schemas and database read/write boundaries.
- If adapting logic into LatentStrat, prefer small, explicit extraction code over copying large chunks of the Statbotics platform.
- If investigating upstream Statbotics behavior, inspect the current source and API response shape before assuming paths from older examples still exist.
