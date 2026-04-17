# Released Timeline Data

This directory contains the public timeline portion of the SteeM release.

## Scope

- Domains: `research` and `tutoring`
- Cases per domain: 194
- Total released cases: 388
- Released files per case: `events.json`, `stats.json`
- Excluded files: `interactions.json`

## Directory Layout

```text
data/timeline/
  research/
    0/
      events.json
      stats.json
    1/
      events.json
      stats.json
    ...
  tutoring/
    0/
      events.json
      stats.json
    1/
      events.json
      stats.json
    ...
```

## File Semantics

- `events.json`

  - A structured project timeline.
  - Each item corresponds to one event in the trajectory.
  - Common fields include:
    - `event_id`
    - `time_index`
    - `domain`
    - `topic`
    - `subject`
    - `event_type`
    - `description`
    - `required_artifacts`
    - `generated_artifacts`
    - `reason`
- `stats.json`

  - Lightweight generation statistics for the corresponding case.
  - Typically includes token counts and `num_steps`.
