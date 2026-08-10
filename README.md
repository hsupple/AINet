# AINet

Local voice AI over ESP32 hardware, backed by a strict rolling personal database.

## System

Talk into a mic on an ESP32. The ESP32 sends the audio signal to a **Windows PC**. The PC runs a local Ollama model, then returns audio for the ESP32 to play through a speaker.

```
Mic → ESP32 → Windows PC (Ollama + AINet tools) → ESP32 → Speaker
```

**macOS edition:** see [`mac/`](mac/) (self-contained `ainet/` + `ollama/` + `db/`, zsh scripts). Root tree stays Windows-oriented.

Database tool paths always use forward slashes (example: `Hayden/Read.json`), including on Windows.
The model does not operate from free-floating chat memory alone. Every useful turn goes through an organized, efficiency-bound database that stays current as life changes.

## Database

This repo is primarily the database layout and rules for that system. The store must be:

- **Rolling** — updated continuously; stale state is pruned or archived into History
- **Organized** — every domain has a clear COP (context of purpose) and fixed file roles
- **Efficient** — strict read/write rules so the AI loads only what it needs

### Top-level controls

| File | Role |
|------|------|
| `Rules.txt` | Read-only by the AI. Hard constraints for behavior and DB access. |
| `Calendar.json` | Mutable by code only. Schedule source of truth. |
| `Folderrules.json` | How folders may be created, named, and used. |
| `Changelog.json` | Append-only record of structural and content changes. |

### Tree

```
├── Rules.txt                 ** Read only by AI **
├── Calendar.json             ## Mutable by code only ##
├── Folderrules.json
├── Changelog.json
│
├── Hayden/
│   ├── Profile.json
│   ├── Read.json
│   ├── Preferences/
│   │   └── [Music, tastes, recipes, …]
│   ├── Values.json
│   ├── Habits/
│   │   └── [Hobbies, interests, schedule, …]
│   ├── Desires.json
│   ├── Relationships.json
│   ├── Memories/
│   │   └── [Age, experiences, …]
│   └── History/
│
├── School/
│   ├── Profile.json
│   ├── Plan
│   ├── Courses/
│   │   └── [Course COP
│   │       ├── Profile
│   │       ├── Read
│   │       ├── Plan
│   │       ├── History/
│   │       └── Files/]
│   └── History/
│
├── Work/
│   ├── Profile
│   ├── Read
│   ├── Plan
│   ├── Projects/
│   │   └── [Project COP
│   │       ├── Profile
│   │       ├── Read
│   │       ├── Plan
│   │       ├── Decisions
│   │       ├── Open Questions
│   │       ├── History/
│   │       ├── Files/
│   │       └── [AI-created subfolders as needed]]
│   └── History/
│
└── Household/
    ├── Profile
    ├── Read
    ├── Wants
    ├── Pantry/
    ├── Maintenance/
    └── History/
```

### Domain notes

- **Hayden/** — personal identity, preferences, values, habits, relationships, and memories.
- **School/** — plan plus per-course COPs (profile, read surface, plan, history, files).
- **Work/** — plan plus per-project COPs, including decisions and open questions; AI may add subfolders when needed under project rules.
- **Household/** — home ops: wants, pantry, maintenance, and history.

`Profile` / `Read` / `Plan` (and similar) are the standard surfaces inside a COP: who/what it is, what the AI should load first, and current intent. `History/` holds rolled-off material so active folders stay small and query-efficient.
