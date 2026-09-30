# METAGROSS deck design tokens

These tokens apply to the 6-slide SIH26126 deck (Canva, 1920×1080, **white background**), to every chart made by
`metagross/eval/plot_style.py`, and to the diagrams in `deck_assets/diagrams/` and the Figma file.
The look is clean, modern and professional: white surfaces, hairline borders, one deep-blue accent, and no gradients, clip-art, neon or paper textures.

Code source of truth: `metagross/eval/plot_style.py` (`TOKENS`, `HONESTY_COLORS`, `MODE_COLORS`, `CELL_HEX`).
The cell-state colours come from `metagross.contracts.messages.CELL_COLORS` and are never copied by hand.

## Neutrals and brand

| Token | Hex | Use |
|---|---|---|
| `bg` | `#FFFFFF` | Slide, chart and diagram background |
| `surface` | `#F8FAFC` | Subtle panels and cards, output chips |
| `border` | `#E2E8F0` | Card borders, hairlines, axis spines |
| `grid` | `#EEF2F6` | Chart grid (lighter than the border so it recedes) |
| `text` | `#0F172A` | Primary text |
| `text_secondary` | `#475569` | Axis labels, captions, detail lines |
| `text_muted` | `#94A3B8` | Eyebrow labels (ALL CAPS), footnotes |
| `accent` | `#1D4ED8` | Brand accent (deep blue): key stage, links, governor |
| `navy` | `#0B1F4B` | Heading accents, stage badges, firewall outline |
| `vo_blue` | `#2563EB` | Localisation / VO |
| `path` | `#0F172A` | Planned path |

## Semantic key: BEV cell states (from `CELL_COLORS`)

| State | RGB | Hex | Label on slides |
|---|---|---|---|
| GROUND | (34, 160, 90) | `#22A05A` | Seen ground |
| UNSEEN | (203, 208, 214) | `#CBD0D6` | Unseen |
| DITCH_CANDIDATE | (200, 40, 160) | `#C828A0` | Ditch / missing ground |
| CREST_SHADOW | (240, 160, 30) | `#F0A01E` | Crest shadow |
| POSITIVE / DEPRESSION | (220, 50, 47) | `#DC322F` | Lethal |
| WATER | (20, 140, 190) | `#148CBE` | Water / mud |
| OCCLUDED | (150, 156, 164) | `#969CA4` | Occluded |
| DYNAMIC | (250, 90, 20) | `#FA5A14` | Dynamic |

The same key is used by the operator console, the video and the deck. Reusable strips are `deck_assets/diagrams/legend_cell_states.png` (all 8 states) and `legend_cell_states_core6.png` (6 states).

## Health modes (supervisor `DriveMode`)

These are status colours. Always show them with the mode name, never as colour alone.

| Mode | Hex | Entry rule (from `autonomy/safety/supervisor.py`) |
|---|---|---|
| NOMINAL | `#15803D` | q > 0.7 |
| CAUTION | `#CA8A04` | 0.4 < q ≤ 0.7 |
| DEGRADED | `#EA580C` | 0.2 < q ≤ 0.4 |
| STOP_AND_LOOK | `#DC2626` | no progress for 3 s |
| SAFE_STOP | `#7F1D1D` | q < 0.2 for 3 s, or E-stop (latched) |

## Honesty chips (claims ledger labels)

Every number shown on the deck carries one of these chips as a small outlined pill: white fill, 2 px border and bold text, both in the chip colour.
PNGs with a transparent background are in `deck_assets/diagrams/chips/`.

| Label | Hex |
|---|---|
| Tested | `#0F766E` |
| Simulated | `#1D4ED8` |
| Estimated | `#B45309` |
| Proposed | `#6B7280` |
| Literature | `#7C3AED` |

## Typography

* Sans: **Inter** (Regular 400 and Bold 700; Semi Bold is available in Figma). Fallbacks are Segoe UI, then Arial.
* Numbers, formulas and code identifiers: a monospace face, **JetBrains Mono** (Figma). Fallbacks are Cascadia Mono, then Consolas.
* Diagram scale at 1× (slide pixels): stage titles 19 px bold; body 15 px; detail 13 to 14 px; eyebrow labels 11 to 12 px bold with +1 px tracking.
* Chart scale: the PNG is exported at 2× and 300 dpi. Body text is 8.5 pt, which is about 17 slide px; titles are 10.5 pt, about 22 slide px.

## Chart rules (`plot_style.py`)

* White background, no top or right spines, light grid, and left-aligned bold titles.
* The honesty chip sits in the top-right corner of every chart. Formulas and assumptions go in a muted footnote (`add_footnote`).
* Sequential data uses one hue (near-white, then brand blue, then navy). Categorical series follow the semantic key above.
* Named sizes in slide px: `card` 900×520, `card_tall` 900×700, `square` 700×700, `wide` 1600×600, `full` 1760×860. `save_fig` writes `<name>.png` at 2× and 300 dpi, plus `<name>.svg` with text kept as text.

## Layout

* Corner radius: 14 px for cards and panels, 10 px for inner boxes, fully rounded for pills.
* Borders: 1.5 px `border`. The key or highlighted element gets a 2 px `accent` border.
* Spacing grid of 4 px. Card padding is 20 px and the gap between cards is 10 to 24 px.
