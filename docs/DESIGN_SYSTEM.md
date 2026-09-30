# Design System: [Product name]

<!--
Every visual decision lives here so the agent never invents one.
Fill the values once. When you want the product to look different,
change this file, not the prompt. The values below are a working
default: keep them, replace them, or delete what you do not need.
-->

## 01 Brand identity

- Personality: [3 words, e.g. calm, precise, confident]
- Tone in UI copy: [e.g. plain, short sentences, no exclamation marks]
- Feel: [e.g. light, spacious, product first. Never busy.]
- Reference sites we like: [URL], [URL]

## 02 Color palette

Use tokens in components. Never raw hex values.

| Token | Value | Use |
| --- | --- | --- |
| --color-bg | #FFFFFF | Page background |
| --color-surface | #F7F7F8 | Cards, panels, inputs |
| --color-border | #E5E5E7 | Dividers, input borders |
| --color-text | #111111 | Body text |
| --color-text-muted | #6B6B70 | Secondary text, captions |
| --color-primary | #2563EB | Buttons, links, active states |
| --color-primary-hover | #1D4ED8 | Hover on primary |
| --color-success | #16A34A | Confirmations |
| --color-warning | #D97706 | Non-blocking warnings |
| --color-danger | #DC2626 | Errors, destructive actions |

Rules:
- One primary color. Accent colors are for state, not decoration.
- Text on any background passes WCAG AA (4.5:1 for body text).
- Dark mode: [not supported in v1 / swap tokens, never hard-code a color]

## 03 Typography

| Role | Font | Size | Weight | Line height |
| --- | --- | --- | --- | --- |
| Display | Inter | 48px | 700 | 1.1 |
| H1 | Inter | 36px | 700 | 1.15 |
| H2 | Inter | 28px | 600 | 1.2 |
| H3 | Inter | 22px | 600 | 1.3 |
| Body | Inter | 16px | 400 | 1.6 |
| Small | Inter | 14px | 400 | 1.5 |
| Label | Inter | 12px | 500 | 1.4, uppercase, 0.04em tracking |
| Code | JetBrains Mono | 14px | 400 | 1.5 |

Rules:
- Headings use tight letter spacing: H1 -0.02em, H2 -0.01em.
- Body text line length stays under 70 characters.
- Never more than two font families on a page.

## 04 Spacing

Base unit: 4px. Only use values from this scale.

| Token | Value | Typical use |
| --- | --- | --- |
| space-1 | 4px | Icon gaps |
| space-2 | 8px | Inside small components |
| space-3 | 12px | Input padding |
| space-4 | 16px | Default gap between elements |
| space-6 | 24px | Card padding |
| space-8 | 32px | Between groups |
| space-12 | 48px | Between sections (mobile) |
| space-24 | 96px | Between sections (desktop) |

Layout:
- Max content width 1200px, centered, 24px side padding on mobile.
- Grid: 12 columns on desktop, 4 on mobile, 24px gutter.

## 05 Border radius and shadows

| Token | Value | Use |
| --- | --- | --- |
| radius-sm | 6px | Inputs, tags, small buttons |
| radius-md | 10px | Buttons, dropdowns |
| radius-lg | 16px | Cards, modals |
| radius-full | 9999px | Avatars, pills |

Shadows: one level only, for elevated surfaces (dropdowns, modals).
`0 4px 16px rgba(0, 0, 0, 0.08)`. Flat cards get a border, not a shadow.

## 06 Components

Reuse before creating. Check the ui components folder first.
If a variant is not listed here, ask before adding it.

Buttons

| Variant | Background | Text | Border | Use |
| --- | --- | --- | --- | --- |
| Primary | primary | white | none | One per screen, the main action |
| Secondary | surface | text | border | Supporting actions |
| Ghost | transparent | text | none | Low-emphasis actions, toolbars |
| Danger | danger | white | none | Destructive actions, always confirm |

- Height 40px (44px on touch screens), padding 0 16px, radius-md, Small 500.
- States: hover (darken 8%), active (darken 12%), focus (2px primary ring),
  disabled (50% opacity, no pointer).
- Loading: spinner replaces the label, width stays fixed.

Inputs
- Height 40px, radius-sm, border color-border, focus ring primary.
- Label above the field, 12px, always visible. Placeholder is a hint, never the label.
- Error: border danger, message below in danger at 12px.

Cards
- Surface background, radius-lg, padding space-6, 1px color-border.
- No shadow unless the card is interactive.

Modals
- Max width 480px, radius-lg, overlay rgba(0, 0, 0, 0.4).
- Close on overlay click and on Escape.

Empty states
- One line saying what belongs here, plus one primary action. Never a blank panel.

## 07 Icons

- Set: Lucide. One set only.
- Sizes: 16px inline with text, 20px in buttons, 24px standalone.
- Stroke width 1.75.
- An icon is never the only label on an action. Add text or an aria-label.

## 08 Responsive breakpoints

| Name | Min width | Notes |
| --- | --- | --- |
| sm | 640px | Phones in landscape |
| md | 768px | Tablets |
| lg | 1024px | Laptops, side navigation appears |
| xl | 1280px | Desktops, max content width applies |

Rules:
- Mobile first. Every screen works at 375px wide before anything else.
- Touch targets are at least 44px.
- No horizontal scrolling anywhere except tables and code blocks.

## 09 Motion

- Duration: 150ms for micro interactions, 250ms for panels, 400ms for page transitions.
- Easing: ease-out when entering, ease-in when leaving.
- Respect prefers-reduced-motion: disable non-essential animation.
