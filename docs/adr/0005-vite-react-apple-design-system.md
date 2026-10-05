# ADR 0005: Vite + React 18 with Apple HIG & Claude Aesthetic Design System

## Context
The legacy UI used raw inline HTML files (`frontend.html`) with generic styling and AI-generated tropes (neon gradients, pulsing orbs, cartoon chat bubbles). The platform required a submission-ready, human-designed interface that feels calm, obvious, and refined.

## Decision
We built a modern React SPA using:
- **Build tool:** Vite + TypeScript strict.
- **Styling:** Bespoke design tokens in TailwindCSS v4 without generic stock component defaults.
- **Tokens:** Warm editorial stone surfaces (`#FAF9F5` / `#1A1918`), muted terracotta accents (`#C2603F`), and crisp 1px hairlines.
- **Acoustic visualizer:** Calm monochrome concentric circle scaling with live RMS speech energy (zero neon glow).
- **Living Style Guide:** Route `/design-system` documenting all tokens, specimens, and component states.

## Consequences
- **Positive:** Pristine first impression comparable to Apple and Anthropic products.
- **Positive:** Zero AI cliches; tactile, honest, and accessible (WCAG 2.2 AA).
