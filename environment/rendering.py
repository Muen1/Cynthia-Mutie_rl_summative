"""
Pygame visualization for ArtAuthenticationEnv.

Renders a simulated gallery-intake wall: each resolved item becomes a
small "frame" on the wall, colour-coded by the agent's decision and
whether that decision was correct. The current item under review is
shown large in the centre with its live evidence bars (forgery
probability, embedding similarity, metadata completeness), so a viewer
can see exactly why the agent chose an action — this is what should be
shown full-screen in the required demo video.
"""

import pygame
import numpy as np

_WIDTH, _HEIGHT = 900, 600
_screen = None
_font = None
_clock = None

_COLOR_BG = (24, 22, 30)
_COLOR_PANEL = (40, 36, 48)
_COLOR_TEXT = (235, 230, 220)
_COLOR_CORRECT = (86, 180, 120)
_COLOR_WRONG = (200, 70, 70)
_COLOR_NEUTRAL = (150, 150, 160)
_COLOR_BAR_BG = (60, 56, 68)
_COLOR_ACCENT = (214, 158, 46)  # ochre — nods to African textile palettes

ACTION_LABELS = {
    0: "APPROVE",
    1: "FLAG FORGERY",
    2: "INVESTIGATE",
    3: "ESCALATE TO EXPERT",
    4: "REQUEST PROVENANCE",
}


def _ensure_init():
    global _screen, _font, _clock
    if _screen is None:
        pygame.init()
        _screen = pygame.display.set_mode((_WIDTH, _HEIGHT))
        pygame.display.set_caption("ArtGuard Africa — Verification Agent")
        _font = pygame.font.SysFont("arial", 18)
        _clock = pygame.time.Clock()


def _bar(surface, x, y, w, h, frac, color):
    pygame.draw.rect(surface, _COLOR_BAR_BG, (x, y, w, h), border_radius=4)
    fw = max(0, min(w, int(w * frac)))
    pygame.draw.rect(surface, color, (x, y, fw, h), border_radius=4)


def render_frame(env):
    _ensure_init()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
    surf = _screen
    surf.fill(_COLOR_BG)

    item = env.current_item
    # --- header ---
    title = _font.render(
        f"Item {env.items_resolved + 1}/{env.queue_size}   "
        f"step {env.step_count}", True, _COLOR_TEXT)
    surf.blit(title, (20, 16))

    # --- central evidence panel for the item currently being decided ---
    panel_rect = (250, 60, 400, 260)
    pygame.draw.rect(surf, _COLOR_PANEL, panel_rect, border_radius=10)
    labels = [
        ("CNN forgery probability", item["cnn_forgery_prob"], _COLOR_WRONG),
        ("Embedding similarity", item["embedding_similarity"], _COLOR_CORRECT),
        ("Metadata completeness", item["metadata_completeness"], _COLOR_ACCENT),
        ("Seller trust", item["seller_trust_score"], _COLOR_NEUTRAL),
        ("Artist risk index", item["artist_risk_index"], _COLOR_WRONG),
    ]
    yy = 80
    for label, val, color in labels:
        txt = _font.render(label, True, _COLOR_TEXT)
        surf.blit(txt, (270, yy))
        _bar(surf, 270, yy + 20, 340, 12, val, color)
        yy += 46

    # --- gallery wall of resolved items (history) ---
    cols = 10
    tile = 60
    ox, oy = 20, 350
    for i, (hist_item, action, correct) in enumerate(env.history[-40:]):
        cx = ox + (i % cols) * (tile + 6)
        cy = oy + (i // cols) * (tile + 6)
        if correct is None:
            color = _COLOR_NEUTRAL
        else:
            color = _COLOR_CORRECT if correct else _COLOR_WRONG
        pygame.draw.rect(surf, color, (cx, cy, tile, tile), border_radius=6)
        lbl = _font.render(ACTION_LABELS[action][:1], True, (20, 20, 20))
        surf.blit(lbl, (cx + tile // 2 - 6, cy + tile // 2 - 10))

    # --- legend ---
    legend = _font.render(
        "Green = correct decision   Red = incorrect   Grey = escalated/in-progress",
        True, _COLOR_TEXT)
    surf.blit(legend, (20, _HEIGHT - 30))

    pygame.display.flip()
    _clock.tick(env.metadata.get("render_fps", 4))

    if env.render_mode == "rgb_array":
        arr = pygame.surfarray.array3d(surf)
        return np.transpose(arr, (1, 0, 2))
    return None


def close_renderer():
    global _screen
    if _screen is not None:
        pygame.quit()
        _screen = None
