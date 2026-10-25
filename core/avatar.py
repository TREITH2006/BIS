"""
Compact, pretty animated chatbot character (`ChibiBot` / `HoloAvatar`) for the HUD.

Drop-in replacement for the heavy 3D wireframe head that preserves the exact
`HoloAvatar` lifecycle (`step`, `paint`, `glance`, `SPAN`, `shaded`) while
adding expressive procedural animation (`jump`, `react`, `express`) inspired by
open-source procedural social-robot expression engines:
- ggldnl/Procedural-Expression-Library (parametric lid slopes, pupil dilation,
  and Cozmo/Vector-style saccadic eye rigs)
- FluxGarage/RoboEyes (procedural mood geometries, happy/angry/curious eyelid
  masks, and spring-driven Squash & Stretch)

Pure software rendering via `QPainter` — zero external asset files, zero OpenGL
dependencies, and frame-rate independent exponential & spring physics.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from PyQt6.QtCore import QLineF, QPointF, QRectF, Qt
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
    QPolygonF,
    QRadialGradient,
)

# Mouth timing constants (in seconds) matching the HUD audio/viseme pipeline
_TAU_OPEN = 0.020     # jaw/mouth opening toward a vowel
_TAU_SHUT = 0.011     # fast consonant closure mid-word
_TAU_REST = 0.050     # gentle settle after speech ends
_TAU_SHAPE = 0.018    # viseme openness smoothing
_MIC_FLOOR = 0.14
_CLOSE_FRAC = 0.10


def _rate(dt: float, tau: float) -> float:
    """Frame-rate independent exponential smoothing factor."""
    if tau <= 1e-6:
        return 1.0
    return 1.0 - math.exp(-dt / tau)


def _c(col: QColor, a: float) -> QColor:
    """Copy of `col` at alpha `a` (0..255, clamped)."""
    q = QColor(col)
    q.setAlpha(int(max(0.0, min(255.0, a))))
    return q


def _blend(bg: QColor, col: QColor, a: float) -> QColor:
    """Pre-mix `col` at alpha `a` (0..255) onto `bg`, returning an opaque QColor."""
    f = max(0.0, min(1.0, a / 255.0))
    return QColor(
        int(bg.red() + (col.red() - bg.red()) * f),
        int(bg.green() + (col.green() - bg.green()) * f),
        int(bg.blue() + (col.blue() - bg.blue()) * f),
    )


@dataclass
class _Particle:
    """Lightweight reactive particle (hearts, sparkles, sweat drops, question marks, zzz)."""
    kind: str
    x: float
    y: float
    vx: float
    vy: float
    size: float
    life: float
    max_life: float
    rot: float = 0.0
    vrot: float = 0.0


@dataclass
class _Preset:
    """Parametric facial expression targets (inspired by Procedural-Expression-Library & RoboEyes)."""
    lid_top: float = 0.0        # 0 = open, + = lowered upper lid
    lid_bot: float = 0.0        # 0 = open, + = raised lower lid (happy squint / cheek push)
    lid_slant: float = 0.0      # + = angry inward tilt, - = sad/worried outward tilt
    eye_scale_x: float = 1.0    # horizontal eye scale
    eye_scale_y: float = 1.0    # vertical eye scale
    brow_lift: float = 0.0      # -0.5 (furrowed) .. +1.0 (raised)
    brow_tilt: float = 0.0      # + = angry V-shape, - = sad/curious arch
    brow_asym: float = 0.0      # asymmetric brow offset (for "curious" / "thinking")
    smile: float = 0.35         # -1.0 (frown) .. +1.0 (happy curve)
    blush: float = 0.15         # 0..1 cheek glow intensity
    ear_perk: float = 0.0       # -1.0 (droop) .. +1.0 (perked up)
    head_tilt: float = 0.0      # target roll angle in radians


_EXPRESSIONS: dict[str, _Preset] = {
    "neutral": _Preset(
        lid_top=0.0, lid_bot=0.0, lid_slant=0.0,
        eye_scale_x=1.0, eye_scale_y=1.0,
        brow_lift=0.0, brow_tilt=0.0, smile=0.38, blush=0.18, ear_perk=0.0, head_tilt=0.0,
    ),
    "happy": _Preset(
        lid_top=0.05, lid_bot=0.68, lid_slant=-0.10,
        eye_scale_x=1.06, eye_scale_y=0.92,
        brow_lift=0.45, brow_tilt=-0.18, smile=0.95, blush=0.75, ear_perk=0.55, head_tilt=0.06,
    ),
    "excited": _Preset(
        lid_top=0.0, lid_bot=0.45, lid_slant=-0.08,
        eye_scale_x=1.14, eye_scale_y=1.16,
        brow_lift=0.75, brow_tilt=-0.20, smile=1.0, blush=0.90, ear_perk=0.85, head_tilt=-0.05,
    ),
    "love": _Preset(
        lid_top=0.04, lid_bot=0.55, lid_slant=-0.15,
        eye_scale_x=1.10, eye_scale_y=1.05,
        brow_lift=0.50, brow_tilt=-0.30, smile=0.90, blush=1.0, ear_perk=0.65, head_tilt=0.12,
    ),
    "curious": _Preset(
        lid_top=0.0, lid_bot=0.08, lid_slant=0.0,
        eye_scale_x=1.05, eye_scale_y=1.12,
        brow_lift=0.45, brow_tilt=-0.10, brow_asym=0.55, smile=0.20, blush=0.25, ear_perk=0.70, head_tilt=0.22,
    ),
    "thinking": _Preset(
        lid_top=0.28, lid_bot=0.18, lid_slant=0.12,
        eye_scale_x=0.98, eye_scale_y=0.88,
        brow_lift=-0.15, brow_tilt=0.18, brow_asym=0.65, smile=-0.05, blush=0.10, ear_perk=-0.15, head_tilt=-0.16,
    ),
    "surprised": _Preset(
        lid_top=0.0, lid_bot=0.0, lid_slant=-0.08,
        eye_scale_x=1.22, eye_scale_y=1.28,
        brow_lift=0.95, brow_tilt=-0.25, smile=0.0, blush=0.45, ear_perk=1.0, head_tilt=0.0,
    ),
    "wink": _Preset(
        lid_top=0.0, lid_bot=0.42, lid_slant=-0.08,
        eye_scale_x=1.05, eye_scale_y=1.02,
        brow_lift=0.50, brow_tilt=-0.15, brow_asym=0.45, smile=0.88, blush=0.70, ear_perk=0.50, head_tilt=0.14,
    ),
    "sad": _Preset(
        lid_top=0.42, lid_bot=0.0, lid_slant=-0.48,
        eye_scale_x=0.95, eye_scale_y=0.92,
        brow_lift=-0.25, brow_tilt=-0.55, smile=-0.65, blush=0.10, ear_perk=-0.85, head_tilt=-0.12,
    ),
    "angry": _Preset(
        lid_top=0.38, lid_bot=0.22, lid_slant=0.62,
        eye_scale_x=1.04, eye_scale_y=0.84,
        brow_lift=-0.45, brow_tilt=0.72, smile=-0.55, blush=0.40, ear_perk=-0.35, head_tilt=0.0,
    ),
    "sleepy": _Preset(
        lid_top=0.82, lid_bot=0.12, lid_slant=-0.12,
        eye_scale_x=0.96, eye_scale_y=0.75,
        brow_lift=-0.20, brow_tilt=-0.20, smile=0.12, blush=0.20, ear_perk=-0.65, head_tilt=0.09,
    ),
}


class HoloAvatar:
    """Small, pretty animated companion bot for the HUD centre.

    Compatible with the existing `HoloAvatar` interface while adding:
        av.jump(power=1.0)             # energetic bounce with squash & stretch
        av.react(reaction="celebrate") # compound physical + particle reaction
        av.express(mood="happy")       # set persistent or timed facial expression

    Lifecycle:
        av = HoloAvatar()
        av.step(dt, amp, speaking=..., muted=..., state=...)
        av.paint(painter, cx, cy, r, primary, accent, bg)
    """

    shaded = True
    SPAN = 2.15  # Matches the HUD sizing contract

    def __init__(self) -> None:
        self._t = 0.0
        self._sway = 0.0

        # Pose & Spring Physics (Jump / Squash & Stretch / Tilt)
        self._jump_y = 0.0          # Vertical jump offset in head-radii (+ is up)
        self._jump_vy = 0.0         # Vertical velocity
        self._squash = 0.0          # + = stretched vertically, - = squashed flat
        self._squash_v = 0.0        # Squash spring velocity
        self._flip_angle = 0.0      # Full 360 spin angle (radians)
        self._flip_v = 0.0

        self._roll = 0.0            # Head tilt (z-rotation)
        self._roll_v = 0.0
        self._shake_t = 0.0         # Head-shake or nod timer
        self._shake_kind = ""       # "nod" | "shake" | "shiver" | "wiggle"

        self._yaw = 0.0
        self._pitch = 0.0

        # Audio & Lip-Sync State
        self._mouth = 0.0           # 0..1 smoothed mouth opening
        self._glow = 0.0            # 0..1 overall energy
        self._scan = -1.5           # Holographic scanline phase
        self._v_open = 1.0
        self._v_wide = 0.0
        self._wide = 0.0            # -1 (round "oo") .. +1 (wide "ee")
        self._v_peak = 0.18
        self._amp_slow = 0.0
        self._emph = 0.0

        # Blinking & Gaze Saccades
        self._blink = 0.0
        self._blink_at = 2.6
        self._wink_l = 0.0          # Independent left-eye wink
        self._wink_r = 0.0          # Independent right-eye wink
        self._gaze = [0.0, 0.0]
        self._gaze_tgt = [0.0, 0.0]
        self._gaze_at = 0.0
        self._gaze_bias = [0.0, 0.0]
        self._bias_tgt = [0.0, 0.0]
        self._bias_at = 0.0
        self._glance: tuple[float, float, float] | None = None

        # Expression Blending State
        self._mood = "neutral"
        self._mood_until = 0.0      # 0 = permanent until changed; >0 = reverts at _t
        self._lid_top = 0.0
        self._lid_bot = 0.0
        self._lid_slant = 0.0
        self._eye_sx = 1.0
        self._eye_sy = 1.0
        self._brow_lift = 0.0
        self._brow_tilt = 0.0
        self._brow_asym = 0.0
        self._smile = 0.38
        self._blush = 0.18
        self._ear_perk = 0.0
        self._ear_bounce = 0.0
        self._ear_v = 0.0

        # Floating Emote Particles
        self._particles: list[_Particle] = []
        self._zzz_at = 0.0

    # ── Public Action API: jump, react, express, glance ─────────────────────

    def jump(self, power: float = 1.0, flip: bool = False) -> None:
        """Make the character hop upward with Disney-style squash and stretch.

        Args:
            power: Jump strength multiplier (0.4 = cute hop, 1.0 = normal jump,
                   1.5 = super jump).
            flip:  If True, performs a joyful 360-degree mid-air somersault.
        """
        p = max(0.25, min(2.0, float(power)))
        # Pre-jump anticipation squash, then launch velocity
        self._squash = -0.32 * p
        self._squash_v = 3.8 * p
        self._jump_vy = 3.4 * p
        self._ear_v += 6.5 * p
        if flip:
            self._flip_angle = 0.0
            self._flip_v = math.tau * 1.85
        self._spawn_burst("sparkle", count=int(4 * p))

    def express(self, mood: str = "happy", duration: float = 0.0) -> None:
        """Transition smoothly to a named facial expression.

        Supported moods:
            "neutral", "happy", "excited", "love", "curious", "thinking",
            "surprised", "wink", "sad", "angry", "sleepy"

        Args:
            mood:     Name of the expression preset.
            duration: Seconds to hold before returning to "neutral" (0 = hold
                      indefinitely until the next `express()` call).
        """
        key = (mood or "neutral").strip().lower()
        if key not in _EXPRESSIONS:
            key = "neutral"
        self._mood = key
        self._mood_until = (self._t + float(duration)) if duration > 0.0 else 0.0

        # Subtle physical feedback on expression change
        self._squash_v += 1.1
        self._ear_v += 3.0
        if key == "wink":
            self._wink_r = 1.0
            if duration <= 0.0:
                self._mood_until = self._t + 1.4
        elif key == "love":
            self._spawn_burst("heart", count=4)
        elif key in ("happy", "excited"):
            self._spawn_burst("sparkle", count=3)
        elif key == "curious":
            self._spawn_burst("question", count=1)

    def react(self, kind: str = "wave") -> None:
        """Trigger an expressive compound reaction (jump + gesture + particles + mood).

        Supported reactions:
            - "celebrate" / "cheer": Double-hop spin with sparkles and excited face
            - "love" / "heart":      Blushing bounce with floating heart particles
            - "nod" / "agree":       Affirmative head-nod with happy squint
            - "shake" / "disagree":  Playful head-shake with curious/surprised look
            - "shock" / "surprised": Startled hop back with wide eyes & exclamation
            - "curious" / "question":Head tilt + ear perk + floating '?'
            - "laugh" / "giggle":    Rapid bouncy wiggle with happy crescent eyes
            - "sleep" / "yawn":      Droopy lids + floating 'Zzz' bubbles
            - "angry" / "pout":      Puffed cheeks, furrowed brows & steam/spark
        """
        k = (kind or "celebrate").strip().lower()
        if k in ("celebrate", "cheer", "party", "yay"):
            self.express("excited", duration=3.0)
            self.jump(power=1.15, flip=True)
            self._spawn_burst("sparkle", count=8)
        elif k in ("love", "heart", "adore", "blush"):
            self.express("love", duration=3.2)
            self.jump(power=0.75)
            self._spawn_burst("heart", count=7)
        elif k in ("nod", "agree", "yes", "ack"):
            self.express("happy", duration=1.8)
            self._shake_kind = "nod"
            self._shake_t = 0.75
            self._squash_v += 1.6
        elif k in ("shake", "no", "disagree"):
            self.express("curious", duration=1.8)
            self._shake_kind = "shake"
            self._shake_t = 0.75
        elif k in ("shock", "surprised", " gasp", "alert", "wow"):
            self.express("surprised", duration=2.2)
            self.jump(power=0.95)
            self._spawn_burst("exclaim", count=2)
        elif k in ("curious", "question", "huh", "think"):
            self.express("curious", duration=2.6)
            self._roll_v += 3.2
            self._spawn_burst("question", count=2)
        elif k in ("laugh", "giggle", "haha", "wiggle"):
            self.express("happy", duration=2.5)
            self._shake_kind = "wiggle"
            self._shake_t = 1.1
            self.jump(power=0.65)
            self._spawn_burst("sparkle", count=5)
        elif k in ("sad", "cry", "sorry", "oops"):
            self.express("sad", duration=3.0)
            self._squash = -0.18
            self._spawn_burst("sweat", count=3)
        elif k in ("angry", "mad", "pout", "hmph"):
            self.express("angry", duration=2.6)
            self._shake_kind = "shiver"
            self._shake_t = 0.65
            self.jump(power=0.55)
        elif k in ("wink", "flirt"):
            self.express("wink", duration=1.5)
            self.jump(power=0.5)
            self._spawn_burst("sparkle", count=3)
        elif k in ("sleep", "sleepy", "zzz"):
            self.express("sleepy", duration=4.0)
            self._spawn_burst("zzz", count=2)
        else:
            # Default friendly hop + wave wiggle
            self.express("happy", duration=2.0)
            self.jump(power=0.85)
            self._spawn_burst("sparkle", count=4)

    def glance(self, dx: float, dy: float, hold: float = 1.1) -> None:
        """Look deliberately at (dx, dy) in -1..1 for `hold` seconds."""
        self._glance = (
            max(-1.0, min(1.0, float(dx))),
            max(-1.0, min(1.0, float(dy))),
            self._t + max(0.1, float(hold)),
        )
        self._ear_v += 2.0

    # ── Internal Particle Spawner ───────────────────────────────────────────

    def _spawn_burst(self, kind: str, count: int = 3) -> None:
        for i in range(count):
            ang = random.uniform(-math.pi * 0.85, -math.pi * 0.15)
            spd = random.uniform(0.45, 1.05)
            px = random.uniform(-0.65, 0.65)
            py = random.uniform(-0.75, -0.15)
            if kind in ("question", "exclaim", "zzz"):
                side = 1.0 if i % 2 == 0 else -1.0
                px = side * random.uniform(0.62, 0.85)
                py = random.uniform(-0.85, -0.45)
                ang = -math.pi * 0.5 + side * 0.25
                spd = random.uniform(0.30, 0.55)
            life = random.uniform(0.75, 1.35)
            self._particles.append(
                _Particle(
                    kind=kind,
                    x=px,
                    y=py,
                    vx=math.cos(ang) * spd,
                    vy=math.sin(ang) * spd,
                    size=random.uniform(0.11, 0.18),
                    life=life,
                    max_life=life,
                    rot=random.uniform(-0.3, 0.3),
                    vrot=random.uniform(-2.0, 2.0),
                )
            )

    # ── Audio / Viseme Step ─────────────────────────────────────────────────

    def _mouth_step(
        self,
        dt: float,
        amp: float,
        live: bool,
        v_open: float | None,
        v_level: float | None,
    ) -> None:
        if v_open is None:
            shape = 1.0
        else:
            self._v_open += (v_open - self._v_open) * _rate(dt, _TAU_SHAPE)
            shape = self._v_open

        if v_level is None:
            gated = max(0.0, (amp - _MIC_FLOOR) / (1.0 - _MIC_FLOOR))
            drive = (gated ** 0.6) * (shape ** 0.75)
        else:
            self._v_peak = max(v_level, self._v_peak - dt * 0.55)
            ref = max(0.18, self._v_peak)
            q = (v_level - _CLOSE_FRAC * ref) / (ref * (1.0 - _CLOSE_FRAC))
            drive = max(0.0, min(1.0, q)) ** 0.85 * (shape ** 0.75)

        target = min(1.0, drive) if live else 0.0
        if target > self._mouth:
            tau = _TAU_OPEN
        elif live:
            tau = _TAU_SHUT
        else:
            tau = _TAU_REST
        self._mouth += (target - self._mouth) * _rate(dt, tau)
        if self._mouth < 0.002:
            self._mouth = 0.0

    # ── Main Animation Tick ─────────────────────────────────────────────────

    def step(
        self,
        dt: float,
        amp: float,
        speaking: bool = False,
        muted: bool = False,
        state: str = "",
        v_open: float | None = None,
        v_wide: float = 0.0,
        v_level: float | None = None,
        v_seq: list | None = None,
        v_hop: float = 0.02,
    ) -> None:
        """Advance all physics, lip-sync, saccades, expressions, and particles."""
        dt = max(0.001, min(0.10, float(dt)))
        self._t += dt
        t = self._t
        amp = max(0.0, min(1.0, float(amp)))
        live = speaking and not muted

        # 1. Revert timed expression if expired
        if self._mood_until > 0.0 and t >= self._mood_until:
            self._mood = "neutral"
            self._mood_until = 0.0
            self._wink_l = 0.0
            self._wink_r = 0.0

        # 2. Map assistant HUD state if no explicit mood override is active
        st = (state or "").upper()
        thinking = st in ("THINKING", "PROCESSING")
        asleep = st in ("SLEEPING", "STANDBY", "OFFLINE")
        listening = st == "LISTENING"

        active_mood = self._mood
        if active_mood == "neutral":
            if asleep:
                active_mood = "sleepy"
            elif thinking:
                active_mood = "thinking"
            elif listening and amp > 0.25:
                active_mood = "curious"
            elif live:
                active_mood = "happy"

        preset = _EXPRESSIONS.get(active_mood, _EXPRESSIONS["neutral"])

        # 3. Idle hover & gentle sway
        speed = (1.0 if not muted else 0.55) * (1.30 if live else 1.0)
        self._sway += dt * speed
        s = self._sway
        self._yaw = 0.20 * math.sin(s * 0.45) + 0.07 * math.sin(s * 0.95 + 1.2)
        self._pitch = 0.08 * math.sin(s * 0.38 + 0.6) + 0.03 * math.sin(s * 0.82)

        # 4. Mouth & Viseme Sub-Stepping
        if v_seq is not None:
            for lv, op, _wd in v_seq:
                self._mouth_step(v_hop, amp, live, op, lv)
        else:
            self._mouth_step(dt, amp, live, v_open, v_level)

        wide_t = v_wide if (live and v_open is not None) else 0.0
        self._wide += (max(-1.0, min(1.0, wide_t)) - self._wide) * _rate(dt, 0.030)

        # Speech syllable bounce
        self._emph += (self._mouth - self._emph) * _rate(
            dt, 0.050 if self._mouth > self._emph else 0.28
        )
        self._pitch -= self._emph * 0.06
        self._ear_v += self._emph * 8.0 * dt

        env = amp if live else 0.0
        self._amp_slow += (env - self._amp_slow) * _rate(
            dt, 0.14 if env > self._amp_slow else 0.34
        )

        # 5. Jump & Squash-and-Stretch Spring Physics
        if self._jump_y > 0.0 or self._jump_vy != 0.0:
            gravity = 9.8
            self._jump_vy -= gravity * dt
            self._jump_y += self._jump_vy * dt
            # Stretch in mid-air proportional to vertical speed
            self._squash += (clamp_val := max(-0.25, min(0.35, self._jump_vy * 0.08)) - self._squash) * _rate(dt, 0.04)
            if self._jump_y <= 0.0:
                self._jump_y = 0.0
                impact = min(1.2, abs(self._jump_vy) * 0.22)
                self._jump_vy = 0.0
                # Landing squash impact
                self._squash = -0.28 * impact
                self._squash_v = 2.6 * impact
                self._ear_v -= 5.0 * impact

        # Damped harmonic oscillator for squash & stretch
        k_sq, damp_sq = 150.0, 11.5
        acc_sq = -k_sq * self._squash - damp_sq * self._squash_v
        self._squash_v += acc_sq * dt
        self._squash += self._squash_v * dt

        # Mid-air 360 flip spin
        if self._flip_v > 0.0:
            self._flip_angle += self._flip_v * dt
            if self._flip_angle >= math.tau:
                self._flip_angle = 0.0
                self._flip_v = 0.0

        # Ear / Antenna spring physics
        ear_tgt = preset.ear_perk + 0.35 * self._amp_slow + (0.25 if live else 0.0)
        acc_ear = -120.0 * (self._ear_bounce - ear_tgt) - 10.0 * self._ear_v
        self._ear_v += acc_ear * dt
        self._ear_bounce += self._ear_v * dt

        # Head roll spring & gesture shakes (nod / shake / wiggle / shiver)
        roll_tgt = preset.head_tilt + 0.05 * math.sin(s * 0.55)
        if self._shake_t > 0.0:
            self._shake_t = max(0.0, self._shake_t - dt)
            phase = (0.75 - self._shake_t) * 24.0
            if self._shake_kind == "nod":
                self._pitch += 0.22 * math.sin(phase)
            elif self._shake_kind == "shake":
                self._yaw += 0.38 * math.sin(phase)
            elif self._shake_kind == "wiggle":
                roll_tgt += 0.25 * math.sin(phase * 1.3)
                self._squash += 0.08 * math.cos(phase * 2.0)
            elif self._shake_kind == "shiver":
                self._yaw += 0.12 * math.sin(phase * 2.5)
                roll_tgt += 0.10 * math.cos(phase * 2.5)

        acc_roll = -95.0 * (self._roll - roll_tgt) - 11.0 * self._roll_v
        self._roll_v += acc_roll * dt
        self._roll += self._roll_v * dt

        # 6. Smoothly interpolate expression rig parameters
        r_exp = _rate(dt, 0.095)
        self._lid_top += (preset.lid_top - self._lid_top) * r_exp
        self._lid_bot += (preset.lid_bot - self._lid_bot) * r_exp
        self._lid_slant += (preset.lid_slant - self._lid_slant) * r_exp
        self._eye_sx += (preset.eye_scale_x - self._eye_sx) * r_exp
        self._eye_sy += (preset.eye_scale_y - self._eye_sy) * r_exp

        brow_speech = 0.40 * self._amp_slow
        self._brow_lift += (preset.brow_lift + brow_speech - self._brow_lift) * r_exp
        self._brow_tilt += (preset.brow_tilt - self._brow_tilt) * r_exp
        self._brow_asym += (preset.brow_asym - self._brow_asym) * r_exp
        self._smile += (preset.smile - self._smile) * r_exp
        self._blush += (preset.blush - self._blush) * r_exp

        if self._mood != "wink":
            self._wink_l *= max(0.0, 1.0 - dt * 8.0)
            self._wink_r *= max(0.0, 1.0 - dt * 8.0)

        # 7. Gaze Saccades & State Bias (Cozmo/Vector style fast eye darts)
        if thinking:
            if t >= self._bias_at:
                self._bias_tgt = [
                    random.choice((-1.0, 1.0)) * random.uniform(0.45, 0.85),
                    random.uniform(-0.65, -0.30),  # Look up and to the side while thinking
                ]
                self._bias_at = t + 1.3 + 1.5 * random.random()
        elif asleep:
            self._bias_tgt = [0.0, 0.35]
            if t >= self._zzz_at:
                self._spawn_burst("zzz", count=1)
                self._zzz_at = t + 1.8
        else:
            self._bias_tgt = [0.0, 0.0]
            self._bias_at = 0.0

        for i in (0, 1):
            self._gaze_bias[i] += (self._bias_tgt[i] - self._gaze_bias[i]) * _rate(dt, 0.18)

        if t >= self._gaze_at:
            reach = 0.75 if live else (0.30 if thinking else 0.50)
            self._gaze_tgt = [
                random.uniform(-1.0, 1.0) * reach,
                random.uniform(-0.8, 0.8) * reach * 0.55,
            ]
            if live:
                self._gaze_at = t + 0.5 + 1.4 * random.random()
            elif thinking:
                self._gaze_at = t + 1.6 + 2.0 * random.random()
            else:
                self._gaze_at = t + 1.1 + 2.4 * random.random()

        if self._glance is not None:
            gx, gy, until = self._glance
            if t < until:
                self._gaze_tgt = [gx, gy]
            else:
                self._glance = None

        for i, b in enumerate(self._gaze_bias):
            tgt = max(-1.0, min(1.0, self._gaze_tgt[i] + b))
            # Fast saccade snap (tau ~ 28ms)
            self._gaze[i] += (tgt - self._gaze[i]) * _rate(dt, 0.028)

        # 8. Blinking
        if self._blink > 0.0:
            self._blink = max(0.0, self._blink - dt * 9.5)
        elif t >= self._blink_at:
            if asleep:
                self._blink_at = t + 5.0
            else:
                self._blink = 1.0
                # Occasional cute double-blink
                if random.random() < 0.22:
                    self._blink_at = t + 0.22
                else:
                    gap = 4.8 if thinking else 3.1
                    self._blink_at = t + gap + 2.6 * random.random()

        # 9. Energy Glow & Scanline
        target_glow = 0.0 if muted else max(amp, 0.35 if live else 0.0)
        self._glow += (target_glow - self._glow) * _rate(
            dt, 0.08 if target_glow > self._glow else 0.25
        )
        self._scan += dt * (0.65 + 1.4 * self._glow)
        if self._scan > 1.4:
            self._scan = -1.4

        # 10. Update Floating Emote Particles
        alive: list[_Particle] = []
        for pt in self._particles:
            pt.life -= dt
            if pt.life > 0.0:
                pt.x += pt.vx * dt
                pt.y += pt.vy * dt
                pt.vy -= 0.35 * dt  # Gentle upward buoyancy
                pt.rot += pt.vrot * dt
                alive.append(pt)
        self._particles = alive

    # ── Rendering ───────────────────────────────────────────────────────────

    def paint(
        self,
        p: QPainter,
        cx: float,
        cy: float,
        r: float,
        primary: QColor,
        accent: QColor,
        bg: QColor | None = None,
    ) -> None:
        """Render the compact animated companion character centred at (cx, cy)."""
        if bg is None:
            bg = QColor(8, 12, 20)

        p.save()
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        amp = self._glow
        hover_y = 0.055 * math.sin(self._sway * 1.65)
        total_lift = (self._jump_y + hover_y) * r

        # ── 1. Ambient Holographic Aura & Ground Shadow ─────────────────────
        ar = r * 1.75
        aura = QRadialGradient(cx, cy - total_lift * 0.4, ar)
        aura.setColorAt(0.00, _c(primary, 42 + 70 * amp))
        aura.setColorAt(0.45, _c(primary, 16 + 32 * amp))
        aura.setColorAt(1.00, _c(primary, 0))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(aura))
        p.drawEllipse(QRectF(cx - ar, cy - ar, ar * 2.0, ar * 2.0))

        # Ground shadow shrinks and softens as the character jumps higher
        shadow_scale = max(0.35, 1.0 - self._jump_y * 0.42)
        sw = r * 0.72 * shadow_scale
        sh = r * 0.13 * shadow_scale
        sy = cy + r * 0.88
        sh_grad = QRadialGradient(cx, sy, sw)
        sh_grad.setColorAt(0.0, _c(primary, (75 + 55 * amp) * shadow_scale))
        sh_grad.setColorAt(0.65, _c(primary, 20 * shadow_scale))
        sh_grad.setColorAt(1.0, _c(primary, 0))
        p.setBrush(QBrush(sh_grad))
        p.drawEllipse(QPointF(cx, sy), sw, sh)

        # Holographic emitter ring at the base
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(_c(primary, (65 + 70 * amp) * shadow_scale), 1.2))
        p.drawEllipse(QPointF(cx, sy), sw * 0.78, sh * 0.72)

        # ── 2. Apply Character Transform (Jump, Squash & Stretch, Tilt) ─────
        head_cx = cx + self._yaw * r * 0.16
        head_cy = cy - r * 0.06 - total_lift + self._pitch * r * 0.14

        p.translate(head_cx, head_cy)
        p.rotate(math.degrees(self._roll + self._flip_angle))

        sx = 1.0 - self._squash * 0.48
        sy_scale = 1.0 + self._squash * 0.62
        p.scale(sx, sy_scale)

        # ── 3. Floating Chibi Body & Hands (compact pod below the head) ─────
        self._paint_body_and_hands(p, r, primary, accent, bg, amp)

        # ── 4. Cat-Tech Ears / Expressive Side Fins & Antenna ───────────────
        self._paint_ears_and_antenna(p, r, primary, accent, bg, amp)

        # ── 5. Rounded Visor Head Shell ─────────────────────────────────────
        self._paint_head_shell(p, r, primary, accent, bg, amp)

        # ── 6. Expressive Face (Eyes, Brows, Blush, Mouth) inside Visor ─────
        self._paint_face_features(p, r, primary, accent, bg, amp)

        p.restore()

        # ── 7. Floating Emote Particles (in screen space around head) ───────
        p.save()
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self._paint_particles(p, cx, cy - total_lift * 0.6, r, primary, accent)
        p.restore()

    def _paint_body_and_hands(
        self,
        p: QPainter,
        r: float,
        primary: QColor,
        accent: QColor,
        bg: QColor,
        amp: float,
    ) -> None:
        """Draw a compact floating torso pod and two expressive hovering hands."""
        bw = r * 0.64
        bh = r * 0.44
        by = r * 0.44

        body_rect = QRectF(-bw * 0.5, by, bw, bh)
        body_path = QPainterPath()
        body_path.addRoundedRect(body_rect, bw * 0.42, bh * 0.48)

        if self.shaded:
            b_grad = QLinearGradient(0, by, 0, by + bh)
            b_grad.setColorAt(0.0, _blend(bg, primary, 165))
            b_grad.setColorAt(1.0, _blend(bg, primary, 55))
            p.setBrush(QBrush(b_grad))
        else:
            p.setBrush(QBrush(_c(primary, 32)))

        p.setPen(QPen(_c(primary, 175 + 50 * amp), 1.5))
        p.drawPath(body_path)

        # Glowing chest core (pulses with speech/heartbeats)
        core_r = r * (0.095 + 0.035 * amp + 0.02 * self._mouth)
        core_pt = QPointF(0.0, by + bh * 0.46)
        c_grad = QRadialGradient(core_pt, core_r * 1.9)
        c_grad.setColorAt(0.0, _c(accent, 245))
        c_grad.setColorAt(0.55, _c(primary, 140))
        c_grad.setColorAt(1.0, _c(primary, 0))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(c_grad))
        p.drawEllipse(core_pt, core_r * 1.9, core_r * 1.9)

        # Hovering chibi hands (react to mood, speech, and jumps)
        hand_lift = (0.18 * self._JumpOrExcite() + 0.10 * self._emph) * r
        hand_spread = r * (0.48 + 0.06 * math.sin(self._sway * 1.9))
        for side in (-1.0, 1.0):
            hx = side * hand_spread
            hy = by + bh * 0.36 - hand_lift + side * 0.03 * r * math.sin(self._sway * 2.4)
            p.save()
            p.translate(hx, hy)
            p.rotate(side * (18.0 - 25.0 * self._JumpOrExcite()))
            p.setBrush(QBrush(_blend(bg, primary, 140) if self.shaded else _c(primary, 45)))
            p.setPen(QPen(_c(primary, 190), 1.4))
            p.drawEllipse(QPointF(0.0, 0.0), r * 0.095, r * 0.072)
            p.restore()

    def _JumpOrExcite(self) -> float:
        return min(1.0, self._jump_y * 0.8 + (0.65 if self._mood in ("excited", "happy", "love") else 0.0))

    def _paint_ears_and_antenna(
        self,
        p: QPainter,
        r: float,
        primary: QColor,
        accent: QColor,
        bg: QColor,
        amp: float,
    ) -> None:
        """Draw expressive cat-tech side ears and a glowing top antenna."""
        perk = max(-1.0, min(1.3, self._ear_bounce))

        # Top holographic antenna
        ant_base_y = -r * 0.56
        ant_tip_x = self._roll * r * 0.25 + math.sin(self._sway * 2.2) * r * 0.06
        ant_tip_y = ant_base_y - r * (0.22 + 0.04 * perk)

        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(_c(primary, 185), 2.0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        path = QPainterPath(QPointF(0.0, ant_base_y))
        path.quadTo(QPointF(ant_tip_x * 0.4, (ant_base_y + ant_tip_y) * 0.5), QPointF(ant_tip_x, ant_tip_y))
        p.drawPath(path)

        orb_r = r * (0.065 + 0.025 * amp)
        orb_grad = QRadialGradient(QPointF(ant_tip_x, ant_tip_y), orb_r * 2.2)
        orb_grad.setColorAt(0.0, _c(accent, 255))
        orb_grad.setColorAt(0.5, _c(accent, 160))
        orb_grad.setColorAt(1.0, _c(primary, 0))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(orb_grad))
        p.drawEllipse(QPointF(ant_tip_x, ant_tip_y), orb_r * 2.2, orb_r * 2.2)
        p.setBrush(QBrush(_c(accent, 255)))
        p.drawEllipse(QPointF(ant_tip_x, ant_tip_y), orb_r * 0.75, orb_r * 0.75)

        # Expressive side ears / fins
        for side in (-1.0, 1.0):
            p.save()
            ex = side * r * 0.48
            ey = -r * 0.42
            p.translate(ex, ey)
            # Perk rotates ears upward; sad/sleepy droops them outward
            angle = side * (22.0 - perk * 18.0)
            p.rotate(angle)

            ear = QPainterPath()
            ear.moveTo(-side * r * 0.14, r * 0.06)
            ear.quadTo(side * r * 0.02, -r * 0.28, side * r * 0.16, -r * 0.26)
            ear.quadTo(side * r * 0.20, -r * 0.02, side * r * 0.04, r * 0.14)
            ear.closeSubpath()

            if self.shaded:
                p.setBrush(QBrush(_blend(bg, primary, 150)))
            else:
                p.setBrush(QBrush(_c(primary, 40)))
            p.setPen(QPen(_c(primary, 210), 1.6))
            p.drawPath(ear)

            # Inner ear glow triangle
            inner = QPainterPath()
            inner.moveTo(-side * r * 0.06, r * 0.03)
            inner.lineTo(side * r * 0.11, -r * 0.17)
            inner.lineTo(side * r * 0.03, r * 0.08)
            inner.closeSubpath()
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(_c(accent, 115 + 75 * amp)))
            p.drawPath(inner)
            p.restore()

    def _paint_head_shell(
        self,
        p: QPainter,
        r: float,
        primary: QColor,
        accent: QColor,
        bg: QColor,
        amp: float,
    ) -> None:
        """Draw the glossy rounded robot head casing and dark inner visor screen."""
        hw = r * 1.18
        hh = r * 0.96
        h_rect = QRectF(-hw * 0.5, -hh * 0.58, hw, hh)

        # Outer shell
        shell = QPainterPath()
        shell.addRoundedRect(h_rect, hw * 0.42, hh * 0.46)

        if self.shaded:
            s_grad = QLinearGradient(0.0, h_rect.top(), 0.0, h_rect.bottom())
            s_grad.setColorAt(0.00, _blend(bg, primary, 215))
            s_grad.setColorAt(0.55, _blend(bg, primary, 130))
            s_grad.setColorAt(1.00, _blend(bg, primary, 65))
            p.setBrush(QBrush(s_grad))
        else:
            p.setBrush(QBrush(_c(primary, 28)))

        p.setPen(QPen(_c(primary, 225), 2.0))
        p.drawPath(shell)

        # Specular crown highlight on outer shell
        crown_rect = QRectF(-hw * 0.32, h_rect.top() + hh * 0.06, hw * 0.64, hh * 0.18)
        crown_grad = QLinearGradient(0.0, crown_rect.top(), 0.0, crown_rect.bottom())
        crown_grad.setColorAt(0.0, _c(primary, 110))
        crown_grad.setColorAt(1.0, _c(primary, 0))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(crown_grad))
        p.drawRoundedRect(crown_rect, crown_rect.height() * 0.5, crown_rect.height() * 0.5)

        # Dark glass face visor recessed inside the shell
        vw = hw * 0.84
        vh = hh * 0.74
        v_rect = QRectF(
            -vw * 0.5 + self._yaw * r * 0.08,
            -vh * 0.52 + self._pitch * r * 0.08,
            vw,
            vh,
        )
        visor = QPainterPath()
        visor.addRoundedRect(v_rect, vw * 0.36, vh * 0.40)

        v_grad = QLinearGradient(0.0, v_rect.top(), 0.0, v_rect.bottom())
        v_grad.setColorAt(0.0, _blend(bg, primary, 18))
        v_grad.setColorAt(1.0, _blend(bg, primary, 36))
        p.setBrush(QBrush(v_grad))
        p.setPen(QPen(_c(primary, 145 + 65 * amp), 1.4))
        p.drawPath(visor)

        # Subtle holographic scanline sweep inside the visor
        scan_y = v_rect.center().y() + self._scan * (vh * 0.45)
        if v_rect.top() < scan_y < v_rect.bottom():
            p.save()
            p.setClipPath(visor)
            sc_grad = QLinearGradient(0.0, scan_y - r * 0.08, 0.0, scan_y + r * 0.08)
            sc_grad.setColorAt(0.0, _c(accent, 0))
            sc_grad.setColorAt(0.5, _c(accent, 45 + 50 * amp))
            sc_grad.setColorAt(1.0, _c(accent, 0))
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(sc_grad))
            p.drawRect(QRectF(v_rect.left(), scan_y - r * 0.08, v_rect.width(), r * 0.16))
            p.restore()

    def _paint_face_features(
        self,
        p: QPainter,
        r: float,
        primary: QColor,
        accent: QColor,
        bg: QColor,
        amp: float,
    ) -> None:
        """Paint expressive eyes, brows, rosy cheeks, and lip-synced mouth."""
        ox = self._yaw * r * 0.18 + self._gaze[0] * r * 0.085
        oy = self._pitch * r * 0.15 + self._gaze[1] * r * 0.065

        # ── 1. Soft Cheek Blush (glows brighter when happy, loved, or speaking)
        blush_a = min(1.0, self._blush + 0.25 * amp)
        if blush_a > 0.04:
            for side in (-1.0, 1.0):
                bx = ox * 0.7 + side * r * 0.31
                by = oy * 0.7 + r * 0.06
                br = r * 0.11
                b_grad = QRadialGradient(QPointF(bx, by), br)
                # Warm coral-pink mixed with the theme's accent colour
                pink = QColor(
                    min(255, int(accent.red() * 0.45 + 255 * 0.55)),
                    min(255, int(accent.green() * 0.35 + 105 * 0.65)),
                    min(255, int(accent.blue() * 0.45 + 165 * 0.55)),
                )
                b_grad.setColorAt(0.0, _c(pink, 145 * blush_a))
                b_grad.setColorAt(1.0, _c(pink, 0))
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QBrush(b_grad))
                p.drawEllipse(QPointF(bx, by), br * 1.25, br * 0.82)

        # ── 2. Eyes (Cozmo / RoboEyes style procedural shapes + specular catchlights)
        eye_sep = r * 0.23
        eye_y = oy - r * 0.085
        base_ew = r * 0.135 * self._eye_sx
        base_eh = r * 0.165 * self._eye_sy

        visor_dark = _blend(bg, primary, 26)

        for idx, side in enumerate((-1.0, 1.0)):
            ex = ox + side * eye_sep
            wink = self._wink_l if idx == 0 else self._wink_r
            close_amt = min(1.0, max(self._blink, wink))

            # If eye is winking or happy-squinting strongly, draw a cute curved arc
            if wink > 0.55:
                self._draw_happy_arc_eye(p, ex, eye_y, base_ew, base_eh, accent, primary)
                continue

            ew = base_ew * (1.0 + 0.12 * close_amt)
            eh = base_eh * max(0.08, 1.0 - 0.92 * close_amt)

            p.save()
            p.translate(ex, eye_y)

            # Outer eye glow
            e_glow = QRadialGradient(QPointF(0.0, 0.0), max(ew, eh) * 1.45)
            e_glow.setColorAt(0.0, _c(accent, 95 + 75 * amp))
            e_glow.setColorAt(1.0, _c(primary, 0))
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(e_glow))
            p.drawEllipse(QPointF(0.0, 0.0), ew * 1.45, eh * 1.45)

            # Main glowing iris capsule
            eye_rect = QRectF(-ew, -eh, ew * 2.0, eh * 2.0)
            eye_path = QPainterPath()
            eye_path.addRoundedRect(eye_rect, ew * 0.78, eh * 0.78)

            iris_grad = QLinearGradient(0.0, -eh, 0.0, eh)
            iris_grad.setColorAt(0.0, _c(primary, 245))
            iris_grad.setColorAt(1.0, _c(accent, 255))
            p.setBrush(QBrush(iris_grad))
            p.drawPath(eye_path)

            # Clip internal pupil & catchlights to the eye shape
            p.setClipPath(eye_path)

            # Heart overlay inside the eyes when in "love" mood
            if self._mood == "love" and close_amt < 0.4:
                self._draw_mini_heart(
                    p,
                    self._gaze[0] * ew * 0.22,
                    self._gaze[1] * eh * 0.22,
                    min(ew, eh) * 0.72,
                    QColor(255, 240, 250),
                )
            elif close_amt < 0.65:
                # Deep pupil centre that shifts slightly with gaze
                px = self._gaze[0] * ew * 0.25
                py = self._gaze[1] * eh * 0.25
                p.setBrush(QBrush(_blend(bg, primary, 75)))
                p.drawEllipse(QPointF(px, py), ew * 0.46, eh * 0.48)

                # Dual anime-style bright specular catchlights
                p.setBrush(QBrush(QColor(255, 255, 255, 235)))
                p.drawEllipse(
                    QPointF(px - ew * 0.30, py - eh * 0.32),
                    ew * 0.26,
                    eh * 0.24,
                )
                p.setBrush(QBrush(QColor(255, 255, 255, 180)))
                p.drawEllipse(
                    QPointF(px + ew * 0.28, py + eh * 0.28),
                    ew * 0.13,
                    eh * 0.13,
                )

            # Procedural Upper Eyelid Mask (handles slant for angry/sad & lid droop)
            if self._lid_top > 0.01 or abs(self._lid_slant) > 0.01:
                slant = self._lid_slant * eh * 0.85 * (-side)
                drop = -eh + (self._lid_top * eh * 1.65)
                lid_poly = QPolygonF(
                    [
                        QPointF(-ew * 1.3, -eh * 1.4),
                        QPointF(ew * 1.3, -eh * 1.4),
                        QPointF(ew * 1.3, drop - slant),
                        QPointF(-ew * 1.3, drop + slant),
                    ]
                )
                p.setBrush(QBrush(visor_dark))
                p.drawPolygon(lid_poly)

            # Procedural Lower Eyelid Mask (curved upward for happy crescent squint)
            if self._lid_bot > 0.02:
                rise = eh - (self._lid_bot * eh * 1.35)
                bot_path = QPainterPath()
                bot_path.moveTo(-ew * 1.3, eh * 1.4)
                bot_path.lineTo(-ew * 1.3, rise + eh * 0.35)
                bot_path.quadTo(0.0, rise - eh * 0.35, ew * 1.3, rise + eh * 0.35)
                bot_path.lineTo(ew * 1.3, eh * 1.4)
                bot_path.closeSubpath()
                p.setBrush(QBrush(visor_dark))
                p.drawPath(bot_path)

            p.restore()

        # ── 3. Expressive Floating Brows ────────────────────────────────────
        brow_pen = QPen(_c(primary, 215), max(1.8, r * 0.032), Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
        p.setPen(brow_pen)
        p.setBrush(Qt.BrushStyle.NoBrush)

        for idx, side in enumerate((-1.0, 1.0)):
            asym = self._brow_asym * (0.12 * r if idx == 1 else -0.04 * r)
            bx = ox + side * eye_sep
            by = eye_y - base_eh * 1.32 - self._brow_lift * r * 0.085 - asym
            tilt = self._brow_tilt * r * 0.075 * (-side)
            bw = base_ew * 0.95

            b_path = QPainterPath()
            b_path.moveTo(bx - bw, by - tilt)
            b_path.quadTo(bx, by - r * 0.035 - abs(tilt) * 0.3, bx + bw, by + tilt)
            p.drawPath(b_path)

        # ── 4. Lip-Synced & Expressive Mouth ────────────────────────────────
        mx = ox * 0.85
        my = oy + r * 0.145
        mw = r * (0.115 + 0.035 * self._wide + 0.025 * self._smile)
        open_amt = self._mouth

        if self._mood == "surprised" and open_amt < 0.35:
            open_amt = 0.42
            mw = r * 0.085

        if open_amt > 0.035:
            mh = r * (0.035 + 0.145 * open_amt)
            m_path = QPainterPath()
            if self._smile >= 0.0:
                # Cute 'D'-shaped happy talking mouth
                m_path.moveTo(mx - mw, my - mh * 0.25)
                m_path.quadTo(mx, my - mh * (0.15 - 0.25 * self._smile), mx + mw, my - mh * 0.25)
                m_path.quadTo(mx + mw * 0.85, my + mh, mx, my + mh)
                m_path.quadTo(mx - mw * 0.85, my + mh, mx - mw, my - mh * 0.25)
            else:
                # Rounded 'O' or sad open mouth
                m_path.addEllipse(QPointF(mx, my + mh * 0.2), mw * 0.85, mh * 0.75)

            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(_blend(bg, primary, 22)))
            p.drawPath(m_path)

            # Cute tongue & upper teeth strip inside clipped mouth cavity
            p.save()
            p.setClipPath(m_path)
            # Soft glowing tongue at bottom of mouth
            tongue_col = QColor(
                min(255, int(accent.red() * 0.5 + 245 * 0.5)),
                min(255, int(accent.green() * 0.4 + 110 * 0.6)),
                min(255, int(accent.blue() * 0.5 + 150 * 0.5)),
                210,
            )
            p.setBrush(QBrush(tongue_col))
            p.drawEllipse(QPointF(mx, my + mh * 0.85), mw * 0.72, mh * 0.58)

            # Tiny bright upper tooth bar when speaking clearly
            if open_amt > 0.12:
                p.setBrush(QBrush(QColor(240, 250, 255, 220)))
                p.drawRect(QRectF(mx - mw * 0.65, my - mh * 0.35, mw * 1.3, mh * 0.34))
            p.restore()

            # Glowing lip outline
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.setPen(QPen(_c(primary, 225), 1.6))
            p.drawPath(m_path)
        else:
            # Closed mouth: expressive smile, cat-like ':3' curve, or gentle pout
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.setPen(QPen(_c(primary, 230), max(1.8, r * 0.030), Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            curve = self._smile * r * 0.065
            m_line = QPainterPath()
            if self._mood in ("love", "wink", "curious"):
                # Cute 'w' / cat-mouth curve
                m_line.moveTo(mx - mw, my - curve * 0.25)
                m_line.quadTo(mx - mw * 0.5, my + curve * 0.75, mx, my - curve * 0.15)
                m_line.quadTo(mx + mw * 0.5, my + curve * 0.75, mx + mw, my - curve * 0.25)
            else:
                m_line.moveTo(mx - mw, my - curve * 0.35)
                m_line.quadTo(mx, my + curve, mx + mw, my - curve * 0.35)
            p.drawPath(m_line)

    def _draw_happy_arc_eye(
        self,
        p: QPainter,
        ex: float,
        ey: float,
        ew: float,
        eh: float,
        accent: QColor,
        primary: QColor,
    ) -> None:
        """Draw a glowing '^' happy wink arc."""
        arc = QPainterPath()
        arc.moveTo(ex - ew * 0.95, ey + eh * 0.15)
        arc.quadTo(ex, ey - eh * 0.85, ex + ew * 0.95, ey + eh * 0.15)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(_c(accent, 110), ew * 0.45, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        p.drawPath(arc)
        p.setPen(QPen(_c(primary, 245), ew * 0.24, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        p.drawPath(arc)

    def _draw_mini_heart(
        self, p: QPainter, x: float, y: float, size: float, col: QColor
    ) -> None:
        h = QPainterPath()
        h.moveTo(x, y + size * 0.55)
        h.cubicTo(
            x - size * 1.1, y - size * 0.1,
            x - size * 0.55, y - size * 0.95,
            x, y - size * 0.32,
        )
        h.cubicTo(
            x + size * 0.55, y - size * 0.95,
            x + size * 1.1, y - size * 0.1,
            x, y + size * 0.55,
        )
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(col))
        p.drawPath(h)

    def _paint_particles(
        self,
        p: QPainter,
        cx: float,
        cy: float,
        r: float,
        primary: QColor,
        accent: QColor,
    ) -> None:
        """Render floating reaction particles (hearts, 4-point sparkles, ?, !, Zzz)."""
        for pt in self._particles:
            ratio = max(0.0, min(1.0, pt.life / max(1e-4, pt.max_life)))
            # Smooth fade-in and fade-out envelope
            alpha = math.sin(ratio * math.pi)
            px = cx + pt.x * r
            py = cy + pt.y * r
            sz = pt.size * r * (0.7 + 0.3 * alpha)

            p.save()
            p.translate(px, py)
            p.rotate(math.degrees(pt.rot))

            if pt.kind == "heart":
                col = QColor(
                    min(255, int(accent.red() * 0.3 + 255 * 0.7)),
                    min(255, int(accent.green() * 0.3 + 90 * 0.7)),
                    min(255, int(accent.blue() * 0.3 + 150 * 0.7)),
                    int(240 * alpha),
                )
                self._draw_mini_heart(p, 0.0, 0.0, sz, col)

            elif pt.kind == "sparkle":
                star = QPainterPath()
                star.moveTo(0.0, -sz)
                star.quadTo(0.0, 0.0, sz, 0.0)
                star.quadTo(0.0, 0.0, 0.0, sz)
                star.quadTo(0.0, 0.0, -sz, 0.0)
                star.quadTo(0.0, 0.0, 0.0, -sz)
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QBrush(_c(accent, 245 * alpha)))
                p.drawPath(star)

            elif pt.kind == "question":
                p.setPen(QPen(_c(accent, 245 * alpha), max(2.0, sz * 0.22), Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
                p.setBrush(Qt.BrushStyle.NoBrush)
                q_path = QPainterPath()
                q_path.moveTo(-sz * 0.35, -sz * 0.45)
                q_path.cubicTo(-sz * 0.35, -sz * 0.95, sz * 0.45, -sz * 0.95, sz * 0.45, -sz * 0.35)
                q_path.quadTo(sz * 0.45, -sz * 0.05, 0.0, sz * 0.15)
                p.drawPath(q_path)
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QBrush(_c(accent, 245 * alpha)))
                p.drawEllipse(QPointF(0.0, sz * 0.48), sz * 0.13, sz * 0.13)

            elif pt.kind == "exclaim":
                p.setPen(QPen(_c(accent, 250 * alpha), max(2.2, sz * 0.25), Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
                p.drawLine(QLineF(0.0, -sz * 0.7, 0.0, sz * 0.1))
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QBrush(_c(accent, 250 * alpha)))
                p.drawEllipse(QPointF(0.0, sz * 0.48), sz * 0.14, sz * 0.14)

            elif pt.kind == "zzz":
                p.setPen(QPen(_c(primary, 210 * alpha), max(1.8, sz * 0.18), Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
                p.setBrush(Qt.BrushStyle.NoBrush)
                z_path = QPainterPath()
                z_path.moveTo(-sz * 0.4, -sz * 0.4)
                z_path.lineTo(sz * 0.4, -sz * 0.4)
                z_path.lineTo(-sz * 0.4, sz * 0.4)
                z_path.lineTo(sz * 0.4, sz * 0.4)
                p.drawPath(z_path)

            elif pt.kind == "sweat":
                drop = QPainterPath()
                drop.moveTo(0.0, -sz * 0.65)
                drop.quadTo(sz * 0.55, sz * 0.15, 0.0, sz * 0.65)
                drop.quadTo(-sz * 0.55, sz * 0.15, 0.0, -sz * 0.65)
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QBrush(_c(primary, 210 * alpha)))
                p.drawPath(drop)

            p.restore()


# Alias so callers can import either `HoloAvatar` or `ChibiBot`
ChibiBot = HoloAvatar


# ── Standalone Interactive Demo Harness ─────────────────────────────────────
if __name__ == "__main__":
    import sys
    import time
    from PyQt6.QtCore import QTimer
    from PyQt6.QtWidgets import (
        QApplication,
        QHBoxLayout,
        QPushButton,
        QVBoxLayout,
        QWidget,
    )

    class _DemoCanvas(QWidget):
        def __init__(self) -> None:
            super().__init__()
            self.setWindowTitle("ChibiBot / HoloAvatar — Interactive HUD Demo")
            self.resize(560, 560)
            self.av = HoloAvatar()
            self._last = time.perf_counter()
            self._speaking = False

            root = QVBoxLayout(self)
            root.addStretch(1)

            row1 = QHBoxLayout()
            for label, fn in [
                ("Jump", lambda: self.av.jump(1.0)),
                ("Flip Jump", lambda: self.av.jump(1.25, flip=True)),
                ("Celebrate", lambda: self.av.react("celebrate")),
                ("Love", lambda: self.av.react("love")),
                ("Curious ?", lambda: self.av.react("curious")),
                ("Laugh", lambda: self.av.react("laugh")),
            ]:
                btn = QPushButton(label)
                btn.clicked.connect(fn)
                row1.addWidget(btn)
            root.addLayout(row1)

            row2 = QHBoxLayout()
            for mood in ("happy", "excited", "wink", "thinking", "surprised", "sad", "angry", "sleepy", "neutral"):
                btn = QPushButton(mood.capitalize())
                btn.clicked.connect(lambda _=False, m=mood: self.av.express(m))
                row2.addWidget(btn)
            talk_btn = QPushButton("Toggle Talk")
            talk_btn.setCheckable(True)
            talk_btn.toggled.connect(self._set_talk)
            row2.addWidget(talk_btn)
            root.addLayout(row2)

            self._timer = QTimer(self)
            self._timer.timeout.connect(self._tick)
            self._timer.start(16)

        def _set_talk(self, on: bool) -> None:
            self._speaking = on

        def _tick(self) -> None:
            now = time.perf_counter()
            dt = now - self._last
            self._last = now
            amp = (0.35 + 0.55 * abs(math.sin(now * 9.5))) if self._speaking else 0.05
            v_open = (0.4 + 0.6 * abs(math.sin(now * 12.0))) if self._speaking else None
            v_wide = math.sin(now * 5.0) if self._speaking else 0.0
            self.av.step(dt, amp, speaking=self._speaking, v_open=v_open, v_wide=v_wide, v_level=amp if self._speaking else None)
            self.update()

        def paintEvent(self, _event) -> None:
            p = QPainter(self)
            bg = QColor(10, 14, 24)
            p.fillRect(self.rect(), bg)
            primary = QColor(56, 189, 248)   # Cyan HUD primary
            accent = QColor(168, 85, 247)    # Violet HUD accent
            r = min(self.width(), self.height()) * 0.24
            self.av.paint(p, self.width() * 0.5, self.height() * 0.45, r, primary, accent, bg)

    app = QApplication(sys.argv)
    w = _DemoCanvas()
    w.show()
    sys.exit(app.exec())