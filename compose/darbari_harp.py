"""Original slow Raag Darbari piece for plucked string (harp / swarmandal-like
electric-piano tone), tanpura drone and soft synthesized tabla.

Usage: python darbari_harp.py out.wav [--sa 138.6] [--bpm 72]
"""
import argparse
import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve, butter, sosfilt

SR = 48000
# Darbari: ati-komal Ga and Dha sit a little below their usual komal positions.
RATIO = {"S": 1.0, "R": 9 / 8, "g": 1.185, "m": 4 / 3, "P": 3 / 2, "d": 1.580, "n": 9 / 5}

ALAAP = "S:2 n.:1 S:1 R:1 g~:3 R:1 S:2 n.:1 d.:1 n.:1 S:3"
GAT = [
    "m:.5 P:.5 d~:1 n:.5 P:.5 m:1 P:1 g~:1 m:.5 R:.5 S:1",
    "n.:.5 S:.5 R:1 g~:1 R:.5 S:.5 n.:.5 S:.5 d.:.5 n.:.5 R:1 S:1",
    "m:.5 P:.5 n:.5 S':.5 R':1 S':1 d~:.5 n:.5 P:1 m:.5 P:.5 g~:1",
    "m:.5 R:.5 S:1 n.:.5 S:.5 R:.5 g:.5 m:1 g~:.5 m:.5 R:.5 S:1.5",
]
KEHERWA = ["Dha", "Ge", "Na", "Ti", "Na", "Ka", "Dhi", "Na"]


def parse(seq, sa):
    out = []
    for tok in seq.split():
        name, dur = tok.split(":")
        andolan = "~" in name
        name = name.replace("~", "")
        octave = name.count("'") - name.count(".")
        f = sa * RATIO[name.strip(".'")] * 2.0 ** octave
        out.append((f, float(dur), andolan))
    return out


def pluck(f, ring, vel=1.0, andolan=False):
    """FM tine (DX-style electric piano) blended with a decaying additive harp."""
    t = np.arange(int(ring * SR)) / SR
    cents = np.zeros_like(t)
    if andolan:
        # slow downward sway characteristic of Darbari Ga/Dha
        cents = -18 * (0.5 - 0.5 * np.cos(2 * np.pi * 0.8 * t)) * np.minimum(t / 0.3, 1)
    phase = 2 * np.pi * np.cumsum(f * 2 ** (cents / 1200)) / SR
    index = vel * (2.2 * np.exp(-t / 0.22) + 0.35)
    fm = np.sin(phase + index * np.sin(phase))
    tine = 0.12 * np.exp(-t / 0.04) * np.sin(14 * phase)
    harp = sum(k ** -1.3 * np.sin(k * phase) * np.exp(-t * (0.5 + 0.55 * k)) for k in range(1, 9))
    env = (1 - np.exp(-t / 0.002)) * np.exp(-t / 1.8)
    return vel * env * (0.55 * fm + 0.45 * harp + tine)


def tanpura(sa, dur):
    """Cycling Pa-Sa'-Sa'-Sa drone with jawari-like bright harmonics."""
    out = np.zeros(int(dur * SR) + SR * 6)
    strings = [sa * 3 / 4, sa, sa, sa / 2]
    cycle, step = 4.0, 1.0
    for c in np.arange(0, dur, cycle):
        for i, f in enumerate(strings):
            start = int((c + i * step) * SR)
            t = np.arange(int(5.5 * SR)) / SR
            detune = 1 + 0.0015 * np.sin(2 * np.pi * 0.3 * t)
            ph = 2 * np.pi * np.cumsum(f * detune) / SR
            jawari = 0.6 + 0.4 * np.exp(-t / 1.5)
            tone = sum((k ** -0.9) * np.sin(k * ph) * (jawari if k > 3 else 1) for k in range(1, 16))
            env = (1 - np.exp(-t / 0.05)) * np.exp(-t / 2.5)
            out[start:start + len(t)] += 0.05 * env * tone
    return out


def tabla_stroke(bol, sa):
    t = np.arange(int(0.8 * SR)) / SR
    noise = np.random.default_rng(len(bol)).standard_normal(len(t))
    dayan = sum(a * np.sin(2 * np.pi * sa * 2 * k * t) for k, a in [(1, 1), (2, 0.5), (3, 0.3), (4, 0.15)])
    na = (dayan * np.exp(-t / 0.18) + 0.3 * noise * np.exp(-t / 0.004))
    ti = sosfilt(butter(2, [1500, 5000], "bandpass", fs=SR, output="sos"), noise) * np.exp(-t / 0.03)
    ge_f = 85 + 25 * (1 - np.exp(-t / 0.12))
    ge = np.sin(2 * np.pi * np.cumsum(ge_f) / SR) * np.exp(-t / 0.45)
    return {
        "Na": na, "Ti": 0.8 * ti, "Ka": 0.6 * ti, "Ge": 1.2 * ge,
        "Dha": na + 1.2 * ge, "Dhi": 0.8 * ti + 0.5 * na + 1.2 * ge,
    }[bol]


def reverb(x, secs=2.6, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(secs * SR)) / SR
    ir = rng.standard_normal((2, len(t))) * np.exp(-t * 6.9 / secs)
    ir = sosfilt(butter(1, 6000, fs=SR, output="sos"), ir)
    return np.stack([fftconvolve(x[c], ir[c])[: x.shape[1]] for c in range(2)]) * 0.02


def place(buf, sig, t0, pan=0.0, gain=1.0):
    i = int(t0 * SR)
    n = min(len(sig), buf.shape[1] - i)
    l, r = np.cos((pan + 1) * np.pi / 4), np.sin((pan + 1) * np.pi / 4)
    buf[0, i:i + n] += gain * l * sig[:n]
    buf[1, i:i + n] += gain * r * sig[:n]


def strum(buf, notes, t0, spacing, gain, pan_from, pan_to):
    for i, f in enumerate(notes):
        pan = pan_from + (pan_to - pan_from) * i / max(len(notes) - 1, 1)
        place(buf, pluck(f, 3.0, 0.6), t0 + i * spacing, pan, gain)


def compose(sa, bpm):
    beat = 60 / bpm
    scale = [sa * RATIO[n] * 2 ** o for o in (0, 1) for n in "SRgmPdn"] + [sa * 4]
    alaap = parse(ALAAP, sa * 2)
    gat = [parse(c, sa * 2) for c in GAT]
    alaap_len = 1.6 + sum(d for _, d, _ in alaap) * beat * 1.15
    gat_len = 8 * beat * len(gat)
    total = alaap_len + gat_len + 7.0
    buf = np.zeros((2, int((total + 4) * SR)))

    drone = tanpura(sa, total)
    place(buf, drone[: buf.shape[1]], 0, -0.4, 0.8)
    place(buf, np.roll(drone, int(0.5 * SR))[: buf.shape[1]], 0, 0.4, 0.8)

    # opening swarmandal sweep, then free alaap (slightly rubato)
    strum(buf, scale, 0.0, 0.07, 0.35, -0.6, 0.6)
    t = 1.6
    for f, d, a in alaap:
        place(buf, pluck(f, max(d * beat * 1.15 + 1.2, 2.0), 0.9, a), t, 0.1)
        t += d * beat * 1.15

    # gat over keherwa, tabla fading in through the first cycle
    gat_start = alaap_len
    strum(buf, scale[::2], gat_start - 0.5, 0.05, 0.25, 0.5, -0.5)
    for c, cycle in enumerate(gat):
        c0 = gat_start + c * 8 * beat
        t = c0
        for f, d, a in cycle:
            place(buf, pluck(f, max(d * beat + 1.0, 1.5), 0.85, a), t, 0.1)
            t += d * beat
        for b, bass in [(0, sa), (4, sa * 3 / 4)]:
            place(buf, pluck(bass, 2.5, 0.5), c0 + b * beat, -0.1, 0.6)
        for b, bol in enumerate(KEHERWA):
            fade = min(1.0, (c * 8 + b + 1) / 8)
            place(buf, tabla_stroke(bol, sa), c0 + b * beat, 0.0, 0.18 * fade)

    # closing descending sweep resolving to Sa
    end = gat_start + gat_len
    strum(buf, scale[::-1], end, 0.06, 0.3, 0.6, -0.6)
    place(buf, pluck(sa, 6.0, 0.8), end + len(scale) * 0.06 + 0.3, 0.0)
    place(buf, pluck(sa * 2, 6.0, 0.6), end + len(scale) * 0.06 + 0.3, 0.1)

    buf = buf[:, : int(total * SR)]
    buf = buf + reverb(buf)
    fade = np.ones(buf.shape[1])
    n = int(3 * SR)
    fade[-n:] = np.linspace(1, 0, n) ** 2
    buf *= fade
    return buf / np.max(np.abs(buf)) * 0.89


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--sa", type=float, default=138.6, help="tonic Hz (default C#3)")
    ap.add_argument("--bpm", type=float, default=72)
    args = ap.parse_args()
    audio = compose(args.sa, args.bpm)
    sf.write(args.out, audio.T, SR)
    print(f"wrote {args.out} ({audio.shape[1] / SR:.1f}s)")
