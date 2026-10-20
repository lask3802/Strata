#!/usr/bin/env python3
"""Prepare local web assets from the supplied Strata logo masters.

Requires fontTools (the outlined wordmark) and numpy + Pillow (the Strata Link
recolour of the PNG icons) only while preparing; they are imported before anything
is written. The checked-in exports have no font or Python dependency at app runtime. Source files stay
untouched; public/brand/source-manifest.json records every derivation.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import shutil
import xml.etree.ElementTree as ET

import numpy as np
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.ttLib import TTFont
from PIL import Image


APP = Path(__file__).resolve().parent.parent
MASTERS = APP.parent / "Strata-Logo-DesignFiles"
PUBLIC = APP / "public"
BRAND = PUBLIC / "brand"
FONT = PUBLIC / "fonts" / "GTWalsheimPro-Bold.woff2"
SVG_NS = "http://www.w3.org/2000/svg"
INK = {"light": "#221e1f", "dark": "#eef1f3"}
# Strata Link: the fork's mark has a cyan middle bar so it is not mistaken for upstream Strata's. The masters still carry upstream's
# emerald; LINK_RECOLOUR is applied to everything derived from them, and the SVG/PNG steps below record it in the manifest.
GREEN = {"light": "#06b6d4", "dark": "#22d3ee"}
UPSTREAM_GREEN = {"light": "#10b981", "dark": "#34d399"}
LINK_RECOLOUR = {UPSTREAM_GREEN[theme]: GREEN[theme] for theme in GREEN}
TILE = (0x16, 0x1A, 0x1D)
INK_ON_TILE = (0xEE, 0xF1, 0xF3)


def link_svg(text: str) -> str:
    for upstream, link in LINK_RECOLOUR.items():
        text = text.replace(upstream, link)
    return text


def link_note(theme: str) -> str:
    return f"recoloured for Strata Link: middle bar {UPSTREAM_GREEN[theme]} -> {GREEN[theme]}"


def link_png(path: Path, theme: str) -> None:
    """Move the green share of every pixel to the cyan, in place. A pixel is unmixed into tile, ink and green
    (p - tile = a*(green - tile) + b*(ink - tile)); only a*(cyan - green) is added, so edges stay smooth and
    pixels with no green keep their exact RGBA. Flat-colour icons (the favicons, no tile) are shifted directly."""
    arr = np.array(Image.open(path).convert("RGBA"))
    rgb = arr[..., :3].astype(np.float64)
    hexa = lambda h: np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)], np.float64)
    green, cyan = hexa(UPSTREAM_GREEN[theme]), hexa(GREEN[theme])
    if (arr[..., :3] == TILE).all(-1).mean() > 0.25:
        tile, ink = np.array(TILE, np.float64), np.array(INK_ON_TILE, np.float64)
        u, v, d = green - tile, ink - tile, rgb - tile
        uu, vv, uv = u @ u, v @ v, u @ v
        a = ((d @ u) * vv - (d @ v) * uv) / (uu * vv - uv * uv)
        b = ((d @ v) * uu - (d @ u) * uv) / (uu * vv - uv * uv)
        residual = np.linalg.norm(d - a[..., None] * u - b[..., None] * v, axis=-1)
        hit = (arr[..., 3] > 0) & (a >= 0.02) & (a <= 1.35) & (b >= -0.12) & (residual <= 6.0)
    else:
        a = np.ones(arr.shape[:2])
        hit = (arr[..., 3] > 0) & (rgb[..., 1] - rgb[..., 0] > 50)
    arr[..., :3] = np.where(hit[..., None], np.clip(np.rint(rgb + a[..., None] * (cyan - green)), 0, 255), rgb).astype(np.uint8)
    Image.fromarray(arr, "RGBA").save(path, "PNG")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source(relative: str) -> Path:
    path = MASTERS / relative
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def record(entries: list[dict[str, str]], master: str, output: Path, method: str) -> None:
    entries.append({
        "source": master,
        "source_sha256": digest(source(master)),
        "output": output.relative_to(PUBLIC).as_posix(),
        "output_sha256": digest(output),
        "method": method,
    })


def copy(entries: list[dict[str, str]], master: str, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source(master), output)
    record(entries, master, output, "unchanged copy")


def prepare_mark(entries: list[dict[str, str]], theme: str) -> None:
    master = f"logo/strata-mark-{theme}.svg"
    output = BRAND / f"strata-mark-{theme}.svg"
    text = re.sub(r"<metadata>.*?</metadata>", "", source(master).read_text(), flags=re.S)
    output.write_text(link_svg(text.replace(' xmlns:c2pa="http://c2pa.org/manifest"', "")), newline="\n")
    record(entries, master, output, f"copy of the upstream master; {link_note(theme)}; C2PA block removed (its hash covered the upstream bytes)")


def art_rectangles(master: str) -> str:
    root = ET.parse(source(master)).getroot()
    rects = root.findall(f"{{{SVG_NS}}}rect")
    if len(rects) != 3:
        raise ValueError(f"Expected the three approved bars in {master}")
    return "\n".join(
        f'<rect x="{r.attrib["x"]}" y="{r.attrib["y"]}" '
        f'width="{r.attrib["width"]}" height="{r.attrib["height"]}" '
        f'rx="{r.attrib["rx"]}" fill="{link_svg(r.attrib["fill"])}" />'
        for r in rects
    )


def outlined_wordmark(font: TTFont) -> str:
    cmap = font.getBestCmap()
    glyphs = font.getGlyphSet()
    scale = 44 / font["head"].unitsPerEm
    tracking_units = 1.54 / scale
    cursor = 0.0
    paths = []
    letters = "strata"
    names = [cmap[ord(letter)] for letter in letters]
    for index, name in enumerate(names):
        pen = SVGPathPen(glyphs)
        glyphs[name].draw(pen)
        paths.append(f'<path transform="translate({cursor:.4f} 0)" d="{pen.getCommands()}" />')
        cursor += font["hmtx"][name][0] - tracking_units
        if index + 1 < len(names):
            cursor += pair_kerning(font, name, names[index + 1])
    return (f'<g transform="translate(60 37.75) scale({scale:.6f} -{scale:.6f})">'
            + "".join(paths) + "</g>")


def pair_kerning(font: TTFont, left: str, right: str) -> int:
    """Apply the font's OpenType kern feature for this fixed six-letter word."""
    gpos = font["GPOS"].table
    kern_lookups = {
        index
        for feature in gpos.FeatureList.FeatureRecord
        if feature.FeatureTag == "kern"
        for index in feature.Feature.LookupListIndex
    }
    adjustment = 0
    for index in sorted(kern_lookups):
        lookup = gpos.LookupList.Lookup[index]
        if lookup.LookupType != 2:
            continue
        for subtable in lookup.SubTable:
            if left not in subtable.Coverage.glyphs:
                continue
            value = None
            if subtable.Format == 1:
                row = subtable.PairSet[subtable.Coverage.glyphs.index(left)]
                value = next((pair.Value1 for pair in row.PairValueRecord
                              if pair.SecondGlyph == right), None)
                if value is None:
                    continue
            elif subtable.Format == 2:
                first = subtable.ClassDef1.classDefs.get(left, 0)
                second = subtable.ClassDef2.classDefs.get(right, 0)
                value = subtable.Class1Record[first].Class2Record[second].Value1
            adjustment += (value.XAdvance or 0) if value else 0
            break
    return adjustment


def prepare_lockup(entries: list[dict[str, str]], theme: str, font: TTFont) -> None:
    master = f"logo/strata-lockup-{theme}.svg"
    output = BRAND / f"strata-lockup-{theme}.svg"
    output.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" width="182" height="48" '
        'viewBox="0 0 182 48" role="img">\n'
        f"{art_rectangles(master)}\n"
        f'<g fill="{INK[theme]}">{outlined_wordmark(font)}</g>\n'
        "</svg>\n",
        newline="\n",
    )
    record(entries, master, output, f"outlined GT Walsheim Pro Bold from {FONT.relative_to(APP)}; {link_note(theme)}")


def prepare_favicon(entries: list[dict[str, str]]) -> None:
    master = "web/favicon.svg"
    root = ET.parse(source(master)).getroot()
    rects = root.findall(f"{{{SVG_NS}}}rect")
    if len(rects) != 3:
        raise ValueError("Expected the supplied three-bar favicon")
    output = PUBLIC / "favicon.svg"
    bars = "\n".join(
        f'<rect class="{r.attrib["class"]}" x="{r.attrib["x"]}" '
        f'y="{r.attrib["y"]}" width="{r.attrib["width"]}" '
        f'height="{r.attrib["height"]}" rx="{r.attrib["rx"]}" '
        f'fill="{INK["light"] if r.attrib["class"] == "i" else GREEN["light"]}" />'
        for r in rects
    )
    output.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" '
        'viewBox="3 3 42 42">\n'
        '<style>@media (prefers-color-scheme: dark) {'
        f'.i {{ fill: {INK["dark"]}; }} .a {{ fill: {GREEN["dark"]}; }}'
        '}</style>\n'
        f'{bars}\n</svg>\n',
        newline="\n",
    )
    record(entries, master, output, f"explicit light fills and self-contained dark media style; {link_note('light')} (dark media style {UPSTREAM_GREEN['dark']} -> {GREEN['dark']})")


def prepare_manifest(entries: list[dict[str, str]]) -> None:
    master = "web/site.webmanifest"
    manifest = json.loads(source(master).read_text())
    manifest.update({"id": "/", "start_url": "/#/chat", "scope": "/"})
    output = PUBLIC / "site.webmanifest"
    output.write_text(json.dumps(manifest, indent=2) + "\n", newline="\n")
    record(entries, master, output, "root-local app identity and hash-route launch")


def main() -> None:
    BRAND.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, str]] = []
    for theme in ("light", "dark"):
        prepare_mark(entries, theme)
    font = TTFont(FONT)
    for theme in ("light", "dark"):
        prepare_lockup(entries, theme, font)
    prepare_favicon(entries)
    for name in (
        "favicon-16.png", "favicon-32.png", "favicon-48.png",
        "favicon-dark-16.png", "favicon-dark-32.png", "favicon-dark-48.png",
        "apple-touch-icon.png", "icon-192.png", "icon-512.png",
    ):
        copy(entries, f"web/{name}", PUBLIC / name)
        theme = "light" if name.startswith("favicon-") and "dark" not in name else "dark"  # the tile icons are dark-tile art
        link_png(PUBLIC / name, theme)
        entries[-1].update(output_sha256=digest(PUBLIC / name), method=f"copy of the upstream master; {link_note(theme)}; C2PA chunk removed (its hash covered the upstream bytes)")
    prepare_manifest(entries)
    (BRAND / "source-manifest.json").write_text(json.dumps({
        "source_root": str(MASTERS),
        "preparation": "scripts/prepare-brand-assets.py",
        "font": str(FONT.relative_to(APP)),
        "font_sha256": digest(FONT),
        "note": "Derived SVGs omit source C2PA metadata; original masters remain untouched and each source hash is recorded here. Strata Link recolours the middle bar of the mark to cyan so it is not mistaken for upstream Strata's; source hashes are the upstream masters', output hashes are of the recoloured files.",
        "files": entries,
    }, indent=2) + "\n", newline="\n")
    print(f"Prepared {len(entries)} web assets in {PUBLIC}")


if __name__ == "__main__":
    main()
