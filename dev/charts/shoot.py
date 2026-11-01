"""Turn the built charts into the images the documentation shows.

Every chart image in the docs is a screenshot of a document ``build.py`` produced, so
regenerating the set is one command. Nothing here crops or retouches: what the reader
sees is what the library drew, at the size it was drawn, once in the dark scheme and
once in the light. No chart is described here either, which is why the manifest exists;
this file only knows how to point a browser at one.
"""

from __future__ import annotations

import json
import pathlib
import sys

from playwright.sync_api import sync_playwright

CHARTS = pathlib.Path("/work/charts")

IMAGES = pathlib.Path("/out/imgs/charts")

SCALE = 2

SCHEMES = ("dark", "light")

# the settle is for the renderer, not for the page: lightweight-charts sizes
# itself from the container on the first frame and paints the primitives on the
# next, so a shot taken at load catches an empty canvas
SETTLE_MS = 2500


def shoot(browser, name, viewport, scheme, open_table, caption):
    # a saved chart opens on its reader's scheme, so one document gives both images
    page = browser.new_page(viewport=viewport, device_scale_factor=SCALE,
                            color_scheme=scheme)

    page.goto((CHARTS / f"{name}.html").as_uri())
    page.wait_for_timeout(SETTLE_MS)
    # a table of the caller's opens the panel by itself
    if open_table and page.locator("#tablePanel").is_hidden():
        page.click("#tbl")
        # opening the panel shrinks the plot above it, so the row that slides up
        # under the cursor would come out wearing its hover colour
        page.mouse.move(0, 0)
        page.wait_for_timeout(500)

    path = IMAGES / f"{name}-{scheme}.png"
    page.screenshot(path=str(path))
    page.close()

    width, height = viewport["width"], viewport["height"]
    print(f"{path.name:<30} {width}x{height} at {SCALE}x "
          f"{path.stat().st_size / 1000:>6.0f} kB   {caption}")


def main():
    manifest = CHARTS / "manifest.json"
    if not manifest.is_file():
        print(f"no {manifest}, so there is nothing to shoot; run build.py first")
        return 1

    shots   = json.loads(manifest.read_text(encoding="utf-8"))
    missing = sorted(name for name in shots
                     if not (CHARTS / f"{name}.html").is_file())
    if missing:
        print(f"the manifest names charts that were not built: {missing}")
        return 1

    IMAGES.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as play:
        browser = play.chromium.launch()
        for name in sorted(shots):
            shot     = shots[name]
            viewport = {"width": shot["width"],
                        "height": shot["height"] + shot.get("below", 0)}
            for scheme in SCHEMES:
                shoot(browser, name, viewport, scheme, shot.get("open"), shot["caption"])

        browser.close()

    print(f"\n{len(shots)} charts shot in {len(SCHEMES)} schemes into {IMAGES}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
