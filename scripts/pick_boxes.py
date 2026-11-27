"""Mark where each field's box is on a section's blank template image, by clicking two corners.

    uv run python scripts/pick_boxes.py --form forms/<form> --section <id>           # print boxes
    uv run python scripts/pick_boxes.py --form forms/<form> --section <id> --write   # save into form.yaml
    uv run python scripts/pick_boxes.py some-image.png                              # any image: box_1, box_2, ...

The window opens that section's template image (or pass an image path to use another one). --section can be left
out when the form has only one section.

What to click: the area the handwriting goes in for the field named in the bar at the bottom. Click its top-left
corner, then its bottom-right corner. For a comb field (3 digit cells) take ALL the cells as one box: the
outer corners of the first and last cell. Synth splits it into cells itself.

On screen:
    orange dashed   where form.yaml currently says this field's box is (a hint; may be wrong)
    green           boxes you've done
    red             the box you're drawing

Keys: Enter = keep the orange box as it is, u = undo, s = skip, q / Esc = finish.
Coordinates are in the image's own pixels, even though the window shows it scaled to fit the screen.
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

import yaml
from PIL import Image, ImageOps

from pocketform_distill.config import FormField, FormSpec

Box = tuple[int, int, int, int]


def write_boxes(form_yaml: Path, boxes: dict[str, Box]) -> None:
    """Replace each field's `box: [...]` in form.yaml in place, keeping comments and layout (backup: .bak)."""
    text = form_yaml.read_text()
    for fid, (x, y, w, h) in boxes.items():
        # The first `box: [...]` after `id: <fid>` belongs to that field, in both `{id: a, box: [..]}` and
        # block style.
        pattern = re.compile(rf"(\bid:\s*{re.escape(fid)}\b.*?\bbox:\s*)\[[^\]]*\]", re.DOTALL)
        text, n = pattern.subn(rf"\g<1>[{x}, {y}, {w}, {h}]", text, count=1)
        if n != 1:
            raise SystemExit(f"could not find the box line for {fid} in {form_yaml}")
    shutil.copy(form_yaml, form_yaml.with_suffix(".yaml.bak"))
    form_yaml.write_text(text)


def main() -> None:
    import tkinter as tk  # imported here so write_boxes can be tested without a display

    from PIL import ImageTk

    p = argparse.ArgumentParser()
    p.add_argument("image", nargs="?", help="image to click on (default: the section's template image)")
    p.add_argument("--form", help="form folder: walk one section's fields in form.yaml order")
    p.add_argument("--section", help="section id (default: the form's only section)")
    p.add_argument("--write", action="store_true", help="save the boxes into form.yaml (backup in form.yaml.bak)")
    p.add_argument("--max-size", type=int, default=1200, help="longest window side in screen pixels")
    args = p.parse_args()
    if args.write and not args.form:
        p.error("--write needs --form")
    if not args.image and not args.form:
        p.error("give an image, or --form (and --section) to use a section's template")

    # Read form.yaml without load_form's checks: the old boxes may not fit a new template, which is exactly when
    # you re-pick them.
    fields: list[FormField] = []
    image = args.image
    if args.form:
        spec = FormSpec.model_validate(yaml.safe_load((Path(args.form) / "form.yaml").read_text()))
        try:
            section = spec.section(args.section)
        except ValueError as e:
            p.error(str(e))
        fields = section.fields
        image = image or str(Path(args.form) / section.template_image)

    with Image.open(image) as raw:
        img = ImageOps.exif_transpose(raw).convert("RGB")  # same upright orientation synth uses
    scale = min(1.0, args.max_size / max(img.size))
    shown = img.resize((round(img.width * scale), round(img.height * scale)))

    root = tk.Tk()
    root.title(f"pick_boxes - {image}")
    canvas = tk.Canvas(root, width=shown.width, height=shown.height, cursor="crosshair")
    canvas.pack()
    photo = ImageTk.PhotoImage(shown)
    canvas.create_image(0, 0, image=photo, anchor="nw")
    title = tk.Label(root, anchor="w", font=("Helvetica", 16, "bold"))
    title.pack(fill="x", padx=8)
    hint = tk.Label(root, anchor="w", font=("Helvetica", 12))
    hint.pack(fill="x", padx=8, pady=(0, 6))

    done: list[tuple[str, Box | None, int | None]] = []  # name, box (None = skipped), canvas item
    pending: list[tuple[float, float]] = []
    temp: list[int] = []  # canvas items redrawn on every refresh (hint box, rubber band)

    def current() -> tuple[str, FormField | None] | None:
        i = len(done)
        if fields:
            return (fields[i].id, fields[i]) if i < len(fields) else None
        return f"box_{i + 1}", None

    def screen(box: Box) -> tuple[float, float, float, float]:
        x, y, w, h = box
        return x * scale, y * scale, (x + w) * scale, (y + h) * scale

    def clear_temp() -> None:
        for item in temp:
            canvas.delete(item)
        temp.clear()

    def refresh() -> None:
        clear_temp()
        cur = current()
        if cur is None:
            title.config(text="All fields done")
            hint.config(text="Press q to finish" + (" and save into form.yaml" if args.write else ""))
            return
        name, f = cur
        n = f"{len(done) + 1}/{len(fields)}  " if fields else ""
        title.config(text=f"{n}{name}" + (f"  -  {f.label}" if f else ""))
        what = "comb box: outer corners of ALL cells" if f and f.cells else "the whole writing area"
        step = "click its BOTTOM-RIGHT corner" if pending else "click its TOP-LEFT corner"
        keys = "Enter = keep orange box, " if f else ""
        hint.config(text=f"{step} ({what})   |   {keys}u = undo, s = skip, q = finish")
        if f:
            temp.append(canvas.create_rectangle(*screen(f.box), outline="orange", width=2, dash=(6, 4)))

    def add(name: str, box: Box) -> None:
        item = canvas.create_rectangle(*screen(box), outline="green", width=2)
        done.append((name, box, item))
        print(f"{name}: box: [{', '.join(map(str, box))}]", flush=True)

    def click(event: tk.Event) -> None:
        cur = current()
        if cur is None:
            return
        pending.append((event.x, event.y))
        if len(pending) == 2:
            (x1, y1), (x2, y2) = pending
            pending.clear()
            add(cur[0], (round(min(x1, x2) / scale), round(min(y1, y2) / scale),
                         round(abs(x2 - x1) / scale), round(abs(y2 - y1) / scale)))
        refresh()

    def motion(event: tk.Event) -> None:
        if pending:
            refresh()
            (x1, y1), = pending
            temp.append(canvas.create_rectangle(x1, y1, event.x, event.y, outline="red", width=2))

    def key(event: tk.Event) -> None:
        cur = current()
        if event.keysym in ("q", "Escape"):
            root.destroy()
            return
        if event.keysym == "Return" and cur and cur[1] is not None:
            pending.clear()
            add(cur[0], cur[1].box)
        elif event.keysym == "u" and done:
            pending.clear()
            name, _, item = done.pop()
            if item is not None:
                canvas.delete(item)
            print(f"# undone {name}", flush=True)
        elif event.keysym == "s" and cur:
            pending.clear()
            done.append((cur[0], None, None))
            print(f"# skipped {cur[0]}", flush=True)
        refresh()

    canvas.bind("<Button-1>", click)
    canvas.bind("<Motion>", motion)
    root.bind("<Key>", key)
    refresh()
    root.mainloop()

    picked = {name: box for name, box, _ in done if box is not None}
    print("\n# --- final boxes ---")
    for name, box, _ in done:
        print(f"- id: {name}\n  box: [{', '.join(map(str, box))}]" if box else f"# skipped: {name}")
    if args.write and picked:
        write_boxes(Path(args.form) / "form.yaml", picked)
        print(f"# saved {len(picked)} boxes into {args.form}/form.yaml (previous version: form.yaml.bak)")


if __name__ == "__main__":
    main()
