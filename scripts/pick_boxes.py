"""Click two corners per field on a template image and print `box: [x, y, w, h]` for each.

    uv run python scripts/pick_boxes.py forms/<form>/template.png                   # free picking: box_1, box_2, ...
    uv run python scripts/pick_boxes.py forms/<form>/template.png --form forms/<form>  # walk the spec's field ids

Click the top-left then the bottom-right corner of a box (any two opposite corners work). Keys: u = undo the
last box, s = skip the current field, q / Esc = finish. Coordinates are in the original image's pixels, even
though the window shows it scaled down to fit the screen. Paste the printed lines into form.yaml.
"""

from __future__ import annotations

import argparse
import tkinter as tk

from PIL import Image, ImageTk

from pocketform_distill.config import load_form


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("image")
    p.add_argument("--form", help="form folder; prompts for each field id in form.yaml order")
    p.add_argument("--max-size", type=int, default=1200, help="longest window side in screen pixels")
    args = p.parse_args()

    names = [f.id for f in load_form(args.form).fields] if args.form else []

    img = Image.open(args.image).convert("RGB")
    scale = min(1.0, args.max_size / max(img.size))
    shown = img.resize((round(img.width * scale), round(img.height * scale)))

    root = tk.Tk()
    canvas = tk.Canvas(root, width=shown.width, height=shown.height, cursor="crosshair")
    canvas.pack()
    photo = ImageTk.PhotoImage(shown)
    canvas.create_image(0, 0, image=photo, anchor="nw")
    status = tk.Label(root, anchor="w")
    status.pack(fill="x")

    done: list[tuple[str, tuple[int, int, int, int] | None, int | None]] = []  # name, box (None = skipped), item
    pending: list[tuple[float, float]] = []

    def current_name() -> str | None:
        i = len(done)
        if names:
            return names[i] if i < len(names) else None
        return f"box_{i + 1}"

    def refresh() -> None:
        name = current_name()
        status.config(text=f"Next: {name}  ({'2nd corner' if pending else '1st corner'})  u=undo s=skip q=done"
                      if name else "All fields done - press q")

    def click(event: tk.Event) -> None:
        name = current_name()
        if name is None:
            return
        pending.append((event.x, event.y))
        if len(pending) == 2:
            (x1, y1), (x2, y2) = pending
            pending.clear()
            x, y = round(min(x1, x2) / scale), round(min(y1, y2) / scale)
            w, h = round(abs(x2 - x1) / scale), round(abs(y2 - y1) / scale)
            item = canvas.create_rectangle(min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2), outline="red", width=2)
            done.append((name, (x, y, w, h), item))
            print(f"{name}: box: [{x}, {y}, {w}, {h}]", flush=True)
        refresh()

    def key(event: tk.Event) -> None:
        if event.keysym in ("q", "Escape"):
            root.destroy()
        elif event.keysym == "u" and done:
            pending.clear()
            name, _, item = done.pop()
            if item is not None:
                canvas.delete(item)
            print(f"# undone {name}", flush=True)
        elif event.keysym == "s" and current_name():
            pending.clear()
            done.append((current_name(), None, None))
        refresh()

    canvas.bind("<Button-1>", click)
    root.bind("<Key>", key)
    root.title(f"pick_boxes - {args.image}")
    refresh()
    root.mainloop()

    print("\n# --- final boxes ---")
    for name, box, _ in done:
        print(f"- id: {name}\n  box: [{', '.join(map(str, box))}]" if box else f"# skipped: {name}")


if __name__ == "__main__":
    main()
