# %% [markdown]
# Selector de puntos sobre una imagen (versión .py)
# Ejecuta en Jupyter, VS Code o similar. Requiere backend interactivo para clics.
# Controles: click izq = agregar punto; s=guardar; u/z/backspace=deshacer; c=limpiar; q/enter/esc=cerrar.

# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Cursor
from PIL import Image
from io import BytesIO

# Backend interactivo (intento)
BACKEND = "inline"
try:
    import ipympl  # noqa: F401
    from IPython import get_ipython
    ip = get_ipython()
    if ip is not None:
        ip.run_line_magic("matplotlib", "widget")
        BACKEND = "widget"
except Exception:
    pass
if BACKEND != "widget":
    try:
        from IPython import get_ipython
        ip = get_ipython()
        if ip is not None:
            ip.run_line_magic("matplotlib", "notebook")
            BACKEND = "notebook"
    except Exception:
        BACKEND = "inline"
if BACKEND == "inline":
    print("⚠️ Estás en backend 'inline'. Los clics pueden no funcionar. Instala ipympl o usa %matplotlib notebook.")
else:
    print(f"✅ Backend interactivo activado: {BACKEND}")

# %%
IMG_PATH = ""  # Ruta a tu imagen (recomendado)
SAVE_BASENAME = "points"
ROUND_COORDS = True
SHOW_ANNOTATIONS = True

# %%
import os, csv

class PointPicker:
    def __init__(self, img, round_coords=True, annotations=True, save_basename="points"):
        self.img = img
        self.round_coords = round_coords
        self.annotations = annotations
        self.points = []
        self.texts = []
        self.save_basename = save_basename

        self.fig, self.ax = plt.subplots(constrained_layout=True)
        self.ax.imshow(self.img, origin="upper")
        self.ax.set_axis_off()
        self.ax.set_title("Haz clic para agregar puntos. Teclas: s=guardar, u=deshacer, c=limpiar, q/cerrar")

        self.sc = self.ax.scatter([], [], s=40, edgecolors="white", facecolors="none", linewidths=1.5)
        try:
            self.cursor = Cursor(self.ax, useblit=True, color="black", linewidth=1)
        except Exception:
            self.cursor = None

        self.cid_click = self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.cid_key   = self.fig.canvas.mpl_connect("key_press_event",   self.on_key)

        btn_h = 0.06; btn_w = 0.12
        ax_save  = self.fig.add_axes([0.80, 0.01, btn_w, btn_h]); self.btn_save  = Button(ax_save,  "Guardar")
        ax_undo  = self.fig.add_axes([0.67, 0.01, btn_w, btn_h]); self.btn_undo  = Button(ax_undo,  "Deshacer")
        ax_clear = self.fig.add_axes([0.54, 0.01, btn_w, btn_h]); self.btn_clear = Button(ax_clear, "Limpiar")
        self.btn_save.on_clicked(self._on_btn_save)
        self.btn_undo.on_clicked(self._on_btn_undo)
        self.btn_clear.on_clicked(self._on_btn_clear)

    def on_click(self, event):
        if event.inaxes != self.ax:
            return
        if event.button != 1:
            return
        if (event.xdata is None) or (event.ydata is None):
            return
        x, y = event.xdata, event.ydata
        if self.round_coords:
            x, y = int(round(x)), int(round(y))
        self.points.append((x, y))
        self._update_plot()

    def on_key(self, event):
        if event.key in ("u", "backspace", "z"):
            self.undo()
        elif event.key == "c":
            self.clear()
        elif event.key == "s":
            self.save()
        elif event.key in ("q", "enter", "escape"):
            plt.close(self.fig)

    def _on_btn_save(self, _event):
        self.save()

    def _on_btn_undo(self, _event):
        self.undo()

    def _on_btn_clear(self, _event):
        self.clear()

    def _update_plot(self):
        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]
        try:
            self.sc.remove()
        except Exception:
            pass
        self.sc = self.ax.scatter(xs, ys, s=40, edgecolors="white", facecolors="none", linewidths=1.5)
        for t in self.texts:
            try:
                t.remove()
            except Exception:
                pass
        self.texts.clear()
        if self.annotations:
            for i, (x, y) in enumerate(self.points, start=1):
                txt = self.ax.text(x+3, y+3, str(i), fontsize=8, color="white",
                                   bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.5))
                self.texts.append(txt)
        self.fig.canvas.draw_idle()

    def undo(self):
        if self.points:
            self.points.pop()
            self._update_plot()

    def clear(self):
        self.points.clear()
        self._update_plot()

    def save(self, basename=None):
        if basename is not None:
            self.save_basename = basename
        arr = np.array(self.points, dtype=(np.int32 if self.round_coords else np.float32))
        if arr.size == 0:
            print("No hay puntos que guardar.")
            return
        base, ext = os.path.splitext(self.save_basename)
        if ext.lower() not in (".csv", ".npy", ".json", ""):
            base = self.save_basename
        csv_path = base + ".csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["x", "y"])
            writer.writerows(arr.tolist())
        print(f"✅ Guardado {len(arr)} puntos en {csv_path}")
        npy_path = base + ".npy"
        np.save(npy_path, arr)
        print(f"✅ Guardado también en {npy_path} (NumPy)")
        try:
            import json
            json_path = base + ".json"
            with open(json_path, "w", encoding="utf-8") as jf:
                json.dump({"points": arr.tolist()}, jf, ensure_ascii=False, indent=2)
            print(f"✅ Guardado también en {json_path} (JSON)")
        except Exception as e:
            print("No fue posible guardar JSON:", e)

# %% Uso
if IMG_PATH:
    pil = Image.open(IMG_PATH).convert("RGB")
    img = np.array(pil)
    picker = PointPicker(img, round_coords=ROUND_COORDS, annotations=SHOW_ANNOTATIONS, save_basename=SAVE_BASENAME)
    plt.show()
else:
    print("Configura IMG_PATH con la ruta a tu imagen.")
