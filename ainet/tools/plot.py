"""Build clean Plotly specs for chat (2D/3D, data or equations)."""

from __future__ import annotations

import ast
import math
import re
from typing import Any

_MAX_SERIES = 8
_MAX_POINTS = 800
_MAX_GRID = 60
_TITLE_MAX = 120

_CHARTS_2D = frozenset(
    {
        "line",
        "scatter",
        "bar",
        "area",
        "histogram",
        "box",
        "pie",
        "heatmap",
        "contour",
    }
)
_CHARTS_3D = frozenset({"surface", "scatter3d", "line3d", "mesh3d", "isosurface"})
_CHARTS = _CHARTS_2D | _CHARTS_3D
_MAX_ISO_GRID = 36
_EQ_MAX = 500

_MATH_NAMES: dict[str, Any] = {
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau,
    "abs": abs,
    "min": min,
    "max": max,
    "round": round,
    "pow": pow,
}
for _name in (
    "sin",
    "cos",
    "tan",
    "asin",
    "acos",
    "atan",
    "atan2",
    "sinh",
    "cosh",
    "tanh",
    "exp",
    "log",
    "log10",
    "log2",
    "sqrt",
    "ceil",
    "floor",
    "fabs",
    "degrees",
    "radians",
    "hypot",
    "erf",
    "erfc",
    "gamma",
    "lgamma",
):
    _MATH_NAMES[_name] = getattr(math, _name)

# Soft engineering palette — readable on light chat cards
_COLORS = (
    "#0c8f55",
    "#2f6fed",
    "#c2472c",
    "#b7791f",
    "#6b4ea3",
    "#0f766e",
    "#be185d",
    "#334155",
)


class _SafeEval(ast.NodeVisitor):
    """Allow only math expressions over x/y (and constants)."""

    _ALLOWED_NODES = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Call,
        ast.Name,
        ast.Load,
        ast.Constant,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.FloorDiv,
        ast.Mod,
        ast.Pow,
        ast.USub,
        ast.UAdd,
        ast.Compare,
        ast.Eq,
        ast.NotEq,
        ast.Lt,
        ast.LtE,
        ast.Gt,
        ast.GtE,
        ast.IfExp,
        ast.And,
        ast.Or,
        ast.BoolOp,
        ast.Not,
    )

    def __init__(self, allowed_vars: set[str]) -> None:
        self.allowed_vars = allowed_vars

    def visit(self, node: ast.AST) -> Any:
        if not isinstance(node, self._ALLOWED_NODES):
            raise ValueError(f"Disallowed expression: {type(node).__name__}")
        return super().visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id not in self.allowed_vars and node.id not in _MATH_NAMES:
            raise ValueError(f"Unknown name in equation: {node.id}")

    def visit_Call(self, node: ast.Call) -> None:
        if not isinstance(node.func, ast.Name) or node.func.id not in _MATH_NAMES:
            raise ValueError("Only math functions are allowed in equations")
        for arg in node.args:
            self.visit(arg)
        for kw in node.keywords:
            self.visit(kw.value)


def _clip(text: str, limit: int) -> str:
    s = re.sub(r"\s+", " ", (text or "").strip())
    if len(s) <= limit:
        return s
    return s[: max(0, limit - 1)] + "…"


def _as_float_list(raw: Any, *, limit: int = _MAX_POINTS) -> list[float]:
    if raw is None:
        return []
    if isinstance(raw, str):
        parts = re.split(r"[\s,;]+", raw.strip())
        out: list[float] = []
        for p in parts:
            if not p:
                continue
            out.append(float(p))
            if len(out) >= limit:
                break
        return out
    if not isinstance(raw, (list, tuple)):
        raise ValueError("series values must be a list of numbers")
    out = []
    for item in raw[:limit]:
        out.append(float(item))
    return out


def _normalize_equation(expr: str) -> tuple[str, bool]:
    """Turn LaTeX / casual math into a Python expression.

    Returns (expression, is_level_set) where is_level_set means F(...)=0 style.
    """
    s = (expr or "").strip()
    if not s:
        return "", False
    # Drop surrounding math delimiters
    if (s.startswith("$$") and s.endswith("$$")) or (s.startswith("$") and s.endswith("$")):
        s = s.strip("$").strip()
    # Common LaTeX wrappers
    s = re.sub(r"\\mathrm\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    # \frac{a}{b} -> ((a)/(b)) — repeat for nesting
    for _ in range(8):
        nxt = re.sub(
            r"\\frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}",
            r"((\1)/(\2))",
            s,
        )
        if nxt == s:
            break
        s = nxt
    s = re.sub(r"\^\{([^}]+)\}", r"**(\1)", s)
    s = re.sub(r"\^(\d+)", r"**\1", s)
    s = s.replace(r"\left", "").replace(r"\right", "")
    s = s.replace(r"\cdot", "*").replace(r"\times", "*").replace(r"\ast", "*")
    s = s.replace(r"\,", "").replace(r"\;", "").replace(r"\!", "")
    # Strip remaining unknown LaTeX commands (\alpha etc. leave name if simple)
    s = re.sub(r"\\([a-zA-Z]+)", r"\1", s)
    s = s.replace("{", "(").replace("}", ")")
    s = s.replace("^", "**")
    # Unicode superscripts / operators
    for a, b in (
        ("²", "**2"),
        ("³", "**3"),
        ("⁴", "**4"),
        ("×", "*"),
        ("·", "*"),
        ("−", "-"),
        ("–", "-"),
    ):
        s = s.replace(a, b)
    s = re.sub(r"\s+", "", s)
    level = False
    # F(x,y,z)=0 or F=c  → use left-hand side (shift constant for =c)
    m = re.fullmatch(r"(.+?)=(.+)", s)
    if m:
        left, right = m.group(1), m.group(2)
        level = True
        if right in {"0", "0.0"}:
            s = left
        else:
            s = f"({left})-({right})"
    # Implicit multiply: 2x, 8(x, )( 
    s = re.sub(r"(\d)([xyz])\b", r"\1*\2", s)
    s = re.sub(r"(\))(\()", r"\1*\2", s)
    s = re.sub(r"(\d)\(", r"\1*(", s)
    s = re.sub(r"([xyz])\(", r"\1*(", s)
    return s, level


def _compile_equation(expr: str, variables: set[str]) -> Any:
    text, _level = _normalize_equation(expr)
    if not text:
        raise ValueError("equation is empty")
    if len(text) > _EQ_MAX:
        raise ValueError("equation is too long")
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Invalid equation: {exc.msg}") from exc
    _SafeEval(variables).visit(tree)
    code = compile(tree, "<ainet-plot>", "eval")
    return code


def _equation_uses_z(expr: str) -> bool:
    text, _ = _normalize_equation(expr)
    return bool(re.search(r"\bz\b", text))


def _eval_eq(code: Any, **vars_: float) -> float:
    env = dict(_MATH_NAMES)
    env.update(vars_)
    val = eval(code, {"__builtins__": {}}, env)  # noqa: S307 — AST-gated
    return float(val)


def _linspace(a: float, b: float, n: int) -> list[float]:
    n = max(2, min(_MAX_POINTS, int(n)))
    if n == 1:
        return [a]
    step = (b - a) / (n - 1)
    return [a + i * step for i in range(n)]


def _sample_1d(equation: str, x_min: float, x_max: float, n: int) -> tuple[list[float], list[float]]:
    code = _compile_equation(equation, {"x"})
    xs = _linspace(x_min, x_max, n)
    ys: list[float] = []
    for x in xs:
        try:
            ys.append(_eval_eq(code, x=x))
        except Exception as exc:
            raise ValueError(f"Equation failed at x={x}: {exc}") from exc
    return xs, ys


def _sample_2d(
    equation: str,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    n: int,
) -> tuple[list[float], list[float], list[list[float]]]:
    code = _compile_equation(equation, {"x", "y"})
    n = max(8, min(_MAX_GRID, int(n)))
    xs = _linspace(x_min, x_max, n)
    ys = _linspace(y_min, y_max, n)
    z: list[list[float]] = []
    for y in ys:
        row: list[float] = []
        for x in xs:
            try:
                row.append(_eval_eq(code, x=x, y=y))
            except Exception as exc:
                raise ValueError(f"Equation failed at ({x},{y}): {exc}") from exc
        z.append(row)
    return xs, ys, z


def _sample_isosurface(
    equation: str,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    z_min: float,
    z_max: float,
    n: int,
) -> dict[str, list[float]]:
    """Sample F(x,y,z) on a regular grid for Plotly isosurface."""
    code = _compile_equation(equation, {"x", "y", "z"})
    n = max(10, min(_MAX_ISO_GRID, int(n)))
    xs = _linspace(x_min, x_max, n)
    ys = _linspace(y_min, y_max, n)
    zs = _linspace(z_min, z_max, n)
    flat_x: list[float] = []
    flat_y: list[float] = []
    flat_z: list[float] = []
    flat_v: list[float] = []
    for z in zs:
        for y in ys:
            for x in xs:
                try:
                    v = _eval_eq(code, x=x, y=y, z=z)
                except Exception as exc:
                    raise ValueError(f"Equation failed at ({x},{y},{z}): {exc}") from exc
                if not math.isfinite(v):
                    continue
                flat_x.append(x)
                flat_y.append(y)
                flat_z.append(z)
                flat_v.append(v)
    if len(flat_v) < 8:
        raise ValueError("Isosurface sampling produced too few finite points")
    return {"x": flat_x, "y": flat_y, "z": flat_z, "value": flat_v}


def _palette(i: int) -> str:
    return _COLORS[i % len(_COLORS)]


def _layout(
    *,
    title: str,
    xlab: str,
    ylab: str,
    zlab: str,
    chart: str,
    animate: bool,
) -> dict[str, Any]:
    is3d = chart in _CHARTS_3D
    font = {"family": "Figtree, Helvetica, sans-serif", "size": 13, "color": "#1d2620"}
    axis = {
        "showgrid": True,
        "gridcolor": "rgba(18,20,18,0.06)",
        "zeroline": True,
        "zerolinecolor": "rgba(18,20,18,0.18)",
        "linecolor": "rgba(18,20,18,0.22)",
        "tickfont": {"size": 11, "color": "#6a726c"},
        "title": {"font": {"size": 12, "color": "#6a726c"}},
    }
    layout: dict[str, Any] = {
        "title": {
            "text": title,
            "font": {"family": "Syne, Figtree, sans-serif", "size": 18, "color": "#121412"},
            "x": 0.02,
            "xanchor": "left",
        },
        "font": font,
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "#fbfcf9",
        "margin": {"l": 56, "r": 18, "t": 52 if title else 28, "b": 48},
        "showlegend": True,
        "legend": {
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "x": 0,
            "font": {"size": 11},
            "bgcolor": "rgba(0,0,0,0)",
        },
        "hovermode": "closest",
        "ainet_animate": bool(animate),
        "ainet_chart": chart,
    }
    if is3d:
        layout["scene"] = {
            "xaxis": {**axis, "title": xlab or "x"},
            "yaxis": {**axis, "title": ylab or "y"},
            "zaxis": {**axis, "title": zlab or "z"},
            "bgcolor": "#fbfcf9",
            "camera": {"eye": {"x": 1.45, "y": 1.45, "z": 1.15}},
        }
        layout["margin"] = {"l": 8, "r": 8, "t": 52 if title else 24, "b": 8}
        layout["height"] = 420
    else:
        layout["xaxis"] = {**axis, "title": xlab or ""}
        layout["yaxis"] = {**axis, "title": ylab or ""}
        layout["height"] = 360
        if chart == "pie":
            layout["showlegend"] = True
            layout["margin"] = {"l": 20, "r": 20, "t": 52 if title else 28, "b": 20}
    return layout


def _series_trace(
    chart: str,
    series: dict[str, Any],
    index: int,
) -> dict[str, Any]:
    name = _clip(str(series.get("name") or f"Series {index + 1}"), 64)
    color = str(series.get("color") or _palette(index))
    equation = str(series.get("equation") or series.get("expr") or "").strip()
    mode = str(series.get("mode") or "").strip().lower()

    x = _as_float_list(series.get("x"))
    y = _as_float_list(series.get("y"))
    z = series.get("z")

    x_min = float(series.get("x_min", series.get("xmin", -5)))
    x_max = float(series.get("x_max", series.get("xmax", 5)))
    y_min = float(series.get("y_min", series.get("ymin", -5)))
    y_max = float(series.get("y_max", series.get("ymax", 5)))
    z_min = float(series.get("z_min", series.get("zmin", -5)))
    z_max = float(series.get("z_max", series.get("zmax", 5)))
    n = int(series.get("n") or series.get("points") or (36 if chart in _CHARTS_3D else 200))
    n = max(8, min(_MAX_POINTS, n))

    # Implicit F(x,y,z)=0 → Plotly isosurface (also when chart=surface but eq uses z)
    if equation and (
        chart in {"isosurface", "implicit"}
        or (chart == "surface" and _equation_uses_z(equation))
    ):
        field = _sample_isosurface(
            equation,
            x_min,
            x_max,
            y_min,
            y_max,
            z_min,
            z_max,
            min(n, _MAX_ISO_GRID),
        )
        return {
            "type": "isosurface",
            "name": name,
            "x": field["x"],
            "y": field["y"],
            "z": field["z"],
            "value": field["value"],
            "isomin": 0,
            "isomax": 0,
            "surface": {"count": 1, "fill": 0.92, "pattern": "odd"},
            "caps": {"x": {"show": False}, "y": {"show": False}, "z": {"show": False}},
            "colorscale": [
                [0.0, "#e8f5ee"],
                [0.45, "#5fb892"],
                [1.0, "#0a4f32"],
            ],
            "showscale": False,
            "opacity": 0.95,
            "hovertemplate": "%{x:.3g}, %{y:.3g}, %{z:.3g}<extra>" + name + "</extra>",
        }

    if equation and chart in {"line", "scatter", "area"}:
        x, y = _sample_1d(equation, x_min, x_max, n)
    elif equation and chart in {"surface", "contour", "heatmap"}:
        xs, ys, zz = _sample_2d(equation, x_min, x_max, y_min, y_max, min(n, _MAX_GRID))
        if chart == "surface":
            return {
                "type": "surface",
                "name": name,
                "x": xs,
                "y": ys,
                "z": zz,
                "colorscale": [
                    [0.0, "#e8f5ee"],
                    [0.35, "#7bc4a0"],
                    [0.7, "#0c8f55"],
                    [1.0, "#0a4f32"],
                ],
                "showscale": True,
                "opacity": 0.96,
                "hovertemplate": "%{x:.3g}, %{y:.3g}, %{z:.3g}<extra>" + name + "</extra>",
            }
        return {
            "type": "heatmap" if chart == "heatmap" else "contour",
            "name": name,
            "x": xs,
            "y": ys,
            "z": zz,
            "colorscale": [
                [0.0, "#f7f8f5"],
                [0.4, "#9ec9ff"],
                [0.75, "#2f6fed"],
                [1.0, "#173a8a"],
            ],
            "hovertemplate": "%{x:.3g}, %{y:.3g}, %{z:.3g}<extra>" + name + "</extra>",
            **(
                {"contours": {"coloring": "heatmap"}}
                if chart == "contour"
                else {}
            ),
        }

    if chart == "histogram":
        vals = y or x
        if not vals:
            raise ValueError(f"Series '{name}' needs values for histogram")
        return {
            "type": "histogram",
            "name": name,
            "x": vals,
            "marker": {"color": color, "line": {"width": 0}},
            "opacity": 0.88,
        }

    if chart == "box":
        vals = y or x
        if not vals:
            raise ValueError(f"Series '{name}' needs values for box")
        return {
            "type": "box",
            "name": name,
            "y": vals,
            "marker": {"color": color},
            "boxmean": True,
            "line": {"color": color},
        }

    if chart == "pie":
        labels = series.get("labels") or series.get("x") or []
        if isinstance(labels, str):
            labels = [p for p in re.split(r"[\s,;]+", labels) if p]
        values = y or _as_float_list(series.get("values"))
        if not labels or not values:
            raise ValueError(f"Series '{name}' needs labels and values for pie")
        return {
            "type": "pie",
            "name": name,
            "labels": [str(v) for v in labels[:_MAX_POINTS]],
            "values": values[:_MAX_POINTS],
            "hole": 0.35,
            "textinfo": "label+percent",
            "marker": {
                "colors": [_palette(i) for i in range(len(values[:_MAX_POINTS]))],
                "line": {"color": "#fff", "width": 1.5},
            },
        }

    if chart in {"heatmap", "contour"} and not equation:
        if isinstance(z, list) and z and isinstance(z[0], (list, tuple)):
            zz = [[float(v) for v in row[:_MAX_GRID]] for row in z[:_MAX_GRID]]
        else:
            raise ValueError(f"Series '{name}' needs 2D z (or equation) for {chart}")
        return {
            "type": "heatmap" if chart == "heatmap" else "contour",
            "name": name,
            "x": x or None,
            "y": y or None,
            "z": zz,
            "colorscale": [
                [0.0, "#f7f8f5"],
                [0.4, "#9ec9ff"],
                [0.75, "#2f6fed"],
                [1.0, "#173a8a"],
            ],
        }

    if chart == "surface" and not equation:
        if isinstance(z, list) and z and isinstance(z[0], (list, tuple)):
            zz = [[float(v) for v in row[:_MAX_GRID]] for row in z[:_MAX_GRID]]
        else:
            raise ValueError(f"Series '{name}' needs 2D z (or equation) for surface")
        return {
            "type": "surface",
            "name": name,
            "x": x or None,
            "y": y or None,
            "z": zz,
            "colorscale": [
                [0.0, "#e8f5ee"],
                [0.35, "#7bc4a0"],
                [0.7, "#0c8f55"],
                [1.0, "#0a4f32"],
            ],
            "showscale": True,
        }

    if chart in {"scatter3d", "line3d", "mesh3d"}:
        zz = _as_float_list(z)
        if equation and "y" not in equation and "z" not in equation.lower():
            # z = f(x) extruded along a trivial y if only x given — prefer explicit data
            pass
        if not x or not y or not zz:
            # Allow equation z=f(x) with y=0 for a 3D curve
            eqz = str(series.get("equation_z") or equation).strip()
            if eqz and (not zz):
                x, zz = _sample_1d(eqz, x_min, x_max, n)
                y = [0.0] * len(x)
            else:
                raise ValueError(f"Series '{name}' needs x, y, z for {chart}")
        if chart == "mesh3d":
            return {
                "type": "mesh3d",
                "name": name,
                "x": x,
                "y": y,
                "z": zz,
                "opacity": 0.85,
                "color": color,
            }
        return {
            "type": "scatter3d",
            "name": name,
            "x": x,
            "y": y,
            "z": zz,
            "mode": "lines" if chart == "line3d" else (mode or "markers"),
            "marker": {"size": 3.5, "color": color},
            "line": {"width": 4, "color": color},
        }

    # Default 2D cartesian
    if equation:
        x, y = _sample_1d(equation, x_min, x_max, n)
    if not x and y:
        x = list(range(len(y)))
    if not y and x:
        raise ValueError(f"Series '{name}' needs y values or an equation")
    if len(x) != len(y):
        m = min(len(x), len(y))
        x, y = x[:m], y[:m]
    if not x:
        raise ValueError(f"Series '{name}' is empty")

    if chart == "bar":
        return {
            "type": "bar",
            "name": name,
            "x": x,
            "y": y,
            "marker": {"color": color, "line": {"width": 0}},
            "opacity": 0.92,
        }
    if chart == "area":
        # Soft fill under the curve
        fill = color
        if color.startswith("#") and len(color) == 7:
            r = int(color[1:3], 16)
            g = int(color[3:5], 16)
            b = int(color[5:7], 16)
            fill = f"rgba({r},{g},{b},0.18)"
        return {
            "type": "scatter",
            "name": name,
            "x": x,
            "y": y,
            "mode": "lines",
            "fill": "tozeroy",
            "line": {"color": color, "width": 2.4, "shape": "spline"},
            "fillcolor": fill,
        }
    if chart == "scatter":
        return {
            "type": "scatter",
            "name": name,
            "x": x,
            "y": y,
            "mode": mode or "markers",
            "marker": {"size": 8, "color": color, "line": {"width": 0}},
            "line": {"color": color, "width": 2},
        }
    # line (default)
    return {
        "type": "scatter",
        "name": name,
        "x": x,
        "y": y,
        "mode": mode or "lines",
        "line": {"color": color, "width": 2.8, "shape": "spline"},
        "marker": {"size": 5, "color": color},
    }


def create_plot(
    title: str = "",
    *,
    chart: str = "line",
    xlab: str = "",
    ylab: str = "",
    zlab: str = "",
    series: list[dict[str, Any]] | dict[str, Any] | None = None,
    x: Any = None,
    y: Any = None,
    z: Any = None,
    equation: str = "",
    x_min: float | None = None,
    x_max: float | None = None,
    y_min: float | None = None,
    y_max: float | None = None,
    z_min: float | None = None,
    z_max: float | None = None,
    n: int | None = None,
    animate: bool = True,
    source: str = "",
) -> dict[str, Any]:
    """Build a Plotly figure for the chat UI."""
    kind = (chart or "line").strip().lower().replace(" ", "")
    aliases = {
        "lines": "line",
        "plot": "line",
        "xy": "line",
        "points": "scatter",
        "scatterplot": "scatter",
        "bars": "bar",
        "column": "bar",
        "hist": "histogram",
        "surf": "surface",
        "3d": "scatter3d",
        "scatter3": "scatter3d",
        "line3": "line3d",
        "implicit": "isosurface",
        "levelset": "isosurface",
        "iso": "isosurface",
    }
    kind = aliases.get(kind, kind)

    # Auto-pick isosurface for F(x,y,z)=0 when the model still says "surface"
    eq_probe = equation or ""
    if not eq_probe and isinstance(series, dict):
        eq_probe = str(series.get("equation") or series.get("expr") or "")
    elif not eq_probe and isinstance(series, list) and series:
        first = series[0]
        if isinstance(first, dict):
            eq_probe = str(first.get("equation") or first.get("expr") or "")
    if eq_probe and _equation_uses_z(eq_probe) and kind in {"surface", "line", "plot", "3d"}:
        kind = "isosurface"

    if kind not in _CHARTS:
        raise ValueError(
            f"Unsupported chart '{chart}'. Use one of: {', '.join(sorted(_CHARTS))}"
        )

    rows: list[dict[str, Any]] = []
    if isinstance(series, dict):
        rows = [series]
    elif isinstance(series, list):
        rows = [s for s in series if isinstance(s, dict)]
    if not rows:
        # Convenience: top-level x/y/equation -> one series
        row: dict[str, Any] = {"name": "Series 1"}
        if x is not None:
            row["x"] = x
        if y is not None:
            row["y"] = y
        if z is not None:
            row["z"] = z
        if equation:
            row["equation"] = equation
        if x_min is not None:
            row["x_min"] = x_min
        if x_max is not None:
            row["x_max"] = x_max
        if y_min is not None:
            row["y_min"] = y_min
        if y_max is not None:
            row["y_max"] = y_max
        if z_min is not None:
            row["z_min"] = z_min
        if z_max is not None:
            row["z_max"] = z_max
        if n is not None:
            row["n"] = n
        # Sensible default box for this classic quartic-ish surface
        if kind == "isosurface" and z_min is None and z_max is None:
            row.setdefault("x_min", -5)
            row.setdefault("x_max", 5)
            row.setdefault("y_min", -5)
            row.setdefault("y_max", 5)
            row.setdefault("z_min", -5)
            row.setdefault("z_max", 5)
            row.setdefault("n", 32)
        rows = [row]

    if len(rows) > _MAX_SERIES:
        rows = rows[:_MAX_SERIES]

    traces: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        traces.append(_series_trace(kind, row, i))

    fig = {
        "data": traces,
        "layout": _layout(
            title=_clip(title, _TITLE_MAX),
            xlab=_clip(xlab, 64),
            ylab=_clip(ylab, 64),
            zlab=_clip(zlab, 64),
            chart=kind,
            animate=animate and kind not in {"pie", "box"},
        ),
    }
    out: dict[str, Any] = {
        "ok": True,
        "chart": kind,
        "title": _clip(title, _TITLE_MAX),
        "series_count": len(traces),
        "animate": bool(fig["layout"].get("ainet_animate")),
        "figure": fig,
        "summary": f"{kind} · {len(traces)} series",
    }
    if source:
        out["source"] = _clip(str(source), 200)
        out["summary"] += f" · {_clip(str(source), 80)}"
    return out
