import math

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from .compairing_trace import CompairingTrace, ComparisonStatus


class CompairingTraceVisualizer:
    """Plot stored comparison statistics without running the model again."""

    @staticmethod
    def plot_metrics(
        compairing_trace: CompairingTrace,
        metric_names: str | list[str],
        *,
        show: bool = True,
        log_scale: bool = True,
        title: str | None = None,
    ) -> dict[str, Figure]:
        """Create one scatter plot per metric, with FWD/BWD in activation order.

        Supported metrics: max_abs, mean_abs, rmse, rel_l2, abs_l2 and
        compared_elements. Missing values are omitted, not plotted as zeros.
        FWD circles and metric diamonds encode status by color (OK green,
        DIFF red, other statuses gray). BWD squares are blue, hollow for OK
        and filled for DIFF; other BWD statuses are gray.
        log_scale uses symlog with a linear region below the smallest positive
        plotted value, preserving exact zeros. False selects a linear Y axis.
        The scalar metric comparison is appended as a diamond at the final tick.
        FWD/BWD maxima have dashed reference lines and labels beside the points.
        The legend and final metric summary sit above the plot.
        Ties mark the first maximum.
        title supplies an optional heading (e.g. the model name) for each figure.
        Returned figures can be saved with ``fig.savefig(...)``.
        """
        labels = {
            "max_abs": "Maximum absolute difference",
            "mean_abs": "Mean absolute difference",
            "rmse": "Root mean square error (RMSE)",
            "rel_l2": "Relative L2 difference",
            "abs_l2": "Absolute L2 difference",
            "compared_elements": "Compared elements",
        }
        metrics = [metric_names] if isinstance(metric_names, str) else list(metric_names)
        if not metrics:
            raise ValueError("Select at least one metric to plot")
        for metric in metrics:
            if metric not in labels:
                raise ValueError(f"Unknown metric {metric!r}; choose from {', '.join(labels)}")

        names = list(compairing_trace.activations)
        if not names and compairing_trace.metric is None:
            raise ValueError("The trace has no hook points or metric to plot")

        figures = {}
        tick_names = names + (["metric"] if compairing_trace.metric is not None else [])
        positions = list(range(len(tick_names)))
        has_gradients = any(name in compairing_trace.gradients for name in names)
        series = [("FWD", dict(enumerate(compairing_trace.activations.values())), "o", -0.1 if has_gradients else 0)]
        if has_gradients:
            series.append(("BWD", {
                index: compairing_trace.gradients[name]
                for index, name in enumerate(names) if name in compairing_trace.gradients
            }, "s", 0.1))
        if compairing_trace.metric is not None:
            series.append(("METRIC", {len(names): compairing_trace.metric}, "D", 0))

        for metric in dict.fromkeys(metrics):
            fig, (header, ax) = plt.subplots(
                2, 1, height_ratios=(0.22, 1),
                figsize=(max(10, len(tick_names) * 0.42), 8), layout="constrained"
            )
            header.set_axis_off()
            figures[metric] = fig
            fig.set_facecolor("white")
            if title:
                fig.suptitle(title, fontsize=17, fontweight="bold")
            ax.set_facecolor("#f8fafc")
            available = False
            smallest_positive = math.inf
            for label, rows, marker, offset in series:
                # Separate coincident FWD/BWD points while keeping one tick per hook.
                points_by_status = {}
                peak = None
                for position, row in rows.items():
                    value = getattr(row, metric)
                    if value is None or not math.isfinite(value):
                        continue
                    xs, ys = points_by_status.setdefault(row.status, ([], []))
                    xs.append(position + offset)
                    ys.append(float(value))
                    if peak is None or value > peak[1]:
                        peak = (position + offset, float(value), row.status)
                    if value > 0:
                        smallest_positive = min(smallest_positive, float(value))
                available |= bool(points_by_status)
                for status, (xs, ys) in points_by_status.items():
                    color = {
                        ComparisonStatus.OK: "#16a34a",
                        ComparisonStatus.DIFF: "#dc2626",
                    }.get(status, "#64748b")
                    if label == "BWD" and status in (ComparisonStatus.OK, ComparisonStatus.DIFF):
                        color = "#2563eb"
                    facecolor = "none" if label == "BWD" and status == ComparisonStatus.OK else color
                    ax.scatter(
                        xs, ys, label=f"{label} · {status}",
                        facecolors=facecolor, edgecolors=color, linewidths=1.2,
                        marker=marker, s=32, alpha=0.85, zorder=3, clip_on=False,
                    )

                if label in ("FWD", "BWD") and peak is not None:
                    peak_x, peak_y, peak_status = peak
                    peak_color = "#2563eb" if label == "BWD" else {
                        ComparisonStatus.OK: "#16a34a",
                        ComparisonStatus.DIFF: "#dc2626",
                    }.get(peak_status, "#64748b")
                    ax.axhline(
                        peak_y, color=peak_color, linewidth=1.1, alpha=0.7,
                        linestyle=(0, (3, 3)) if label == "FWD" else (3, (3, 3)),
                        zorder=2, clip_on=False,
                    )
                    ax.scatter(
                        [peak_x], [peak_y], marker=marker, s=90,
                        facecolors="none", edgecolors=peak_color, linewidths=1.4,
                        zorder=4, clip_on=False,
                    )
                    ax.annotate(
                        f"{label} max = {peak_y:.3e}", xy=(peak_x, peak_y),
                        xytext=(0, 14 if label == "FWD" else 34),
                        textcoords="offset points", ha="center", va="bottom",
                        fontsize=9, fontweight="bold", color=peak_color,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                                  edgecolor=peak_color, alpha=0.9),
                        arrowprops=dict(arrowstyle="-", color=peak_color, linewidth=0.8),
                        annotation_clip=False,
                    )

            metric_row = compairing_trace.metric
            if metric_row is not None:
                value = getattr(metric_row, metric)
                value_text = "N/A" if value is None else f"{value:.3e}"
                summary = f"METRIC · {metric_row.status}\n{metric} = {value_text}"
                if metric_row.difference is not None and metric_row.difference.numel() == 1:
                    summary += f"\nΔ (candidate − reference) = {metric_row.difference.item():+.3e}"
                elif metric_row.abs_l2 is not None:
                    summary += f"\n|Δ| = {metric_row.abs_l2:.3e}"
                header.text(
                    0.99, 0.5, summary, transform=header.transAxes,
                    ha="right", va="center", fontsize=9,
                    color={ComparisonStatus.OK: "#16a34a", ComparisonStatus.DIFF: "#dc2626"}.get(
                        metric_row.status, "#64748b"
                    ),
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#cbd5e1"),
                )

            header.set_title(labels[metric], loc="left", fontsize=15, fontweight="bold", pad=12)
            ax.set_ylabel(f"{metric} (symlog)" if log_scale else metric, fontsize=11)
            ax.set_xlabel("Hook points · activation trace order · final metric", labelpad=12)
            ax.set_xticks(positions, labels=tick_names, rotation=65, ha="right", fontsize=8)
            ax.text(
                0.99, 0.02,
                f"atol={compairing_trace.atol:g}\nrtol={compairing_trace.rtol:g}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
                color="#475569", bbox=dict(facecolor="white", edgecolor="#cbd5e1", alpha=0.9),
            )
            ax.tick_params(axis="both", length=0, pad=6)
            if log_scale:
                threshold = smallest_positive if math.isfinite(smallest_positive) else 1e-8
                ax.set_yscale("symlog", linthresh=threshold)
            else:
                ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3), useOffset=False)
            ax.grid(axis="y", color="#cbd5e1", alpha=0.6, linewidth=0.7)
            ax.set_axisbelow(True)
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.margins(x=0.02, y=0.2)
            ax.set_ylim(bottom=0)
            if available:
                handles, legend_labels = ax.get_legend_handles_labels()
                header.legend(
                    handles, legend_labels, frameon=False, loc="center left",
                    ncol=2, fontsize=9,
                )
            if not available:
                ax.text(
                    0.5, 0.5, "No comparable values", transform=ax.transAxes,
                    ha="center", va="center", color="#64748b",
                )

        if show:
            plt.show()
        return figures
