# visualizer.py (interactive map)
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

def plot_interactive_map(prob_map, region_bounds, outfile="risk_map.html"):
    lon_min, lon_max, lat_min, lat_max = region_bounds
    fig = go.Figure(data=go.Heatmap(
        z=prob_map,
        zmin=0, zmax=prob_map.max(),
        colorscale='Hot',
        colorbar=dict(title="P(M>=X)"),
        x=np.linspace(lon_min, lon_max, prob_map.shape[1]),
        y=np.linspace(lat_min, lat_max, prob_map.shape[0])))
    fig.update_layout(
        title="Seismic Risk Map (interactive)",
        xaxis_title="Longitude", yaxis_title="Latitude")
    fig.write_html(outfile, include_plotlyjs='cdn')


def plot_static_map(prob_map: np.ndarray,
                    region_bounds: tuple,
                    title: str = "Seismic Risk Map",
                    outfile: str = "risk_map.png") -> None:
    """
    Статическая карта вероятностей (Matplotlib).

    prob_map       – 2D numpy массив вероятностей (y, x)
    region_bounds  – (lon_min, lon_max, lat_min, lat_max)
    title/outfile  – заголовок и имя png-файла
    """
    lon_min, lon_max, lat_min, lat_max = region_bounds
    extent = [lon_min, lon_max, lat_min, lat_max]

    plt.figure(figsize=(8, 6))
    plt.imshow(prob_map,
               extent=extent,
               origin="lower",
               cmap="hot_r",
               aspect="auto")
    plt.colorbar(label="P(M≥X)")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    plt.close()
