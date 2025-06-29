import marimo

__generated_with = "0.14.9"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import plotly.express as px
    import glob
    import os
    return glob, mo, os, pd, px


@app.cell
def _(mo):
    mo.md("""# Benchmark Results Viewer""")
    return


@app.cell
def _(glob, mo, os):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_files = glob.glob(os.path.join(current_dir, '*.csv'))

    if not csv_files:
        mo.md("No benchmark CSV files found in this directory.")

    # Create a mapping from a user-friendly name to the file path
    csv_options = {os.path.basename(f): f for f in csv_files}

    csv_selector = mo.ui.dropdown(
        options=csv_options,
        label="Select Benchmark CSV:",
        #value=csv_options.values(),
    )
    csv_selector
    return (csv_selector,)


@app.cell
def _(csv_selector, pd):
    if csv_selector.value:
        df = pd.read_csv(csv_selector.value)
    else:
        df = pd.DataFrame()
    return (df,)


@app.cell
def _(df, mo):
    assert not df.empty

    # Identify columns to create filters for (exclude known non-filter columns)
    cols_to_filter = [
        col for col in df.columns 
        if col not in ['value', 'measurement'] and df[col].dtype == 'object'
    ]
    print(cols_to_filter)

    # The x-axis is likely the numeric column that isn't 'value'
    x_axis_col = next((col for col in df.columns if df[col].dtype != 'object' and col != 'value'), None)
    print(x_axis_col)

    assert cols_to_filter

    filters = {
        col: mo.ui.multiselect(
            df[col].unique().tolist(), 
            label=f"Filter {col}", 
        )
        for col in cols_to_filter
    }
    print(filters)

    filters_form = mo.md("{activation} {dtype} {competitor}").batch(**filters).form(show_clear_button=True)
    return filters_form, x_axis_col


@app.cell
def _(filters_form):
    filters_form
    return


@app.cell
def _(df, filters_form):
    filtered_df = df.copy()
    for col, control in filters_form.value.items():
        if control is not None:
            filtered_df = filtered_df[filtered_df[col].isin(control)]
    #filtered_df
    return (filtered_df,)


@app.cell
def _(filtered_df, mo, px, x_axis_col):
    if filtered_df.empty or not x_axis_col:
        mo.md("No data to plot. Adjust filters or select a different CSV.")

    series_cols = [
        col
        for col in filtered_df.columns
        if col not in ["value", "measurement"] and filtered_df[col].dtype == "object"
    ]

    plot_df = filtered_df.copy()
    if series_cols:
        plot_df["series"] = plot_df[series_cols].apply(
            lambda row: "-".join(row.values.astype(str)), axis=1
        )
        color_arg = "series"
    else:
        color_arg = None

    color_discrete_map = None
    if color_arg:
        series_names = plot_df[color_arg].unique()
        colors = px.colors.qualitative.Plotly
        color_discrete_map = {
            series: colors[i % len(colors)] for i, series in enumerate(series_names)
        }

    plots = {}
    metrics = plot_df["measurement"].unique()
    plot_titles = [
        "Forward Time (ms)",
        "Backward Time (ms)",
        "Forward Peak Memory (GB)",
        "Backward Peak Memory (GB)",
    ]

    for metric in plot_titles:
        if metric in metrics:
            metric_df = plot_df[plot_df["measurement"] == metric]
            if not metric_df.empty:
                fig = px.line(
                    metric_df,
                    x=x_axis_col,
                    y="value",
                    color=color_arg,
                    title=metric,
                    markers=True,
                    color_discrete_map=color_discrete_map,
                )
                fig.update_layout(
                    margin=dict(l=30, r=30, t=40, b=30), showlegend=False
                )
                plots[metric] = fig

    if not plots:
        mo.md("No metrics measured for this selection.")

    # Create a custom legend
    legend_items = []
    if color_discrete_map:
        for series, color in color_discrete_map.items():
            legend_items.append(
                mo.md(
                    f"""
                    <div style="display: flex; align-items: center; margin-right: 15px;">
                        <div style="width: 12px; height: 12px; background-color: {color}; margin-right: 5px;"></div>
                        <span>{series}</span>
                    </div>
                    """
                )
            )
    custom_legend = mo.hstack(legend_items, justify="center")

    # Display plots in a 2x2 grid, handling missing plots gracefully
    row1 = mo.hstack(
        [plots.get(plot_titles[0]), plots.get(plot_titles[1])], justify="center"
    )
    row2 = mo.hstack(
        [plots.get(plot_titles[2]), plots.get(plot_titles[3])], justify="center"
    )
    mo.vstack([custom_legend, row1, row2])
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
