"""Presentation layer: panels, layouts, colouring, orientation, video and charts.

All figures are plain :class:`matplotlib.figure.Figure` objects drawn with the
Agg canvas (``FigureCanvasAgg``); pyplot and interactive backends are never
used, so rendering is headless and safe inside GUI worker threads.
"""
