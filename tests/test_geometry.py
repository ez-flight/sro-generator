"""Тесты процедурной геометрии объектов (версия 1.1.0): инварианты формы."""

from __future__ import annotations

import math

import numpy as np
import pytest
from shapely.geometry import Polygon

from sro_generator.geometry import (
    generate_polygon_shape,
    is_proper_polygon,
    polygon_aspect_ratio,
)


class TestShapeInvariants:
    @pytest.mark.parametrize("irregularity", [0.0, 0.25, 0.5, 0.75, 1.0])
    @pytest.mark.parametrize("n", [3, 4, 5, 7, 10])
    def test_all_invariants_hold(self, irregularity, n):
        rng = np.random.default_rng(1000 + n * 17 + int(irregularity * 100))
        target_area = 12345.6
        max_ratio = 2.0
        poly = generate_polygon_shape(rng, n, target_area, irregularity, 123.0, max_ratio)
        assert poly.is_valid
        assert poly.is_simple  # нет самопересечений
        assert len(poly.exterior.coords) - 1 == n  # число вершин сохранено
        assert poly.area == pytest.approx(target_area, rel=1e-9)  # точная площадь
        assert polygon_aspect_ratio(poly) <= max_ratio + 1e-6  # отношение сторон в пределах

    def test_extreme_irregularity_many_seeds(self):
        """Крайнее значение irregularity=1.0 не ломает геометрию."""
        rng = np.random.default_rng(2024)
        ok = 0
        for _ in range(300):
            n = int(rng.integers(3, 11))
            rot = float(rng.uniform(0, 360))
            try:
                poly = generate_polygon_shape(rng, n, 5000.0, 1.0, rot, 3.0)
            except ValueError:
                continue
            assert is_proper_polygon(poly, n)
            assert polygon_aspect_ratio(poly) <= 3.0 + 1e-6
            ok += 1
        assert ok >= 250  # подавляющее большинство форм строится с первого раза

    def test_zero_irregularity_is_regular_polygon(self):
        rng = np.random.default_rng(5)
        poly = generate_polygon_shape(rng, 6, 1000.0, 0.0, 0.0, 10.0)
        # радиусы вершин от центроида равны (правильный многоугольник)
        cx, cy = poly.centroid.x, poly.centroid.y
        radii = [math.hypot(x - cx, y - cy) for x, y in poly.exterior.coords[:-1]]
        assert max(radii) - min(radii) < 1e-9

    def test_rotation_is_applied(self):
        rng = np.random.default_rng(9)
        rot = 33.0
        poly = generate_polygon_shape(rng, 4, 1000.0, 0.0, rot, 10.0)
        # первая вершина правильного многоугольника при irregularity=0 находится
        # на угле 0° (базовые углы linspace(0, 2π, n, endpoint=False))
        x0, y0 = poly.exterior.coords[0]
        angle = math.degrees(math.atan2(y0, x0)) % 360
        assert abs(angle - rot % 360) < 1e-6

    def test_degenerate_arguments_rejected(self):
        rng = np.random.default_rng(1)
        with pytest.raises(ValueError):
            generate_polygon_shape(rng, 2, 100.0, 0.0, 0.0, 2.0)
        with pytest.raises(ValueError):
            generate_polygon_shape(rng, 3, 0.0, 0.0, 0.0, 2.0)
        with pytest.raises(ValueError):
            generate_polygon_shape(rng, 3, -5.0, 0.0, 0.0, 2.0)
        with pytest.raises(ValueError):
            generate_polygon_shape(rng, 3, 100.0, 0.0, 0.0, 1.0)

    def test_is_proper_polygon_rejects_bowtie(self):
        bowtie = Polygon([(0, 0), (2, 2), (2, 0), (0, 2)])  # самопересечение
        assert not bowtie.is_valid
        assert not is_proper_polygon(bowtie, 4)
        zero_area = Polygon([(0, 0), (1, 0), (2, 0), (3, 0)])  # вырожденный
        assert zero_area.area == 0.0
        assert not is_proper_polygon(zero_area, 4)

    def test_aspect_ratio_helper(self):
        rect = Polygon([(0, 0), (20, 0), (20, 5), (0, 5)])
        assert polygon_aspect_ratio(rect) == pytest.approx(4.0)
        square = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        assert polygon_aspect_ratio(square) == pytest.approx(1.0)

    def test_area_scaling_accuracy(self):
        """Площадь после нормировки совпадает с целевой с относительной погрешностью 1e-9."""
        rng = np.random.default_rng(314)
        for _ in range(50):
            n = int(rng.integers(3, 11))
            area = float(rng.uniform(100.0, 1e7))
            poly = generate_polygon_shape(rng, n, area, 0.8, 45.0, 2.5)
            assert abs(poly.area - area) / area < 1e-9
