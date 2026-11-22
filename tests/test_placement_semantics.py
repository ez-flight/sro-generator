"""Тесты семантики размещения (версия 1.1.0): пересечения, расстояния, касание границ.

Воспроизводят исправленные ошибки 1.0.0:
- положительное min_distance_m совместно с allow_overlap=true фактически
  запрещало разрешённые перекрытия (проверка расстояния выполнялась раньше
  проверки перекрытия и отклоняла любое пересечение);
- допустимая область центров вычислялась через описанный радиус
  (inner.buffer(-R)), что ложно сокращало область и давало ложные отказы.
"""

from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import Point, Polygon

from sro_generator.placement import (
    PlacementError,
    exact_center_region,
    pair_placement_ok,
)


def sq(cx: float, cy: float, half: float) -> Polygon:
    return Polygon([
        (cx - half, cy - half), (cx + half, cy - half),
        (cx + half, cy + half), (cx - half, cy + half),
    ])


class TestPairPlacementOk:
    """Единая семантика: не пересекаются → расстояние ≥ min; пересекаются по
    площади → правила перекрытия; касание границ → правило расстояния."""

    def test_separate_beyond_distance_ok(self):
        a, b = sq(0, 0, 1), sq(10, 0, 1)
        assert pair_placement_ok(a, b, 5.0, False, 0.0)
        assert pair_placement_ok(a, b, 8.0, False, 0.0)

    def test_separate_below_distance_rejected(self):
        a, b = sq(0, 0, 1), sq(5, 0, 1)
        assert not pair_placement_ok(a, b, 5.0, False, 0.0)

    def test_touching_allowed_at_zero_min_distance(self):
        a, b = sq(0, 0, 1), sq(2, 0, 1)  # общая граница x=1
        assert a.intersection(b).area == 0.0
        assert a.distance(b) == 0.0
        assert pair_placement_ok(a, b, 0.0, False, 0.0)

    def test_touching_rejected_at_positive_min_distance(self):
        a, b = sq(0, 0, 1), sq(2, 0, 1)
        assert not pair_placement_ok(a, b, 0.5, False, 0.0)

    def test_overlap_forbidden_when_not_allowed(self):
        a, b = sq(0, 0, 10), sq(5, 0, 10)
        assert not pair_placement_ok(a, b, 0.0, False, 0.0)
        assert not pair_placement_ok(a, b, 100.0, False, 0.0)

    def test_overlap_within_fraction_allowed(self):
        a, b = sq(0, 0, 10), sq(19, 0, 10)  # пересечение полосой ширины 1
        inter = a.intersection(b).area  # 1 × 20 = 20
        frac = inter / min(a.area, b.area)  # 20/400 = 0.05
        assert 0.0 < frac < 0.1
        assert pair_placement_ok(a, b, 0.0, True, 0.1)
        assert not pair_placement_ok(a, b, 0.0, True, 0.01)

    def test_overlap_fraction_measured_against_smaller_polygon(self):
        # b полностью внутри a: area(∩)/min(area) = area(b)/area(b) = 1
        a, b = sq(0, 0, 100), sq(0, 0, 1)
        assert not pair_placement_ok(a, b, 0.0, True, 0.99)
        assert pair_placement_ok(a, b, 0.0, True, 1.0)

    def test_min_distance_does_not_block_allowed_overlap(self):
        """Регрессия 1.0.0: min_distance_m > 0 не должен запрещать разрешённое перекрытие."""
        a, b = sq(0, 0, 10), sq(19, 0, 10)
        assert pair_placement_ok(a, b, 500.0, True, 0.1)

    def test_symmetry(self):
        a, b = sq(0, 0, 10), sq(5, 3, 4)
        for min_d, allow, frac in [(0.0, False, 0.0), (2.0, False, 0.0), (0.0, True, 0.3), (7.0, True, 0.7)]:
            assert pair_placement_ok(a, b, min_d, allow, frac) == pair_placement_ok(b, a, min_d, allow, frac)

    def test_numerical_touch_noise(self):
        """Касание с float-шумом координат трактуется как касание, не пересечение."""
        eps = 1e-13
        a = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        b = Polygon([(1 - eps, 0), (2, 0), (2, 1), (1 - eps, 1)])
        # площадь «пересечения» ~ 1e-13 ≪ 1e-12·min(area) → касание
        assert pair_placement_ok(a, b, 0.0, False, 0.0)


class TestExactCenterRegion:
    """Точная допустимая область центров против приближения описанного радиуса."""

    def test_square_region_exact(self):
        inner = sq(0, 0, 1000)  # [-1000,1000]^2
        shape = sq(0, 0, 100)   # квадрат 200×200
        region = exact_center_region(inner, shape)
        assert region is not None
        # квадрат сдвигается внутрь квадрата: t ≤ 1000 − 100 по каждой оси
        minx, miny, maxx, maxy = region.bounds
        assert (minx, miny, maxx, maxy) == pytest.approx((-900, -900, 900, 900), abs=1e-6)
        assert region.area == pytest.approx(1800.0 ** 2, rel=1e-9)

    def test_region_corners_keep_shape_inside(self):
        rng = np.random.default_rng(7)
        inner = sq(0, 0, 5000)
        for _ in range(20):
            n = int(rng.integers(3, 11))
            angles = np.sort(rng.uniform(0, 2 * np.pi, n))
            radii = rng.uniform(0.3, 1.0, n)
            pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)]) * 300
            shape = Polygon(pts)
            if not shape.is_valid or shape.area <= 0:
                continue
            region = exact_center_region(inner, shape)
            assert region is not None
            corners = list(region.exterior.coords)
            for cx, cy in corners:
                shifted = Polygon([(x + cx, y + cy) for x, y in shape.exterior.coords[:-1]])
                # допуск ~1e-9 м на численный шум пересечения полуплоскостей
                assert inner.covers(shifted.buffer(1e-9)) or inner.distance(shifted) < 1e-6

    def test_exact_region_not_narrower_than_legacy(self):
        """Приближение описанного радиуса никогда не шире точной области."""
        inner = sq(0, 0, 2000)
        rng = np.random.default_rng(11)
        for _ in range(30):
            n = int(rng.integers(3, 11))
            angles = np.sort(rng.uniform(0, 2 * np.pi, n))
            radii = rng.uniform(0.3, 1.0, n)
            pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)]) * 400
            shape = Polygon(pts)
            if not shape.is_valid or shape.area <= 0:
                continue
            exact = exact_center_region(inner, shape)
            rad = max(np.hypot(pts[:, 0], pts[:, 1]))
            legacy = inner.buffer(-rad)
            assert exact is not None
            assert exact.area >= legacy.area - 1e-6

    def test_thin_triangle_legacy_empty_exact_nonempty(self):
        """Регрессия 1.0.0: треугольник с большим описанным радиусом и малой
        опорной функцией по нормали — legacy-область пуста, точная непуста."""
        inner = sq(0, 0, 2000)  # 4×4 км
        shape = Polygon([(0, 0), (2500, 0), (0, 10)])  # длинный узкий треугольник
        exact = exact_center_region(inner, shape)
        assert exact is not None and exact.area > 0
        legacy = inner.buffer(-2500.0)
        assert legacy.is_empty

    def test_rotated_rectangle_district(self):
        from sro_generator.geometry import build_district_polygon, LocalCRS

        local = LocalCRS(55.0, 60.0)
        district = build_district_polygon(local, 4000.0, 4000.0, 37.0)
        inner = district.buffer(-50.0)
        shape = sq(0, 0, 100)
        region = exact_center_region(inner, shape)
        assert region is not None and region.area > 0
        # все углы области — допустимые центры
        for cx, cy in region.exterior.coords:
            shifted = Polygon([(x + cx, y + cy) for x, y in shape.exterior.coords[:-1]])
            assert inner.covers(shifted.buffer(1e-9)) or inner.distance(shifted) < 1e-6

    def test_non_convex_inner_returns_none(self):
        # Г-образная невыпуклая область → None (вызывающий код переходит на legacy)
        lshape = Polygon([(0, 0), (2000, 0), (2000, 1000), (1000, 1000), (1000, 2000), (0, 2000)])
        assert exact_center_region(lshape, sq(0, 0, 100)) is None


class TestOverlapScenario:
    """Интеграционные сценарии с ручными центрами, воспроизводящие ошибку 1.0.0."""

    def _config(self, **over):
        import json
        from pathlib import Path
        from sro_generator.config import ScenarioConfig

        data = json.loads((Path(__file__).resolve().parent.parent / "example_config.json").read_text(encoding="utf-8"))
        for key, value in over.items():
            parts = key.split("__")
            node = data
            for p in parts[:-1]:
                node = node[p]
            node[parts[-1]] = value
        return ScenarioConfig.model_validate(data)

    def test_overlapping_manual_centers_allowed_in_1_1(self):
        """В 1.0.0 этот сценарий падал: min_distance_m=500 отклонял пересечение,
        хотя allow_overlap=true и доля 1.0 его разрешала."""
        from sro_generator.config import ManualCenter
        from sro_generator.generator import generate_scenario

        cfg = self._config(
            objects__count=2,
            objects__vertices_min=4,
            objects__vertices_max=4,
            objects__area_min_m2=2000.0,
            objects__area_max_m2=2000.0,
            objects__irregularity=0.0,
            placement__min_distance_m=500.0,
            placement__boundary_margin_m=100.0,
            placement__allow_overlap=True,
            placement__max_overlap_fraction=1.0,
        )
        cfg.placement.manual_centers = [
            ManualCenter(object_id="OBJ-001", latitude=55.0, longitude=60.0),
            ManualCenter(object_id="OBJ-002", latitude=55.0, longitude=60.0),  # тот же центр
        ]
        scenario = generate_scenario(cfg)
        assert len(scenario.objects) == 2

    def test_overlap_beyond_fraction_still_rejected(self):
        from sro_generator.config import ManualCenter
        from sro_generator.generator import generate_scenario

        cfg = self._config(
            objects__count=2,
            objects__vertices_min=4,
            objects__vertices_max=4,
            objects__area_min_m2=2000.0,
            objects__area_max_m2=2000.0,
            objects__irregularity=0.0,
            placement__min_distance_m=0.0,
            placement__boundary_margin_m=100.0,
            placement__allow_overlap=True,
            placement__max_overlap_fraction=0.2,
        )
        cfg.placement.manual_centers = [
            ManualCenter(object_id="OBJ-001", latitude=55.0, longitude=60.0),
            ManualCenter(object_id="OBJ-002", latitude=55.0, longitude=60.0),
        ]
        with pytest.raises(PlacementError):
            generate_scenario(cfg)
