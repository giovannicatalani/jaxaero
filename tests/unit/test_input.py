"""The case format: strict parsing of case.json + numerical.json into configs,
overrides, grids, and the example cases."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from jaxaero import CaseError, load_case
from jaxaero.case import apply_override, parse_case

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples"


def minimal() -> tuple[dict, dict]:
    """The smallest runnable case: everything else takes its default."""
    return ({"general": {"case_name": "mine"},
             "domain": {"grid": "o_grid", "airfoil": "naca0012", "cells_airfoil": 32, "cells_normal": 32,
                        "wall_spacing": 1e-3},
             "freestream": {"mach": 0.3}}, {})


def test_a_minimal_case_takes_the_defaults() -> None:
    case = parse_case(*minimal())
    assert case.flow.inviscid and case.grid_spec.grid == "o_grid"
    assert (case.numerics.flux, case.numerics.limiter, case.numerics.solver) == ("hllc", "minmod", "explicit")
    grid = case.grid().mesh[0]
    assert (grid.ni, grid.nj) == (32, 32)


@pytest.mark.parametrize("name", sorted(d.name for d in EXAMPLES.iterdir() if (d / "case.json").is_file()))
def test_the_examples_parse(name) -> None:
    case = load_case(EXAMPLES / name)
    assert case.name == name and case.output == EXAMPLES / name / "results"


@pytest.mark.parametrize("where, key", [
    ("case", ("domain", "secton")),
    ("case", ("freestream", "reynold")),
    ("numerical", ("convective_fluxes", "limter")),
    ("numerical", ("implicit", "cfl_maximum")),
])
def test_a_misspelt_key_is_an_error_naming_it(where, key) -> None:
    """A typo must never fall back silently to a default."""
    case, numerical = minimal()
    (case if where == "case" else numerical).setdefault(key[0], {})[key[1]] = 1
    with pytest.raises(CaseError, match=f"{key[0]}.{key[1]}"):
        parse_case(case, numerical)


def test_an_unknown_section_is_an_error() -> None:
    case, numerical = minimal()
    numerical["time_integrator"] = {}
    with pytest.raises(CaseError, match="time_integrator"):
        parse_case(case, numerical)


@pytest.mark.parametrize("where, section, key, value", [
    ("numerical", "convective_fluxes", "riemann_solver", "ausm"),
    ("numerical", "convective_fluxes", "limiter", "superbee"),
    ("numerical", "time_integration", "rk_scheme", "rk3"),
    ("numerical", "implicit", "preconditioner", "ilu"),
    ("case", "domain", "grid", "unstructured"),
    ("case", "boundary_conditions", "wall", {"type": "pressure_outlet"}),
    ("case", "boundary_conditions", "farfield", {"type": "far_field", "model": "vortex"}),
    ("case", "turbulence", "model", "k_omega"),
])
def test_an_unimplemented_option_is_an_error(where, section, key, value) -> None:
    """A key the solver cannot honour is refused, not ignored."""
    case, numerical = minimal()
    (case if where == "case" else numerical).setdefault(section, {})[key] = value
    with pytest.raises(CaseError, match=key):
        parse_case(case, numerical)


def test_viscous_material_properties_are_checked() -> None:
    case, numerical = minimal()
    case["material_properties"] = {"prandtl_number": 0.72}
    with pytest.raises(CaseError, match="inviscid"):
        parse_case(case, numerical)
    case["freestream"]["reynolds"] = 1000.0
    assert parse_case(case, numerical).flow.mu_inf == pytest.approx(0.3 / 1000.0)
    case["material_properties"] = {"prandtl_number": 0.71}
    with pytest.raises(CaseError, match="prandtl_number"):
        parse_case(case, numerical)


def test_sutherland_takes_the_ratio_of_two_temperatures() -> None:
    case, numerical = minimal()
    case["freestream"].update(reynolds=6e6, temperature=540.0)
    case["material_properties"] = {"dynamic_viscosity": "sutherland", "sutherland_temperature": 198.6}
    assert parse_case(case, numerical).flow.sutherland_ratio == pytest.approx(198.6 / 540.0)
    del case["freestream"]["temperature"]
    with pytest.raises(CaseError, match="freestream.temperature"):
        parse_case(case, numerical)


def test_turbulence_needs_a_viscous_case() -> None:
    case, numerical = minimal()
    case["turbulence"] = {"model": "sa_noft2"}
    with pytest.raises(CaseError, match="viscous"):
        parse_case(case, numerical)
    case["freestream"]["reynolds"] = 1e6
    assert parse_case(case, numerical).flow.turbulence == "sa_noft2"


def test_grid_keys_follow_the_grid_type() -> None:
    case, numerical = minimal()
    case["domain"] = {"grid": "file", "cells_around": 32, "file": "a.p2dfmt"}
    with pytest.raises(CaseError, match="cells_around"):
        parse_case(case, numerical)
    case["domain"] = {"grid": "file"}
    with pytest.raises(CaseError, match="domain.file"):
        parse_case(case, numerical)
    case["domain"] = {**minimal()[0]["domain"], "coordinate_scale": 2.0}
    with pytest.raises(CaseError, match="coordinate_scale"):
        parse_case(case, numerical)
    case["domain"] = {"grid": "file", "file": "a.x", "patches": [{"x_min": 0.0}]}
    with pytest.raises(CaseError, match=r"patches\[0\]"):
        parse_case(case, numerical)
    case["domain"] = {"grid": "file", "file": "a.x", "format": "vtk"}
    with pytest.raises(CaseError, match="domain.format"):
        parse_case(case, numerical)
    case["domain"] = {"grid": "c_grid", "airfoil": "naca0012", "cells_airfoil": 32, "cells_normal": 32,
                      "wall_spacing": 1e-3}
    with pytest.raises(CaseError, match="cells_wake"):
        parse_case(case, numerical)


def test_a_missing_grid_file_names_where_it_looked(tmp_path) -> None:
    case, numerical = minimal()
    case["domain"] = {"grid": "file", "file": "nowhere.p2dfmt"}
    with pytest.raises(FileNotFoundError, match="nowhere.p2dfmt"):
        parse_case(case, numerical, tmp_path).grid()


def test_overrides_reach_the_case_and_the_saved_record() -> None:
    case = load_case(EXAMPLES / "naca0012_laminar", ("numerical.convective_fluxes.limiter=minmod",
                                                     "case.freestream.alpha_deg=2"))
    assert case.numerics.limiter == "minmod" and case.flow.alpha_deg == 2.0
    assert case.raw["numerical"]["convective_fluxes"]["limiter"] == "minmod"
    assert load_case(EXAMPLES / "naca0012_laminar").numerics.limiter == "none"


def test_override_values_are_json_with_a_string_fallback() -> None:
    docs = {"case": {}, "numerical": {}}
    apply_override(docs, "case.freestream.reynolds=null")
    apply_override(docs, "numerical.convective_fluxes.limiter=none")
    assert docs["case"]["freestream"]["reynolds"] is None
    assert docs["numerical"]["convective_fluxes"]["limiter"] == "none"
    for bad in ("numerical.convective_fluxes.limiter", "physics.x.y=1", "case.x=1"):
        with pytest.raises(CaseError):
            apply_override(docs, bad)


def test_a_case_directory_anywhere_loads(tmp_path) -> None:
    case, numerical = minimal()
    (tmp_path / "case.json").write_text(json.dumps(case))
    assert load_case(tmp_path).name == "mine"          # numerical.json is optional
    (tmp_path / "numerical.json").write_text("{ not json")
    with pytest.raises(CaseError, match="numerical.json"):
        load_case(tmp_path)
    with pytest.raises(FileNotFoundError):
        load_case(tmp_path / "nothing")


def test_boundary_conditions_are_per_patch() -> None:
    """``{patch: {"type": ..., "model": ...}}``; the far-field model is the
    case's, and must be one."""
    case, numerical = minimal()
    assert parse_case(case, numerical).numerics.far_field == "characteristic"
    case["boundary_conditions"] = {"farfield": {"type": "far_field", "model": "point_vortex"}}
    assert parse_case(case, numerical).numerics.far_field == "characteristic_point_vortex"
    case["boundary_conditions"] = {"wall": {"type": "slip_wall", "model": "point_vortex"}}
    with pytest.raises(CaseError, match="far_field patch only"):
        parse_case(case, numerical)
    case["boundary_conditions"] = {"wall": {"type": "no_slip_wall"}}
    with pytest.raises(CaseError, match="inviscid"):
        parse_case(case, numerical)
    case["boundary_conditions"] = {"wall": "slip_wall"}
    with pytest.raises(CaseError, match="must be an object"):
        parse_case(case, numerical)
    case["boundary_conditions"] = {"a": {"type": "far_field"}, "b": {"type": "far_field", "model": "point_vortex"}}
    with pytest.raises(CaseError, match="one model per case"):
        parse_case(case, numerical)


def test_a_patch_the_grid_does_not_have_is_an_error_naming_the_grids_patches() -> None:
    case, numerical = minimal()
    case["boundary_conditions"] = {"airfoil": {"type": "slip_wall"}}
    with pytest.raises(CaseError, match=r"airfoil.*\['wall', 'farfield'\]"):
        parse_case(case, numerical).grid()


def test_patches_the_case_does_not_name_take_the_defaults() -> None:
    case = parse_case(*minimal())
    kinds = {s.kind for _, s in case.grid().mesh[0].sides.boundaries()}
    assert kinds == {"slip_wall", "far_field"}


def test_check_reports_the_grid_and_stops(tmp_path, capsys) -> None:
    from jaxaero.__main__ import main
    assert main([str(EXAMPLES / "clarky_sa"), "--check", "--out", str(tmp_path),
                 "--set", "case.domain.cells_airfoil=64", "--set", "case.domain.cells_wake=16",
                 "--set", "case.domain.cells_normal=24", "--set", "case.domain.wall_spacing=1e-4"]) == 0
    out = capsys.readouterr().out
    assert "trailing-edge gap 0.0012 closed" in out
    assert "patch 'wall': no_slip_wall, 64 faces" in out and "body 0:" in out
    assert not list(tmp_path.glob("*_fields.npz"))
