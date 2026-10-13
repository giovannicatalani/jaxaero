"""The case schema: one table, kept equal to the configs it fills and the page generated from it."""

from __future__ import annotations

from dataclasses import MISSING, fields

import pytest

from jaxaero.case import CaseError, parse_case
from jaxaero.case.domain import GridSpec
from jaxaero.case.schema import CONDITIONS_HELP, FAR_FIELD_MODELS, REFERENCE, REQUIRED, keys, reference
from jaxaero.core.config import ImplicitConfig
from jaxaero.mesh.block import CONDITION_KINDS
from jaxaero.numerics.boundary import FAR_FIELDS


def test_the_reference_page_is_generated_from_the_schema() -> None:
    assert REFERENCE.read_text() == reference(), "regenerate: python -m jaxaero.case --write"


@pytest.mark.parametrize("section, cls", [("domain", GridSpec), ("implicit", ImplicitConfig)])
def test_the_schema_has_the_fields_and_defaults_of_its_config(section, cls) -> None:
    file = "case.json" if section == "domain" else "numerical.json"
    declared = keys(file, section)
    assert set(declared) == {f.name for f in fields(cls)}
    for f in fields(cls):
        key = declared[f.name]
        if f.default is not MISSING and key.default is not REQUIRED and not key.required_for:
            assert key.default == (list(f.default) if isinstance(f.default, tuple) else f.default), f.name


def test_every_condition_and_far_field_model_is_documented_and_implemented() -> None:
    assert set(CONDITIONS_HELP) == set(CONDITION_KINDS)
    assert set(FAR_FIELD_MODELS.values()) <= set(FAR_FIELDS)


def case(**freestream):
    return {"general": {"case_name": "t"},
            "domain": {"grid": "o_grid", "airfoil": "naca0012", "cells_airfoil": 32, "cells_normal": 32,
                       "wall_spacing": 1e-3},
            "freestream": {"mach": 0.5, **freestream}}


@pytest.mark.parametrize("freestream, message", [
    ({"mach": "0.5"}, "freestream.mach' must be a number"),
    ({"alpha_deg": True}, "freestream.alpha_deg' must be a number"),
    ({"mach": None}, "freestream.mach' may not be null"),
])
def test_a_value_of_the_wrong_type_is_refused_by_its_path(freestream, message) -> None:
    with pytest.raises(CaseError, match=message):
        parse_case(case(**freestream), {})


def test_an_integer_key_refuses_a_fraction() -> None:
    doc = case()
    doc["domain"]["cells_airfoil"] = 32.5
    with pytest.raises(CaseError, match="cells_airfoil' must be an integer"):
        parse_case(doc, {})
