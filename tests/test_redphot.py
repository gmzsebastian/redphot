"""Integration and regression tests that do not require network access.

Only the named KeplerCam file is treated as observational test data.  Other
conditions are deterministic synthetic proxies and are not substitutes for
the release-validation observations listed in ``docs/validation.rst``.
"""

from hashlib import sha256
from pathlib import Path
import subprocess

import numpy as np
import pytest
from astropy import units as u
from astropy.io import fits
from astropy.nddata import CCDData
from astropy.table import Table
from astropy.wcs import WCS

from redphot.config import get_default_settings, resolve_settings, validate_settings
from redphot.image import (
    apply_cosmic_rays,
    assess_image_quality_batch,
    correct_fringe,
    detect_sources_and_measure_quality,
    detect_trails,
    model_background,
    read_fits_image,
)
from redphot.output import assemble_output_products, resolve_output_policy
from redphot.photometry import perform_science_image_photometry
from redphot.pipeline import (
    initialize_pipeline,
    load_pipeline_state,
    pipeline_stage_names,
    rerun_image,
    review_image,
    run_pipeline_stage,
    run_pipeline_through,
    set_image_overrides,
    set_run_overrides,
    skip_pipeline_stage,
)
from redphot.catalogs import (
    attach_photometric_references,
    plate_solve_with_astrometry_net,
)
from redphot.image import build_valid_region
from redphot.photometry import model_fwhm_pixels
from redphot.subtraction import (
    _run_hotpants,
    choose_hotpants_parameters,
    evaluate_subtraction,
)


DATA = Path(__file__).parent / "data"
FLWO_FILE = DATA / "AT_2024rmj_r_FLWO_2024.1012.fits"
# Pipeline tests that use stand-in stage functions skip the per-stage figures.
NO_PLOTS = {"diagnostics": {"save_stage_plots": False}}


def _digest(path):
    digest = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _wcs_header(shape=(96, 96)):
    header = fits.Header()
    header["CTYPE1"] = "RA---TAN"
    header["CTYPE2"] = "DEC--TAN"
    header["CRVAL1"] = 20.0
    header["CRVAL2"] = -30.0
    header["CRPIX1"] = shape[1] / 2 + 0.5
    header["CRPIX2"] = shape[0] / 2 + 0.5
    header["CD1_1"] = -0.4 / 3600.0
    header["CD1_2"] = 0.0
    header["CD2_1"] = 0.0
    header["CD2_2"] = 0.4 / 3600.0
    return header


def _lco_header(shape=(96, 96)):
    header = _wcs_header(shape)
    header.update({
        "OBJECT": "AT_TEST",
        "TELESCOP": "1m0-04",
        "INSTRUME": "fa04",
        "SITEID": "lsc",
        "EXPTIME": 120.0,
        "MJD-OBS": 61000.0,
        "FILTER": "rp",
        "GAIN": 1.5,
        "RDNOISE": 8.0,
        "SATURATE": 55000.0,
        "RLEVEL": 91,
    })
    return header


def _write_lco(path, compressed=False):
    rng = np.random.default_rng(2026)
    data = rng.normal(100.0, 4.0, (96, 96)).astype(np.float32)
    header = _lco_header(data.shape)
    if compressed:
        primary = fits.PrimaryHDU(header=header)
        science = fits.CompImageHDU(data=data, header=_wcs_header(data.shape), name="SCI")
        fits.HDUList([primary, science]).writeto(path)
    else:
        fits.PrimaryHDU(data=data, header=header).writeto(path)
    return data


@pytest.mark.skipif(not FLWO_FILE.exists(), reason="optional KeplerCam regression file absent")
def test_keplercam_duplicate_metadata_and_input_unchanged():
    before = _digest(FLWO_FILE)
    settings = resolve_settings("KeplerCam", image_name=FLWO_FILE.name)
    ccd, metadata = read_fits_image(FLWO_FILE, settings)
    assert ccd.shape == (1025, 1040)
    assert metadata["exposure_time"] == pytest.approx(300.0)
    assert "EXPOSURE_TIME_CONFLICT" in metadata["quality_flags"]
    assert "METADATA_CONFLICT" in metadata["quality_flags"]
    assert metadata["metadata_conflicts"][0]["values"] == [900.0, 300.0]
    assert _digest(FLWO_FILE) == before


def test_lco_single_hdu_and_compressed_fits_are_read_only(tmp_path):
    ordinary = tmp_path / "lco.fits"
    compressed = tmp_path / "lco.fits.fz"
    expected = _write_lco(ordinary)
    _write_lco(compressed, compressed=True)
    original = {_digest(path) for path in (ordinary, compressed)}
    settings = resolve_settings("LCO")

    ccd_single, metadata_single = read_fits_image(ordinary, settings)
    ccd_compressed, metadata_compressed = read_fits_image(compressed, settings)

    assert np.allclose(ccd_single.data, expected)
    assert ccd_single.shape == ccd_compressed.shape == expected.shape
    assert metadata_single["data_hdu"] == 0
    assert metadata_compressed["data_hdu"] == 1
    assert metadata_compressed["data_extname"] == "SCI"
    assert metadata_single["filter"] == metadata_compressed["filter"] == "r"
    assert {_digest(path) for path in (ordinary, compressed)} == original


def test_optional_image_stages_disable_independently():
    rng = np.random.default_rng(31)
    ccd = CCDData(rng.normal(100.0, 3.0, (64, 64)), unit=u.adu)
    original = np.array(ccd.data, copy=True)
    settings = get_default_settings()
    settings["masks"]["cosmic_rays"].update({"enabled": False, "mode": "off"})
    settings["fringe"]["enabled"] = False
    settings["background"].update({"enabled": False, "mode": "off"})
    settings["source_detection"]["enabled"] = False

    cosmic_ccd, _, cosmic_info = apply_cosmic_rays(ccd, settings=settings)
    fringe_ccd, _, fringe_info = correct_fringe(ccd, settings=settings)
    background_ccd, _, background_info = model_background(ccd, settings=settings)
    sources, segmentation, _ = detect_sources_and_measure_quality(ccd, settings=settings)

    assert cosmic_info["skipped"] == "disabled"
    assert fringe_info["skipped"] == "disabled"
    assert background_info["skipped"] == "off"
    assert len(sources) == 0 and not np.any(segmentation)
    for derivative in (cosmic_ccd, fringe_ccd, background_ccd):
        assert np.array_equal(derivative.data, original)
    assert np.array_equal(ccd.data, original)


def test_synthetic_trail_is_detected_and_masked():
    rng = np.random.default_rng(4)
    data = rng.normal(0.0, 1.0, (128, 128))
    data[63:65, 15:115] += 50.0
    settings = get_default_settings()
    settings["masks"].update({
        "trail_sigma": 4.0,
        "trail_min_length_pixels": 40,
        "trail_min_pixels": 20,
        "trail_min_elongation": 4.0,
    })
    mask, trails, info = detect_trails(data, settings=settings)
    assert info["detected"] is True
    assert len(trails) >= 1
    assert np.count_nonzero(mask[60:68, 10:120]) > 0


def test_synthetic_poor_seeing_and_shallow_epoch_is_rejected():
    def quality(fwhm, rms):
        return {
            "fwhm_arcsec": fwhm,
            "ellipticity": 0.1,
            "background": 100.0,
            "background_rms": rms,
            "quality_status": "PASS",
            "checks": [],
            "quality_flags": [],
        }

    assessed = assess_image_quality_batch(
        [quality(2.0, 5.0), quality(2.1, 5.0), quality(1.9, 5.0), quality(7.0, 20.0)]
    )
    bad = assessed[-1]
    assert bad["quality_status"] == "FAIL"
    assert "SEEING_POOR" in bad["quality_flags"]
    assert "BACKGROUND_RMS_HIGH" in bad["quality_flags"]
    assert "QUALITY_BATCH_OUTLIER" in bad["quality_flags"]


def test_synthetic_usable_and_failed_subtraction_quality():
    rng = np.random.default_rng(81)
    shape = (96, 96)
    yy, xx = np.indices(shape)
    science = rng.normal(100.0, 2.0, shape)
    positions = [(25.0, 25.0), (70.0, 25.0), (48.0, 70.0)]
    for x, y in positions:
        science += 5000.0 * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * 2.0 ** 2))
    record = {
        "image_id": "synthetic",
        "ccd": CCDData(science, unit=u.adu),
        "quality": {"fwhm_pixels": 4.7},
    }
    aligned = {"mask": np.zeros(shape, dtype=bool)}
    stars = Table({
        "image_id": ["synthetic"] * 3,
        "source_id": ["S1", "S2", "S3"],
        "x": [item[0] for item in positions],
        "y": [item[1] for item in positions],
        "role_qc_anchor": [True, False, False],
        "role_calibration": [False, True, True],
    })
    good = evaluate_subtraction(record, aligned, np.zeros(shape), quality_stars=stars)
    bad = evaluate_subtraction(record, aligned, science - 100.0, quality_stars=stars)
    assert good["status"] == "PASS"
    assert bad["status"] == "FAIL"
    assert "SUBTRACTION_RESIDUAL_HIGH" in bad["flags"]


def test_forced_noise_measurement_retains_signed_nondetection_flux():
    rng = np.random.default_rng(900)
    shape = (64, 64)
    wcs = WCS(_wcs_header(shape))
    ccd = CCDData(rng.normal(0.0, 2.0, shape), unit=u.adu, wcs=wcs)
    model_y, model_x = np.indices((15, 15))
    model = np.exp(-((model_x - 7) ** 2 + (model_y - 7) ** 2) / (2 * 1.7 ** 2))
    model /= model.sum()
    record = {
        "image_id": "noise",
        "ccd": ccd,
        "metadata": {"filename": "noise.fits", "filter": "r", "exposure_time": 30.0},
        "quality": {"background_rms": 2.0, "fwhm_pixels": 4.0},
    }
    target = {"ra_deg": 20.0, "dec_deg": -30.0, "frozen": True, "version": "test"}
    psf = {
        "image_id": "noise", "model": model, "model_native": model,
        "fwhm_pixels": 4.0,
        "approved_for_photometry": True, "status": "PASS", "review_state": "REVIEWED",
        "model_type": "gaussian", "normalization": 1.0,
    }
    result = perform_science_image_photometry(record, target, psf)
    target_rows = result["measurements"][
        np.asarray(result["measurements"]["source_type"], dtype=str) == "target"
    ]
    assert len(target_rows) == 3
    assert all(bool(row["valid"]) for row in target_rows)
    assert all(abs(float(row["snr"])) < 3.0 for row in target_rows)
    assert all(np.isfinite(float(row["flux"])) for row in target_rows)


def _stage_functions(fail_image=None):
    batch_stages = {
        "star_selection", "usability", "alignment", "calibration", "templates",
        "batch_consistency", "outputs",
    }

    def image_runner(stage):
        def run(context, image_id, settings):
            calls = context["shared"].setdefault("calls", {})
            key = "{}:{}".format(image_id, stage)
            calls[key] = calls.get(key, 0) + 1
            failed = context["shared"].setdefault("failed", set())
            if stage == "source_quality" and image_id == fail_image and image_id not in failed:
                failed.add(image_id)
                raise RuntimeError("synthetic contained failure")
            return {"status": "WARN" if stage == "psf" else "PASS"}
        return run

    def batch_runner(stage):
        def run(context, image_id, settings):
            if stage == "usability":
                decisions = [
                    {"image_id": identifier, "status": "PASS"}
                    for identifier, image in context["_state"]["images"].items()
                    if image.get("stages", {}).get("astrometry", {}).get("status") == "PASS"
                ]
                return {"status": "PASS", "decisions": decisions}
            return {"status": "PASS"}
        return run

    return {
        name: batch_runner(name) if name in batch_stages else image_runner(name)
        for name in pipeline_stage_names()
    }


def test_pipeline_resume_review_skip_override_and_individual_rerun(tmp_path):
    first, second = tmp_path / "one.fits", tmp_path / "two.fits"
    _write_lco(first)
    _write_lco(second)
    run_directory = tmp_path / "run"
    functions = _stage_functions(fail_image=second.name)
    state, context = initialize_pipeline([first, second], settings=NO_PLOTS,
                                         run_directory=run_directory)
    run_pipeline_through(state, context, stage_functions=functions)
    assert state["images"][first.name]["stages"]["psf"]["status"] == "APPROVED"
    assert state["images"][second.name]["stages"]["source_quality"]["status"] == "FAIL"
    assert state["images"][second.name]["stages"]["astrometry"]["blocked"] is True

    state, context = load_pipeline_state(run_directory)
    read_calls = context["shared"]["calls"][first.name + ":read"]
    run_pipeline_through(state, context, stage_functions=functions)
    assert context["shared"]["calls"][first.name + ":read"] == read_calls

    set_image_overrides(state, context, first.name, {"background": {"sigma_clip": 4.0}})
    assert state["images"][first.name]["stages"]["read"]["status"] == "PASS"
    assert state["images"][first.name]["stages"]["background"]["status"] == "STALE"
    skip_pipeline_stage(state, context, "background", first.name, "reviewed unchanged")
    assert state["images"][first.name]["stages"]["background"]["status"] == "SKIPPED"

    review_image(state, context, first.name, "usability", "REJECTED", "synthetic cloud")
    assert state["images"][first.name]["status"] == "REJECTED"
    review_image(state, context, first.name, "usability", "APPROVED", "manual recovery")
    rerun_image(
        state, context, second.name, "source_quality", "astrometry",
        stage_functions=functions,
    )
    assert state["images"][second.name]["stages"]["source_quality"]["status"] == "PASS"
    assert state["images"][second.name]["stages"]["astrometry"]["status"] == "PASS"


def _two_image_run(tmp_path, fail_image=False):
    """Two synthetic images; with ``fail_image`` the second always fails source_quality."""

    tmp_path.mkdir(parents=True, exist_ok=True)
    first, second = tmp_path / "one.fits", tmp_path / "two.fits"
    _write_lco(first)
    _write_lco(second)
    functions = _stage_functions()
    if fail_image:
        passing = functions["source_quality"]

        def always_fail(context, image_id, settings):
            if image_id == second.name:
                raise RuntimeError("synthetic persistent failure")
            return passing(context, image_id, settings)

        functions["source_quality"] = always_fail
    state, context = initialize_pipeline([first, second], settings=NO_PLOTS,
                                         run_directory=tmp_path / "run")
    return first.name, second.name, functions, state, context


def _pending(state, stage):
    return sorted(
        image_id for image_id, image in state["images"].items()
        if image.get("stages", {}).get(stage, {}).get("review_status") == "PENDING"
    )


def test_resume_keeps_rejected_image_history_and_review_decisions(tmp_path):
    """Regression: resuming erased a rejected image's earlier products and decisions."""

    first, second, functions, state, context = _two_image_run(tmp_path)
    run_pipeline_through(state, context, stage_functions=functions, mode="stepwise")
    assert _pending(state, "usability") == sorted([first, second])
    review_image(state, context, first, "usability", "APPROVED", "kept")
    review_image(state, context, second, "usability", "REJECTED", "clouds")
    run_pipeline_through(state, context, stage_functions=functions, mode="stepwise")
    assert _pending(state, "psf") == [first]
    review_image(state, context, first, "psf", "APPROVED", "PSF inspected")
    run_pipeline_through(state, context, stage_functions=functions, mode="stepwise")

    earlier = pipeline_stage_names()[:pipeline_stage_names().index("star_selection")]
    kept = {name: dict(state["images"][second]["stages"][name]) for name in earlier}
    reads = context["shared"]["calls"][second + ":read"]

    for mode in ("stepwise", "automatic"):
        state, context = load_pipeline_state(tmp_path / "run")
        run_pipeline_through(state, context, stage_functions=functions, mode=mode)
        rejected = state["images"][second]
        for name in earlier:
            assert rejected["stages"][name] == kept[name], (mode, name)
        assert context["shared"]["calls"][second + ":read"] == reads
        assert rejected["stages"]["usability"]["status"] == "REJECTED"
        assert rejected["review_decisions"]["usability"]["note"] == "clouds"
        assert rejected["stages"]["psf"]["status"] == "SKIPPED"
        assert "psf" not in rejected["review_decisions"]
        approved = state["images"][first]
        assert approved["stages"]["psf"]["status"] == "APPROVED"
        assert approved["review_decisions"]["psf"]["note"] == "PSF inspected"
        assert _pending(state, "usability") == [] and _pending(state, "psf") == []


def test_gates_do_not_review_images_blocked_upstream(tmp_path):
    first, second, functions, state, context = _two_image_run(tmp_path, fail_image=True)
    run_pipeline_through(state, context, stage_functions=functions, mode="stepwise")
    review_image(state, context, first, "usability", "APPROVED")
    run_pipeline_through(state, context, stage_functions=functions, mode="stepwise")
    blocked = state["images"][second]["stages"]["psf"]
    assert blocked["status"] == "SKIPPED" and blocked["blocked"] is True
    assert _pending(state, "psf") == [first]

    first, second, functions, state, context = _two_image_run(
        tmp_path / "automatic", fail_image=True
    )
    run_pipeline_through(state, context, stage_functions=functions, mode="automatic")
    assert state["images"][first]["stages"]["psf"]["status"] == "APPROVED"
    assert state["images"][second]["stages"]["psf"]["status"] == "SKIPPED"
    assert "psf" not in state["images"][second]["review_decisions"]


def test_output_profiles_and_traceable_core_products(tmp_path):
    assert resolve_output_policy(profile="minimal")["products"]["difference_image"] is False
    assert resolve_output_policy(profile="full")["products"]["background_model"] is True
    settings = get_default_settings()
    settings["output"]["overwrite"] = True
    record = {
        "image_id": "one", "path": "/input/one.fits", "status": "PASS",
        "metadata": {"object": "AT_TEST", "filename": "one.fits", "data_hdu": 0},
    }
    photometry = Table({
        "image_id": ["one"], "image_kind": ["science"], "method": ["psf"],
        "source_id": ["target"], "source_type": ["target"], "flux": [10.0],
        "calibration_catalog": ["ps1"], "zeropoint_mag": [25.0],
    })
    lightcurve = Table({
        "measurement_index": [0], "image_id": ["one"], "image_kind": ["science"],
        "method": ["psf"], "magnitude": [22.5], "included_in_final": [True],
    })
    products = assemble_output_products(
        [record], photometry=photometry, lightcurve=lightcurve, settings=settings,
        output_directory=tmp_path / "products", profile="minimal",
    )
    for name in ("images.ecsv", "sources.ecsv", "photometry.ecsv", "lightcurve.ecsv"):
        assert (tmp_path / "products" / name).exists()
    assert not list((tmp_path / "products" / "fits").glob("*"))
    assert products["lightcurve"]["source_measurement_id"][0] == (
        products["photometry"]["measurement_id"][0]
    )


def test_default_configuration_is_valid():
    validate_settings(get_default_settings())


def test_pipeline_applies_instrument_filter_and_image_precedence(tmp_path):
    first = tmp_path / "first.fits"
    second = tmp_path / "second.fits"
    _write_lco(first)
    _write_lco(second)
    state, context = initialize_pipeline(
        [first, second],
        instrument_name="LCO",
        filter_settings={"r": {"background": {"box_size": [77, 77]}}},
        image_overrides={
            first.name: {"background": {"box_size": [99, 99]}},
        },
        settings=NO_PLOTS,
        run_directory=tmp_path / "precedence",
    )
    assert context["images"][second.name]["settings"]["crop"]["size_arcmin"] == 15.0
    run_pipeline_stage(state, context, "read", save=False)
    assert context["images"][first.name]["settings"]["background"]["box_size"] == [99, 99]
    assert context["images"][second.name]["settings"]["background"]["box_size"] == [77, 77]


def test_unimplemented_options_are_rejected_instead_of_silently_ignored():
    cases = (
        ("apertures", "perform_optimal", True),
        ("psf", "spatial_order", 1),
        ("calibration", "apply_color_term", True),
        ("upper_limits", "injection_recovery", True),
    )
    for section, name, value in cases:
        settings = get_default_settings()
        settings[section][name] = value
        with pytest.raises(ValueError):
            validate_settings(settings)


def test_hotpants_kernel_tracks_the_measured_seeing():
    settings = get_default_settings()
    science = {
        "image_id": "science",
        "data": np.ones((128, 128), dtype=float),
        "quality": {"fwhm_pixels": 5.0},
        "metadata": {"saturation": 50000.0},
    }
    template = {
        "metadata": {"fwhm_pixels": 3.0, "saturation": 50000.0}
    }
    aligned = {
        "data": np.ones((128, 128), dtype=float),
        "mask": np.zeros((128, 128), dtype=bool),
        "wcs": None,
    }

    parameters = choose_hotpants_parameters(
        science, template, aligned, settings
    )
    expected = np.sqrt((5.0 / 2.354820045) ** 2 - (3.0 / 2.354820045) ** 2)
    assert parameters["convolve"] == "template"
    assert parameters["matching_sigma_pixels"] == pytest.approx(expected)
    assert parameters["gaussian_components"][1][1] == pytest.approx(expected)
    assert parameters["gaussian_components"][0][1] == pytest.approx(0.5 * expected)
    assert parameters["gaussian_components"][2][1] == pytest.approx(2.0 * expected)


def _image_with_nonfinite_pixels(shape=(64, 64)):
    rng = np.random.default_rng(3)
    data = rng.normal(100.0, 5.0, shape)
    data[10, 12] = np.inf
    data[20, 30] = -np.inf
    data[40, 5] = np.nan
    return data


def _argument(command, flag):
    return command[command.index(flag) + 1]


def test_plate_solve_input_is_finite_float32_and_science_is_unchanged(monkeypatch):
    """Regression: CFITSIO overflowed on inf pixels from zero-valued flats."""

    data = _image_with_nonfinite_pixels()
    ccd = CCDData(data.copy(), unit="adu", wcs=WCS(_wcs_header(data.shape)))
    seen = {}

    def fake_run(command, **kwargs):
        with fits.open(command[-1]) as hdulist:
            seen["bitpix"] = hdulist[0].header["BITPIX"]
            seen["finite"] = bool(np.isfinite(hdulist[0].data).all())
        fits.PrimaryHDU(np.zeros((2, 2), dtype=np.float32),
                        _wcs_header(data.shape)).writeto(_argument(command, "--new-fits"))
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("redphot.catalogs.shutil.which", lambda command: "/fake/solve-field")
    monkeypatch.setattr("redphot.catalogs.subprocess.run", fake_run)

    solved = plate_solve_with_astrometry_net(ccd, {"pixel_scale": 0.4})

    assert solved.has_celestial
    assert seen == {"bitpix": -32, "finite": True}
    np.testing.assert_array_equal(ccd.data, data)


def test_hotpants_inputs_are_finite_and_nonfinite_pixels_are_masked(monkeypatch):
    settings = get_default_settings()
    science = _image_with_nonfinite_pixels()
    template = np.full(science.shape, 100.0)
    template[50, 50] = np.nan
    parameters = choose_hotpants_parameters(
        {"image_id": "science", "data": science, "quality": {"fwhm_pixels": 5.0},
         "metadata": {"saturation": 50000.0}},
        {"metadata": {"fwhm_pixels": 3.0, "saturation": 50000.0}},
        {"data": template, "mask": np.zeros(science.shape, dtype=bool), "wcs": None},
        settings,
    )
    seen = {}

    def fake_run(command, **kwargs):
        science_input = fits.getdata(_argument(command, "-inim"))
        template_input = fits.getdata(_argument(command, "-tmplim"))
        seen["finite"] = bool(np.isfinite(science_input).all()
                              and np.isfinite(template_input).all())
        seen["science_mask"] = fits.getdata(_argument(command, "-imi")).astype(bool)
        seen["template_mask"] = fits.getdata(_argument(command, "-tmi")).astype(bool)
        fits.PrimaryHDU(science_input - template_input).writeto(_argument(command, "-outim"))
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("redphot.subtraction.shutil.which", lambda command: "/fake/hotpants")
    monkeypatch.setattr("redphot.subtraction.subprocess.run", fake_run)

    difference, _ = _run_hotpants(
        science, template, _wcs_header(science.shape), parameters,
        settings["subtraction"],
        science_mask=np.zeros(science.shape, dtype=bool),
    )

    assert seen["finite"]
    assert seen["science_mask"][10, 12] and seen["science_mask"][40, 5]
    assert seen["template_mask"][50, 50]
    assert seen["science_mask"].sum() == 3 and seen["template_mask"].sum() == 1
    for y, x in [(10, 12), (20, 30), (40, 5), (50, 50)]:
        assert np.isnan(difference[y, x])
    assert np.isfinite(difference).sum() == science.size - 4


def test_astrometry_matches_reach_star_selection(monkeypatch):
    """Regression: matches were stored as 'astrometry_matches' and star selection failed."""

    import redphot.catalogs as catalogs
    from redphot.pipeline import _run_astrometry, _run_star_selection

    matches = Table({"x": [1.0], "y": [2.0]})
    catalog = Table({"source_id": ["a"], "ra": [1.0], "dec": [2.0]})
    monkeypatch.setattr(catalogs, "solve_astrometry",
                        lambda *args, **kwargs: (catalog, matches, None, {"quality_status": "PASS"}))
    captured = {}

    def fake_master(records, settings):
        captured["records"] = records
        return Table(), Table()

    monkeypatch.setattr(catalogs, "build_master_source_table", fake_master)
    monkeypatch.setattr(catalogs, "attach_photometric_references",
                        lambda *args, **kwargs: ({}, {"catalogs": {}}))
    monkeypatch.setattr(catalogs, "select_comparison_and_psf_stars",
                        lambda master, measurements, settings, overrides: (master, measurements, []))
    record = {"image_id": "one", "metadata": {}, "sources": Table()}
    state = {"images": {"one": {"status": "PASS", "stages": {"astrometry": {"status": "PASS"}}}}}
    context = {"images": {"one": {"record": record, "products": {}}}, "shared": {}, "_state": state}
    _run_astrometry(context, "one", get_default_settings())
    assert record["matches"] is matches
    _run_star_selection(context, None, get_default_settings())
    assert captured["records"][0]["matches"] is matches


def test_outputs_are_rebuilt_when_an_upstream_stage_changes(tmp_path):
    """Regression: outputs had no dependencies, so stale products were reused."""

    first, second, functions, state, context = _two_image_run(tmp_path)
    flaky = _stage_functions(fail_image=second)
    calls = {"outputs": 0}

    def outputs(context, image_id, settings):
        calls["outputs"] += 1
        return {"status": "PASS"}

    flaky["outputs"] = outputs
    run_pipeline_through(state, context, stage_functions=flaky)
    assert state["images"][second]["stages"]["source_quality"]["status"] == "FAIL"
    assert calls["outputs"] == 1
    run_pipeline_through(state, context, stage_functions=flaky)
    assert state["images"][second]["stages"]["source_quality"]["status"] == "PASS"
    assert calls["outputs"] == 2
    run_pipeline_through(state, context, stage_functions=flaky)
    assert calls["outputs"] == 2


def test_set_run_overrides_keeps_per_image_overrides(tmp_path):
    first, second, functions, state, context = _two_image_run(tmp_path)
    run_pipeline_through(state, context, stage_functions=functions)
    set_image_overrides(state, context, first, {"background": {"box_size": [99, 99]}})
    set_run_overrides(state, context, {"background": {"box_size": [50, 50]}})
    assert context["images"][first]["settings"]["background"]["box_size"] == [99, 99]
    assert context["images"][second]["settings"]["background"]["box_size"] == [50, 50]
    assert state["images"][second]["stages"]["background"]["status"] == "STALE"
    assert state["images"][second]["stages"]["masks"]["status"] != "STALE"


def test_empirical_edges_ignore_lines_outside_the_data_section():
    """Regression: overscan already excluded by DATASEC was reported as BAD_EDGES."""

    rng = np.random.default_rng(7)

    def frame(bad_inner_column):
        data = rng.normal(1000.0, 10.0, (100, 120))
        data[:, :8] = 0.0
        data[:, 112:] = 0.0
        if bad_inner_column:
            data[:, 8] = 9000.0
        header = fits.Header()
        header["DATASEC"] = "[9:112,1:100]"
        return CCDData(data, unit="adu", meta=header)

    settings = get_default_settings()
    _, clean = build_valid_region(frame(False), settings=settings)
    edges = clean["empirical_edges"]
    assert clean["header_section"]["applied"]
    assert edges["left"] == 0 and edges["right"] == 0
    assert edges["left_outside_section"] == 8 and edges["right_outside_section"] == 8
    _, dirty = build_valid_region(frame(True), settings=settings)
    assert dirty["empirical_edges"]["left"] == 1
    assert dirty["empirical_edges"]["right"] == 0


def test_photometric_references_are_attached_by_position():
    gaia = Table({"source_id": ["a", "b", "c"], "ra": [10.0, 10.01, 10.02],
                  "dec": [20.0, 20.0, 20.0]}, masked=True)
    gaia["mag_r"] = np.ma.masked_all(3)
    gaia.meta["catalog_name"] = "gaia"
    offset = 0.2 / 3600.0
    ps1 = Table({"source_id": ["p1", "p2", "p3"],
                 "ra": [10.0 + offset, 10.01 + offset, 10.02 + 10 / 3600.0],
                 "dec": [20.0, 20.0, 20.0], "mag_r": [15.0, 16.0, 17.0],
                 "magerr_r": [0.01, 0.02, 0.03]}, masked=True)
    record = {"catalog": gaia, "metadata": {"filter": "r", "object": "test"}}
    references, info = attach_photometric_references(
        [record], get_default_settings(), supplied={"ps1": ps1})
    assert info["catalogs"]["ps1"]["matched"] == 2
    assert sorted(references["ps1"]["source_id"]) == ["a", "b"]
    filled = np.ma.filled(record["catalog"]["mag_r"].astype(float), np.nan)
    assert filled[:2] == pytest.approx([15.0, 16.0])
    assert np.isnan(filled[2])


def test_psf_fwhm_is_measured_from_the_model():
    """Regression: the PSF FWHM was the median segmentation width of bright stars."""

    yy, xx = np.indices((25, 25), dtype=float)
    sigma = 1.5
    model = np.exp(-((xx - 12) ** 2 + (yy - 12) ** 2) / (2 * sigma ** 2))
    assert model_fwhm_pixels(model / model.sum()) == pytest.approx(2.3548 * sigma, rel=0.01)


def test_rerunning_a_stage_starts_from_its_input_and_writes_diagnostics(tmp_path):
    """Regression: rerunning background used its own output and subtracted twice."""

    from redphot.pipeline import rerun_image
    from redphot.stage_reports import image_file_stem

    state, context = initialize_pipeline(
        [FLWO_FILE], instrument_name="KeplerCam", run_directory=tmp_path / "run",
        settings={"subtraction": {"enabled": False}},
    )
    run_pipeline_through(state, context, through_stage="background", mode="stepwise")
    image_id = FLWO_FILE.name
    background = context["images"][image_id]["products"]["background"]["products"]
    sky_before = float(np.nanmedian(background["background"]))
    assert sky_before > 100.0
    set_image_overrides(state, context, image_id, {"background": {"box_size": [96, 96]}})
    rerun_image(state, context, image_id, "background", "background")
    background = context["images"][image_id]["products"]["background"]["products"]
    sky_after = float(np.nanmedian(background["background"]))
    assert sky_after == pytest.approx(sky_before, rel=0.05)

    folder = tmp_path / "run" / "diagnostics"
    stem = image_file_stem(image_id)
    for stage in ("01_read", "02_region", "03_masks", "06_background"):
        assert (folder / stage / "{}.png".format(stem)).exists(), stage
        assert (folder / stage / "summary.csv").exists(), stage
    assert (folder / "04_cosmic_rays" / "skipped.png").exists()
    entry = state["images"][image_id]["stages"]["background"]
    assert entry.get("diagnostic_plot", "").endswith(".png")
    assert "diagnostic_error" not in entry


def test_every_diagnostic_figure_tolerates_missing_products():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from redphot import diagnostics as plots

    calls = [
        lambda: plots.plot_read_diagnostics(None, {}),
        lambda: plots.plot_region_diagnostics(None, {}, {}),
        lambda: plots.plot_mask_diagnostics(None, {}, {}),
        lambda: plots.plot_cosmic_ray_diagnostics(None, {}, {"skipped": "disabled"}),
        lambda: plots.plot_background_diagnostics(None, {}, {}),
        lambda: plots.plot_image_quality_diagnostics(None, None, None, {}),
        lambda: plots.plot_astrometry_diagnostics(None, None, None, {}),
        lambda: plots.plot_star_selection_diagnostics(None, None, "x"),
        lambda: plots.plot_image_usability_diagnostics(None, {}),
        lambda: plots.plot_alignment_target_diagnostics({}, {}),
        lambda: plots.plot_psf_diagnostics({}),
        lambda: plots.plot_science_photometry_diagnostics({}),
        lambda: plots.plot_calibration_diagnostics({}),
        lambda: plots.plot_calibration_image_diagnostics({}, "x"),
        lambda: plots.plot_subtraction_diagnostics({}),
        lambda: plots.plot_difference_photometry_diagnostics({}),
        lambda: plots.plot_batch_consistency_diagnostics({}),
        lambda: plots.plot_stage_status("Stage", "FAIL", "image", None, "error text"),
    ]
    for call in calls:
        plt.close(call())
    assert not plt.get_fignums()

