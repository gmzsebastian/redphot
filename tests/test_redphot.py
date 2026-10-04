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

from redphot.config import (
    get_default_settings,
    normalize_filter_name,
    resolve_settings,
    validate_settings,
)
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


def test_synthetic_poor_seeing_and_shallow_epoch_is_flagged():
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

    batch = [quality(2.0, 5.0), quality(2.1, 5.0), quality(1.9, 5.0), quality(7.0, 20.0)]
    # Relative-to-batch checks warn by default; only absolute limits reject.
    bad = assess_image_quality_batch(batch)[-1]
    assert bad["quality_status"] == "WARN"
    assert "SEEING_POOR" in bad["quality_flags"]
    assert "BACKGROUND_RMS_HIGH" in bad["quality_flags"]
    assert "QUALITY_BATCH_OUTLIER" in bad["quality_flags"]
    # A fail ratio can still be configured to reject such an epoch.
    settings = get_default_settings()
    settings["image_quality"]["batch_fwhm_ratio_fail"] = 2.5
    assert assess_image_quality_batch(batch, settings)[-1]["quality_status"] == "FAIL"


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


def _decided(state, stage):
    """Images whose gate decision is still open (there should never be any)."""

    return sorted(
        image_id for image_id, image in state["images"].items()
        if image.get("stages", {}).get(stage, {}).get("status") in {"PASS", "WARN"}
    )


def test_gates_never_wait_and_keep_manual_decisions(tmp_path):
    """Gates decide automatically in every mode; manual decisions survive resumes."""

    first, second, functions, state, context = _two_image_run(tmp_path)
    run_pipeline_through(state, context, stage_functions=functions, mode="stepwise")
    # Nothing waits for approval: both gates decided and the run went to the end.
    for stage in ("usability", "psf"):
        assert _decided(state, stage) == []
        for image_id in (first, second):
            assert state["images"][image_id]["stages"][stage]["status"] == "APPROVED"
    assert state["batch_stages"]["outputs"]["status"] == "PASS"

    # A manual rejection still works and is kept when the run is resumed.
    review_image(state, context, second, "usability", "REJECTED", "clouds")
    run_pipeline_through(state, context, stage_functions=functions, mode="stepwise")
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
        approved = state["images"][first]
        assert approved["stages"]["psf"]["status"] == "APPROVED"
        assert approved["review_decisions"]["psf"]["note"] == "PSF inspected"


def test_gates_do_not_review_images_blocked_upstream(tmp_path):
    for mode in ("stepwise", "automatic"):
        first, second, functions, state, context = _two_image_run(
            tmp_path / mode, fail_image=True
        )
        run_pipeline_through(state, context, stage_functions=functions, mode=mode)
        assert state["images"][first]["stages"]["psf"]["status"] == "APPROVED"
        blocked = state["images"][second]["stages"]["psf"]
        assert blocked["status"] == "SKIPPED" and blocked["blocked"] is True
        assert "psf" not in state["images"][second].get("review_decisions", {})


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


def test_filter_names_with_filter_or_band_words_are_recognized():
    for raw, expected in [
        ("g_filter", "g"), ("Filter r", "r"), ("FILTER_i", "i"), ("z-band", "z"),
        ("g'", "g"), ("Sloan_r_filter", "r"), ("R_filter", "R"), ("Bessell_V_filter", "V"),
    ]:
        assert normalize_filter_name(raw) == expected, raw
    # Single letters keep their case and unknown names come back unchanged.
    assert normalize_filter_name("r") == "r"
    assert normalize_filter_name("R") == "R"
    assert normalize_filter_name("filter") == "filter"
    assert normalize_filter_name("Halpha_filter") == "Halpha_filter"


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


def _star_field(shape, positions, fwhm, fluxes, rng, noise=5.0, shift=(0.0, 0.0),
                rotation_degrees=0.0):
    """Gaussian stars plus noise; the field can be rotated and shifted."""

    yy, xx = np.indices(shape)
    sigma = fwhm / 2.354820045
    data = rng.normal(0.0, noise, shape)
    center_y, center_x = (shape[0] - 1) / 2.0, (shape[1] - 1) / 2.0
    angle = np.radians(rotation_degrees)
    for (x, y), flux in zip(positions, fluxes):
        moved_x = center_x + np.cos(angle) * (x - center_x) - np.sin(angle) * (y - center_y) + shift[0]
        moved_y = center_y + np.sin(angle) * (x - center_x) + np.cos(angle) * (y - center_y) + shift[1]
        data += flux / (2 * np.pi * sigma ** 2) * np.exp(
            -((xx - moved_x) ** 2 + (yy - moved_y) ** 2) / (2 * sigma ** 2))
    return data


def _star_positions(shape, count, rng, border=15):
    return [(float(x), float(y)) for x, y in zip(
        rng.uniform(border, shape[1] - border, count), rng.uniform(border, shape[0] - border, count))]


def test_subtraction_quality_stars_come_from_star_selection(monkeypatch):
    """Regression: the pipeline passed no stars, so every subtraction failed its checks."""

    from redphot import pipeline
    from redphot.subtraction import _quality_positions

    stars = Table({
        "image_id": ["a", "a", "a", "b"],
        "persistent_id": ["gaia:1", "gaia:2", "gaia:3", "gaia:4"],
        "x": [10.0, 20.0, 30.0, 40.0],
        "y": [11.0, 21.0, 31.0, 41.0],
        "role_qc_anchor": [True, False, False, True],
        "role_calibration": [False, True, False, True],
        "role_psf": [False, False, False, False],
    })
    # Only image "a" rows with a QC, calibration or PSF role.
    assert _quality_positions({"image_id": "a"}, stars) == [
        (10.0, 11.0, "gaia:1"), (20.0, 21.0, "gaia:2")]

    seen = {}

    def fake_subtraction(record, template, settings, quality_stars, runner):
        seen["stars"] = quality_stars
        return {"image_id": "a", "status": "PASS"}

    monkeypatch.setattr("redphot.subtraction.perform_image_subtraction", fake_subtraction)
    context = {
        "images": {"a": {"record": {"image_id": "a", "metadata": {"filter": "r"}}}},
        "shared": {"templates": {"templates": {"r": {"data": np.zeros((4, 4))}}},
                   "star_selection": {"measurements": stars}},
    }
    pipeline._run_subtraction(context, "a", {"subtraction": {"enabled": True}})
    assert seen["stars"] is stars


def test_hotpants_difference_is_on_the_science_system_and_unsubtracted_pixels_are_nan(
        monkeypatch):
    settings = get_default_settings()
    science = np.full((32, 32), 10.0)
    seen = {}

    def fake_run(command, **kwargs):
        seen["normalize"] = _argument(command, "-n")
        fits.PrimaryHDU(np.zeros(science.shape, dtype=np.float32)).writeto(
            _argument(command, "-outim"))
        flags = np.zeros(science.shape, dtype=np.int32)
        flags[5, 6] = 0x8000          # Hotpants: no valid difference here
        flags[7, 8] = 0x40            # "OK" convolution flag only: keep
        fits.PrimaryHDU(flags).writeto(_argument(command, "-omi"))
        return subprocess.CompletedProcess(command, 0, "1 stamps built", "")

    monkeypatch.setattr("redphot.subtraction.shutil.which", lambda command: "/fake/hotpants")
    monkeypatch.setattr("redphot.subtraction.subprocess.run", fake_run)
    parameters = choose_hotpants_parameters(
        {"image_id": "science", "data": science, "quality": {"fwhm_pixels": 3.0},
         "metadata": {"saturation": 50000.0}},
        {"metadata": {"fwhm_pixels": 2.0, "saturation": 50000.0}},
        {"data": science.copy(), "mask": np.zeros(science.shape, dtype=bool), "wcs": None},
        settings,
    )
    difference, log = _run_hotpants(science, science.copy(), _wcs_header(science.shape),
                                    parameters, settings["subtraction"])
    assert seen["normalize"] == "i"
    assert np.isnan(difference[5, 6]) and difference[7, 8] == 0.0
    assert np.isfinite(difference).sum() == science.size - 1


def test_unknown_saturation_keeps_bright_stars_for_hotpants():
    """Regression: sky + 500 sigma flagged the bright LDSS3 stars, leaving ~8 stamps."""

    rng = np.random.default_rng(5)
    science = rng.normal(0.0, 10.0, (64, 64))
    science[30, 30] = 40000.0
    template = rng.normal(0.0, 2.0, (64, 64))
    template[30, 30] = 9000.0
    parameters = choose_hotpants_parameters(
        {"image_id": "science", "data": science, "quality": {"fwhm_pixels": 3.0},
         "metadata": {}},
        {"metadata": {}},
        {"data": template, "mask": np.zeros(template.shape, dtype=bool), "wcs": None},
        get_default_settings(),
    )
    assert parameters["science_upper"] == pytest.approx(20000.0)
    assert parameters["template_upper"] == pytest.approx(4500.0)
    assert parameters["science_lower"] == pytest.approx(-100.0, rel=0.2)
    assert parameters["background_order"] == 0


def test_template_registration_follows_a_rotated_and_shifted_template():
    """The template is moved onto the science stars when the WCSs disagree."""

    from redphot.subtraction import align_template_to_science, register_template_to_science

    rng = np.random.default_rng(11)
    shape = (200, 200)
    positions = _star_positions(shape, 70, rng)
    fluxes = rng.uniform(2e4, 1e5, len(positions))
    science = _star_field(shape, positions, 3.0, fluxes, rng)
    # The template's stars sit 0.6, -0.4 px and 0.8 degrees away (up to ~1.6 px).
    template_data = _star_field(shape, positions, 2.2, 0.5 * fluxes, rng, noise=2.0,
                                shift=(0.6, -0.4), rotation_degrees=0.8)
    wcs = WCS(_wcs_header(shape))
    record = {"image_id": "science", "ccd": CCDData(science, unit=u.adu, wcs=wcs),
              "wcs": wcs, "quality": {"fwhm_pixels": 3.0}}
    template = {"data": template_data, "mask": np.zeros(shape, dtype=bool), "wcs": wcs,
                "metadata": {"filter": "r"}}
    stars = Table({"image_id": ["science"] * len(positions),
                   "x": [x for x, _ in positions], "y": [y for _, y in positions]})
    settings = get_default_settings()
    aligned = align_template_to_science(record, template, settings)
    aligned, registration, error = register_template_to_science(
        record, template, aligned, stars, settings)
    assert error is None
    assert registration["shift_scatter_pixels"] > 0.3          # before: rotation
    assert registration["applied_offset"]
    assert abs(registration["residual_dx"]) < 0.05 and abs(registration["residual_dy"]) < 0.05
    assert registration["residual_scatter_pixels"] < 0.08
    assert registration["science_fwhm_pixels"] == pytest.approx(3.0, rel=0.05)
    assert registration["template_fwhm_pixels"] == pytest.approx(2.2, rel=0.05)


def test_subtraction_dipole_and_noise_checks():
    rng = np.random.default_rng(21)
    shape = (160, 160)
    positions = _star_positions(shape, 25, rng)
    fluxes = rng.uniform(5e4, 2e5, len(positions))
    stars_only = _star_field(shape, positions, 3.0, fluxes, rng, noise=0.0)
    shifted = _star_field(shape, positions, 3.0, fluxes, rng, noise=0.0, shift=(0.6, 0.0))
    noise_science = rng.normal(0.0, 5.0, shape)
    noise_template = rng.normal(0.0, 3.0, shape)
    science = stars_only + noise_science
    record = {"image_id": "science", "ccd": CCDData(science, unit=u.adu),
              "quality": {"fwhm_pixels": 3.0}}
    stars = Table({"image_id": ["science"] * len(positions),
                   "x": [x for x, _ in positions], "y": [y for _, y in positions]})
    aligned = {"data": 100.0 + noise_template / 0.5, "mask": np.zeros(shape, dtype=bool)}

    # Well subtracted: only the noise of both images remains.
    good = evaluate_subtraction(record, aligned, noise_science - noise_template,
                                quality_stars=stars, template_scale=0.5)
    assert good["status"] == "PASS"
    assert good["noise_ratio"] == pytest.approx(1.0, abs=0.3)
    assert good["median_dipole_fraction"] < 0.02

    # A template 0.6 px off leaves dipoles: first moment = 0.6 px = 0.2 FWHM.
    bad = evaluate_subtraction(record, aligned, stars_only - shifted + noise_science,
                               quality_stars=stars, template_scale=0.5)
    assert bad["median_dipole_fraction"] == pytest.approx(0.2, rel=0.15)

    # No quality stars: a clear flag, not "residual high".
    none = evaluate_subtraction(record, aligned, noise_science, quality_stars=stars[:0])
    assert none["flags"] == ["SUBTRACTION_TOO_FEW_STARS"]


def test_failed_subtraction_draws_its_figure_not_a_blank_card(monkeypatch):
    """Regression: a subtraction that failed its checks showed "No reason recorded."."""

    from redphot import diagnostics
    from redphot.stage_reports import stage_figure

    drawn = {}
    monkeypatch.setattr(diagnostics, "plot_subtraction_diagnostics",
                        lambda product, record, metadata=None, status=None, target=None:
                        drawn.setdefault("status", status) or "figure")
    monkeypatch.setattr(diagnostics, "plot_stage_status",
                        lambda title, status, stem, subtitle, reason: ("card", reason))
    state = {"images": {"a": {"stages": {"subtraction": {"status": "FAIL", "error": None}}}}}
    product = {"status": "FAIL", "flags": ["SUBTRACTION_NOISE_HIGH"],
               "difference": np.zeros((4, 4))}
    context = {"images": {"a": {"record": {"metadata": {}}, "products": {"subtraction": product}}},
               "settings": {}}
    stage_figure(state, context, "subtraction", "a")
    assert drawn["status"] == "FAIL"

    # A backend failure (no difference) gets the card, with the reason.
    product.update({"difference": None, "error": "hotpants: exit code 1"})
    assert stage_figure(state, context, "subtraction", "a") == ("card", "hotpants: exit code 1")


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
                        lambda master, measurements, settings, overrides, image_settings=None:
                        (master, measurements, []))
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
        lambda: plots.plot_cosmic_ray_diagnostics(None, {}, {}),
        lambda: plots.plot_background_diagnostics(None, {}, {}),
        lambda: plots.plot_image_quality_diagnostics(None, None, None, {}),
        lambda: plots.plot_astrometry_diagnostics(None, None, None, {}),
        lambda: plots.plot_star_selection_diagnostics(None, None, "x"),
        lambda: plots.plot_image_usability_diagnostics(None, {}),
        lambda: plots.plot_alignment_target_diagnostics({}, {}),
        lambda: plots.plot_alignment_check(None),
        lambda: plots.plot_alignment_check({"images": [], "sources": [], "cutouts": {}}),
        lambda: plots.plot_psf_diagnostics({}),
        lambda: plots.plot_psf_diagnostics({"model_native": np.zeros((9, 9))}),
        lambda: plots.plot_final_light_curve(None),
        lambda: plots.plot_final_light_curve(Table({"source_type": ["star"]})),
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



# ---------------------------------------------------------------------------
# Seeing, masks, trails, background, usability and batch consistency fixes
# ---------------------------------------------------------------------------

def _gaussian_star(data, x, y, flux, sigma):
    yy, xx = np.mgrid[0:data.shape[0], 0:data.shape[1]]
    data += flux / (2 * np.pi * sigma ** 2) * np.exp(
        -((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2)
    )


def test_seeing_fwhm_does_not_depend_on_brightness():
    rng = np.random.default_rng(11)
    sigma = 1.5
    data = rng.normal(100.0, 5.0, (260, 260))
    fluxes = np.geomspace(2e3, 4e5, 49)
    for index, flux in enumerate(fluxes):
        row, column = divmod(index, 7)
        _gaussian_star(data, 25 + 35 * column, 25 + 35 * row, flux, sigma)
    ccd = CCDData(data, unit="adu")
    sources, _, info = detect_sources_and_measure_quality(ccd, settings=get_default_settings())
    true_fwhm = 2.3548 * sigma
    assert info["fwhm_method"] == "gaussian_fit"
    assert abs(info["fwhm_pixels"] - true_fwhm) < 0.1
    fitted = np.asarray(sources["fwhm_pixels"].filled(np.nan), dtype=float)
    moments = np.asarray(sources["moment_fwhm_pixels"], dtype=float)
    flux = np.asarray(sources["flux"], dtype=float)
    faint = np.isfinite(fitted) & (flux < np.nanpercentile(flux, 40))
    bright = np.isfinite(fitted) & (flux > np.nanpercentile(flux, 60))
    # The Gaussian fit is flat with brightness; footprint moments are not.
    assert abs(np.median(fitted[bright]) - np.median(fitted[faint])) < 0.1
    assert np.median(moments[bright]) - np.median(moments[faint]) > 0.5


def test_saturation_mask_scales_with_the_star_and_ignores_unused_pixels():
    from redphot.image import make_saturation_mask

    data = np.full((200, 200), 1000.0)
    _gaussian_star(data, 60, 100, 6.0e7, 1.6)     # heavily saturated
    _gaussian_star(data, 140, 100, 1.0e6, 1.6)    # barely saturated
    data = np.minimum(data, 60000.0)
    excluded = np.zeros(data.shape, dtype=bool)
    excluded[:, :6] = True
    data[:, :6] = 65000.0                          # junk in an unused strip
    settings = get_default_settings()
    settings["masks"]["saturation_level"] = 50000.0
    components, info = make_saturation_mask(data, settings=settings, exclude=excluded)
    mask = components["saturation"]
    assert not mask[:, :6].any()
    assert not mask[:, 6:12].any()                  # no halo grown from the junk
    big = mask[60:140, 20:100].sum()
    small = mask[60:140, 100:180].sum()
    assert big > 1.4 * small > 0
    assert mask.mean() < 0.05
    assert len(info["regions"]) == 2


def test_faint_trail_is_masked_end_to_end_and_bleeds_are_not_trails():
    from redphot.image import make_saturation_mask

    rng = np.random.default_rng(5)
    data = rng.normal(100.0, 5.0, (300, 400))
    rows = np.arange(400)
    trail_y = (150 + 0.05 * (rows - 200)).round().astype(int)
    for column, row in zip(rows, trail_y):
        data[row - 1:row + 2, column] += 6.0          # ~1 sigma per pixel
    data[trail_y[200:320] - 1, rows[200:320]] += 60.0  # one bright stretch seeds it
    data[trail_y[200:320], rows[200:320]] += 60.0
    base = np.zeros(data.shape, dtype=bool)
    base[:, 90:110] = True                           # a masked gap across the trail
    # A saturated star with a bleed running up from it.
    _gaussian_star(data, 330, 60, 5.0e7, 1.6)
    data[60:130, 329:332] += 400.0
    data = np.minimum(data, 60000.0)
    settings = get_default_settings()
    settings["masks"]["saturation_level"] = 50000.0
    saturation, info = make_saturation_mask(data, settings=settings, exclude=base)
    mask, trails, trail_info = detect_trails(
        data, base_mask=base | saturation["saturation"], settings=settings,
        exclude_mask=saturation["saturation"], saturated_regions=info["regions"],
    )
    assert len(trails) == 1
    for column in (5, 60, 150, 380):
        assert mask[trail_y[column], column]
    assert trail_info["bleeds"]
    assert not mask[80:125, 325:336].any()
    assert trail_info["bleed_mask"][80:125, 325:336].any()


def test_bad_lines_ignore_smooth_gradients_but_catch_dead_columns():
    from redphot.image import make_line_defect_mask

    rng = np.random.default_rng(2)
    data = rng.normal(1000.0, 10.0, (300, 300))
    data += 100.0 * np.exp((np.arange(300) - 299.0) / 10.0)[:, None]  # glow at one edge
    data[:, 150] -= 200.0
    mask, info = make_line_defect_mask(data, settings=get_default_settings())
    assert info["bad_columns"] == [150]
    assert info["bad_rows"] == []


def test_background_reports_the_meshes_photutils_interpolates():
    rng = np.random.default_rng(3)
    data = rng.normal(500.0, 5.0, (256, 256))
    mask = np.zeros(data.shape, dtype=bool)
    mask[:, :128] = (np.arange(128) % 5 < 2)[None, :]   # 40% of each left box
    ccd = CCDData(data, unit="adu", mask=mask)
    settings = get_default_settings()
    settings["background"].update({"box_size": [64, 64], "exclude_percentile": 20.0})
    _, products, info = model_background(ccd, settings=settings)
    assert products["mesh_excluded"].sum() == 8
    assert info["excluded_mesh_fraction"] == pytest.approx(0.5)
    assert "BACKGROUND_MESHES_EXCLUDED" not in info["flags"]
    mask[:, 128:192] = (np.arange(64) % 5 < 2)[None, :]   # now 12 of 16 boxes
    _, _, info = model_background(CCDData(data, unit="adu", mask=mask), settings=settings)
    assert info["excluded_mesh_fraction"] == pytest.approx(0.75)
    assert "BACKGROUND_MESHES_EXCLUDED" in info["flags"]


def test_cosmic_rays_handle_nonfinite_pixels_and_never_mask_the_target(recwarn):
    pytest.importorskip("astroscrappy")
    rng = np.random.default_rng(8)
    shape = (120, 120)
    data = rng.normal(500.0, 10.0, shape)
    data[:, :3] = np.nan
    data[30, 90] += 3000.0          # an ordinary hit
    data[60, 60] += 3000.0          # a hit (or compact source) on the target
    wcs = WCS(_wcs_header(shape))
    ccd = CCDData(data, unit="adu", wcs=wcs)
    target = wcs.pixel_to_world(60, 60)
    settings = get_default_settings()
    settings["masks"]["cosmic_rays"].update({"enabled": True, "gain": 1.0, "read_noise": 5.0})
    _, products, info = apply_cosmic_rays(ccd, settings=settings, target=target)
    assert not [warning for warning in recwarn if issubclass(warning.category, RuntimeWarning)]
    assert products["cosmic_mask"][30, 90]
    assert not products["cosmic_mask"][58:63, 58:63].any()
    assert info["target_overlap"] and info["target_protected_pixels"] > 0


def test_quick_zeropoint_uses_aperture_fluxes_without_a_magnitude_trend():
    from redphot.image import _quick_zeropoint

    magnitude = np.linspace(14.0, 19.5, 40)
    total = 10 ** (-0.4 * (magnitude - 25.0)) * 300.0
    rows = Table({
        "magnitude": magnitude,
        "role_calibration": np.ones(40, dtype=bool),
        "aperture_flux": 0.8 * total,
        "aperture_flux_large": 0.98 * total,
        # Footprint fluxes lose more light for fainter stars.
        "flux": total * (1.0 - 0.6 * (magnitude - 14.0) / 5.5),
    }, masked=True)
    zeropoints, result, inlier, method = _quick_zeropoint(
        rows, {"exposure_time": 300.0}, get_default_settings()
    )
    assert method == "aperture_flux_corrected"
    assert result["zeropoint_scatter_mag"] < 0.01
    assert abs(result["zeropoint_mag"] - 25.0 - 2.5 * np.log10(0.98)) < 0.01
    assert abs(np.polyfit(magnitude, zeropoints, 1)[0]) < 1e-3


def test_batch_quality_compares_sky_within_a_filter_and_only_warns():
    def quality(fwhm, background):
        return {"fwhm_arcsec": fwhm, "ellipticity": 0.05, "background": background,
                "background_rms": 10.0, "quality_status": "PASS",
                "checks": [], "quality_flags": []}

    results = [quality(2.0, 300.0), quality(2.1, 320.0), quality(2.0, 310.0),
               quality(5.6, 305.0), quality(1.9, 3000.0), quality(2.0, 3100.0),
               quality(2.0, 2900.0)]
    groups = ["g", "g", "g", "g", "r", "r", "r"]
    assessed = assess_image_quality_batch(results, get_default_settings(), groups=groups,
                                          exposure_times=[300.0] * 7)
    assert all("BACKGROUND_HIGH" not in item["quality_flags"] for item in assessed)
    assert "SEEING_POOR" in assessed[3]["quality_flags"]
    assert assessed[3]["quality_status"] == "WARN"


def test_comparison_star_stability_uses_errors_floor_and_epochs():
    from redphot.pipeline import build_comparison_star_light_curves

    rows = []
    images = ["a", "b", "c", "d", "e"]
    rng = np.random.default_rng(1)
    for index, image in enumerate(images):
        def add(source, magnitude, error):
            rows.append({"image_id": image, "source_id": source, "filter": "r",
                         "method": "psf", "source_type": "comparison",
                         "image_kind": "science", "valid": True, "mjd_mid": 100.0 + index,
                         "calibrated_magnitude": magnitude,
                         "calibrated_magnitude_uncertainty": error})
        add("bright", 14.0 + rng.normal(0, 0.006), 0.003)
        add("faint", 19.0 + rng.normal(0, 0.08), 0.1)
        add("variable", 15.0 + (0.3 if index % 2 else 0.0), 0.005)
        for star in range(6):
            add("field{}".format(star), 15.5 + rng.normal(0, 0.005), 0.005)
        if index < 2:
            add("sparse", 16.0, 0.01)
    _, stability = build_comparison_star_light_curves(Table(rows), get_default_settings())
    status = {row["source_id"]: str(row["status"]) for row in stability}
    assert status["bright"] == "PASS"
    assert status["faint"] == "PASS"
    assert status["variable"] == "FAIL"
    assert status["sparse"] == "UNTESTED"


def test_light_curve_evolution_is_not_flagged_but_jumps_are():
    from redphot.pipeline import build_preferred_light_curve

    def target(image, mjd, magnitude):
        return {"image_id": image, "source_type": "target", "image_kind": "science",
                "method": "psf", "valid": True, "filter": "r", "mjd_mid": mjd,
                "calibrated_magnitude": magnitude,
                "calibrated_magnitude_uncertainty": 0.02, "flags": ""}

    slow = Table([target("a", 100.0, 19.8), target("b", 117.0, 19.45),
                  target("c", 131.0, 20.0)], masked=True)
    curve = build_preferred_light_curve(slow, Table(), Table(), get_default_settings())
    assert not any("BATCH_MEASUREMENT_OUTLIER" in str(flag) for flag in curve["flags"])
    fast = Table([target("a", 100.0, 19.8), target("b", 101.0, 19.0),
                  target("c", 102.0, 19.8)], masked=True)
    curve = build_preferred_light_curve(fast, Table(), Table(), get_default_settings())
    assert "BATCH_MEASUREMENT_OUTLIER" in str(curve["flags"][1])


def test_epoch_metrics_do_not_fail_on_a_tiny_batch_scatter():
    from redphot.pipeline import build_epoch_metrics

    records = [
        {"image_id": str(index), "metadata": {"filter": "r", "mjd_mid": 100.0 + index},
         "quality": {"fwhm_arcsec": fwhm}}
        for index, fwhm in enumerate([1.92, 1.93, 1.94, 2.17, 2.36])
    ]
    metrics = build_epoch_metrics(records, settings=get_default_settings())
    assert set(str(value) for value in metrics["status"]) == {"PASS"}


def test_region_figure_keeps_its_layout_and_marks_unused_lines():
    import warnings

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from redphot import diagnostics as plots

    data = np.random.default_rng(0).normal(600.0, 10.0, (120, 130))
    valid = np.ones(data.shape, dtype=bool)
    valid[:, :8] = valid[:, 122:] = valid[119:, :] = False
    region = {"header_section": {"applied": True, "keyword": "DATASEC",
                                 "bounds": (8, 122, 0, 119)}}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        figure = plots.plot_region_diagnostics(
            CCDData(data, unit="adu"), region, {"full_valid_mask": valid})
        figure.canvas.draw()
    assert not [item for item in caught if "constrained" in str(item.message).lower()]
    # Constrained layout really ran (the default left margin would be 0.125).
    assert min(axis.get_position().x0 for axis in figure.axes) < 0.06
    texts = [child.get_text() for axis in figure.axes for child in axis.texts]
    assert any("columns 0–7 not used" in text for text in texts)
    plt.close(figure)


# ---------------------------------------------------------------------------
# Survey templates (web services replaced by local fakes)
# ---------------------------------------------------------------------------

def _sky_wcs(ra, dec, scale_arcsec, shape):
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crval = [ra, dec]
    wcs.wcs.crpix = [(shape[1] + 1) / 2.0, (shape[0] + 1) / 2.0]
    wcs.wcs.cdelt = [-scale_arcsec / 3600.0, scale_arcsec / 3600.0]
    return wcs


def _science_record(ra=103.596, dec=17.492, shape=(120, 120), scale=1.0):
    wcs = _sky_wcs(ra, dec, scale, shape)
    ccd = CCDData(np.zeros(shape), unit="adu", wcs=wcs)
    return {"image_id": "science.fits", "ccd": ccd, "wcs": wcs,
            "metadata": {"filter": "r"}}


def _fake_survey_image(params, scale, value, cube=False, invvar=False):
    """A FITS cutout like the services return, centered on the request."""

    ra, dec = float(params["ra"]), float(params["dec"])
    width = int(params.get("width", params.get("size")))
    height = int(params.get("height", params.get("size")))
    wcs = _sky_wcs(ra, dec, scale, (height, width))
    data = np.full((height, width), value, dtype=np.float32)
    data[0, 0] = value + 1.0
    header = wcs.to_header()
    if cube:
        data = data[None]
        header["NAXIS"] = 3
        header["CTYPE3"] = "BAND"
    hdus = [fits.PrimaryHDU(data, header=header)]
    if invvar:
        hdus.append(fits.ImageHDU(np.ones_like(data)))
    buffer = __import__("io").BytesIO()
    fits.HDUList(hdus).writeto(buffer)
    return buffer.getvalue()


def test_template_surveys_ps1_and_legacy_download_tile_and_cache(tmp_path, monkeypatch):
    from redphot import subtraction

    calls = []

    def fake_get(url, params, timeout):
        calls.append((url, dict(params)))
        if url == subtraction.PS1_FILENAMES_URL:
            assert params["filters"] == "r" and params["type"] == "stack"
            text = ("projcell subcell ra dec filter mjd type filename shortname\n"
                    "1785 45 {ra} {dec} r 56000.0 stack "
                    "/rings.v3.skycell/1785/045/rings.v3.skycell.1785.045.stk.r.unconv.fits "
                    "rings.v3.skycell.1785.045.stk.r.unconv.fits\n").format(
                        ra=params["ra"], dec=params["dec"])
            return text.encode()
        if url == subtraction.PS1_CUTOUT_URL:
            assert params["format"] == "fits" and params["red"].endswith(".fits")
            return _fake_survey_image(params, 0.25, 5.0)
        if url == subtraction.LEGACY_CUTOUT_URL:
            assert params["layer"] == "ls-dr10" and params["bands"] == "r"
            return _fake_survey_image(params, 0.262, 7.0, cube=True, invvar=True)
        raise AssertionError(url)

    monkeypatch.setattr(subtraction, "_http_get", fake_get)
    settings = get_default_settings()
    settings["subtraction"]["cache_directory"] = str(tmp_path / "templates")
    settings["subtraction"]["template_margin_arcmin"] = 0.5
    records = [_science_record()]

    template = subtraction.acquire_template(records, "r", settings)
    assert template["acquisition"]["survey"] == "ps1"
    assert template["metadata"]["filter"] == "r"
    assert np.nanmedian(template["data"]) == pytest.approx(5.0)
    assert np.mean(template["mask"]) < 0.01
    assert Path(template["cached_path"]).exists()
    downloads = len(calls)
    # A second request for the same field reads the cache, not the network.
    again = subtraction.acquire_template(records, "r", settings)
    assert again["acquisition"]["from_cache"] and len(calls) == downloads

    # A wide field is cut into tiles no larger than the service allows.
    settings["subtraction"]["template_source"] = "legacy"
    settings["subtraction"]["template_surveys"]["legacy"]["maximum_cutout_pixels"] = 200
    legacy = subtraction.acquire_template(records, "r", settings)
    assert legacy["acquisition"]["survey"] == "legacy"
    cutouts = [params for url, params in calls if url == subtraction.LEGACY_CUTOUT_URL]
    assert len(cutouts) > 1 and max(int(item["width"]) for item in cutouts) <= 200
    assert np.nanmedian(legacy["data"]) == pytest.approx(7.0)


def test_template_survey_without_the_band_falls_through_to_the_next(tmp_path, monkeypatch):
    from redphot import subtraction

    def fake_get(url, params, timeout):
        if url == subtraction.LEGACY_CUTOUT_URL:
            return _fake_survey_image(params, 0.262, 3.0, invvar=True)
        raise AssertionError("PS1 has no u band and must not be queried")

    monkeypatch.setattr(subtraction, "_http_get", fake_get)
    settings = get_default_settings()
    settings["subtraction"]["cache_directory"] = str(tmp_path / "templates")
    settings["subtraction"]["template_survey_priority"] = ["ps1", "legacy"]
    settings["subtraction"]["template_surveys"]["legacy"]["filters"]["u"] = "g"
    record = _science_record()
    record["metadata"]["filter"] = "u"
    template = subtraction.acquire_template([record], "u", settings)
    attempts = template["acquisition"]["attempts"]
    assert attempts[0]["survey"] == "ps1" and attempts[0]["status"] == "FAIL"
    assert "no u band" in attempts[0]["error"]
    assert template["acquisition"]["survey"] == "legacy"


# ---------------------------------------------------------------------------
# Edges, background boxes, processed outputs, alignment check
# ---------------------------------------------------------------------------

def test_edge_ramps_are_trimmed_and_the_frame_is_cut_to_the_usable_area():
    from redphot.image import define_processing_region

    rng = np.random.default_rng(7)
    shape = (300, 320)
    data = rng.normal(1000.0, 10.0, shape)
    data[:, :6] = 0.0                       # overscan outside DATASEC
    data[-25:, :] += np.linspace(0.0, 30.0, 25)[:, None]   # ramp along the top rows
    header = _wcs_header(shape)
    header["DATASEC"] = "[7:320,1:300]"
    ccd = CCDData(data, unit="adu", meta=header, wcs=WCS(header))
    settings = get_default_settings()
    working, region, diagnostics = define_processing_region(ccd, {}, settings)
    edges = region["empirical_edges"]
    assert edges["top_level"] >= 12 and edges["bottom_level"] == 0
    assert edges["left_level"] == 0 and edges["right_level"] == 0
    assert region["crop"]["trimmed_to_valid"]
    (y0, y1), (x0, x1) = region["crop"]["slices"]
    assert (y0, x0, x1) == (0, 6, 320) and y1 < 300 - 12
    assert working.shape == (y1 - y0, x1 - x0)
    # The WCS follows the cut: the same pixel has the same sky position.
    original = WCS(header).pixel_to_world(x0 + 10, y0 + 20)
    assert working.wcs.pixel_to_world(10, 20).separation(original).arcsec < 1e-6
    assert np.allclose(working.data, data[y0:y1, x0:x1])

    settings["crop"]["trim_to_valid"] = False
    settings["crop"]["edge_level_trim"] = False
    untouched, region, _ = define_processing_region(ccd, {}, settings)
    assert untouched.shape == shape and region["empirical_edges"]["top_level"] == 0


def test_background_boxes_span_the_frame_exactly():
    from redphot.image import _effective_background_box

    settings = get_default_settings()
    settings["background"]["box_size"] = [64, 64]
    settings["source_detection"]["fwhm_guess_pixels"] = 3.0
    for shape in ((1024, 1024), (965, 996), (1025, 1040)):
        _, (box_y, box_x), _ = _effective_background_box(shape, settings)
        for size, box in zip(shape, (box_y, box_x)):
            count = int(np.ceil(size / box))
            assert abs(box - 64) <= 6
            # the last (partial) box keeps most of its pixels
            assert size - (count - 1) * box >= 0.75 * box
    settings["background"]["fit_box_to_frame"] = False
    assert _effective_background_box((1025, 1040), settings)[1] == (64, 64)


def test_processed_images_carry_mask_bits_and_background_next_to_the_psf_model(tmp_path):
    from redphot.output import MASK_BITS, save_processed_images

    shape = (40, 50)
    header = _wcs_header(shape)
    saturation = np.zeros(shape, dtype=bool)
    saturation[5:8, 5:8] = True
    trails = np.zeros(shape, dtype=bool)
    trails[20, :] = True
    item = {
        "image_id": "one.fits",
        "data": np.ones(shape),
        "header": header,
        "mask_components": {"saturation": saturation, "trails": trails},
        "background": np.full(shape, 100.0),
        "background_rms": np.full(shape, 5.0),
        "uncertainty": None,
    }
    policy = resolve_output_policy(profile="standard")
    entries = save_processed_images([item], tmp_path, policy, "digest", "run")
    kinds = {kind for kind, _, _ in entries}
    # one final image per input (no resampled "registered" copy any more)
    assert kinds == {"processed_image"}
    assert "registered_image" not in policy["products"]
    processed = next(path for kind, path, _ in entries if kind == "processed_image")
    assert Path(processed) == tmp_path / "fits" / "one_processed.fits"
    with fits.open(processed) as hdulist:
        assert [hdu.name for hdu in hdulist] == ["PRIMARY", "MASK", "BKG", "BKGRMS"]
        bits = hdulist["MASK"].data
        assert bits[6, 6] == MASK_BITS["saturation"]
        assert bits[20, 30] == MASK_BITS["trails"]
        assert bits[30, 30] == 0
        assert hdulist[0].header["RDPPROD"] == "processed_image"
        assert WCS(hdulist[0].header).has_celestial
    assert policy["products"]["lightcurve_plot"] is True
    assert resolve_output_policy(profile="minimal")["products"]["processed_image"] is False


def test_alignment_check_centers_sources_and_measures_wcs_offsets():
    from redphot.alignment import build_alignment_check

    shape = (120, 120)
    header = _wcs_header(shape)
    yy, xx = np.indices(shape)
    data = 10.0 + 500.0 * np.exp(-((xx - 70.0) ** 2 + (yy - 50.0) ** 2) / (2 * 1.5 ** 2))
    wcs = WCS(header)
    star = wcs.pixel_to_world(70.0, 50.0)
    shifted = WCS(header)
    shifted.wcs.crpix = [shifted.wcs.crpix[0] + 2.0, shifted.wcs.crpix[1]]  # 2 px = 0.8″
    records, alignments = [], []
    for name, frame_wcs in (("good.fits", wcs), ("shifted.fits", shifted)):
        ccd = CCDData(data.copy(), unit="adu", wcs=frame_wcs, mask=np.zeros(shape, bool))
        records.append({"image_id": name, "ccd": ccd, "metadata": {"filter": "r"},
                        "quality": {"fwhm_pixels": 3.5}})
        alignments.append({"image_id": name, "wcs": frame_wcs, "status": "PASS",
                           "is_reference": name == "good.fits"})
    source = {"name": "star 1", "kind": "star", "ra_deg": float(star.ra.deg),
              "dec_deg": float(star.dec.deg)}
    check = build_alignment_check(records, alignments, [source], get_default_settings())
    frames = check["cutouts"]["star 1"]["frames"]
    good, moved = frames
    assert good["reliable"] and np.hypot(good["dx_arcsec"], good["dy_arcsec"]) < 0.05
    # The shifted WCS predicts the star 2 px off, so the centroid is 0.8″ away.
    assert np.hypot(moved["dx_arcsec"], moved["dy_arcsec"]) == pytest.approx(0.8, abs=0.08)
    assert check["cutouts"]["star 1"]["sum"].shape == good["data"].shape


def test_profile_found_in_the_header_applies_its_fallback_values(tmp_path):
    """With instrument_name=None a KeplerCam frame still gets the profile's saturation."""

    path = tmp_path / "kepcam.fits"
    header = _wcs_header((64, 64))
    header.update({"DETECTOR": "kepcam", "FILTER": "r", "EXPTIME": 60.0,
                   "MJD-OBS": 58800.0, "OBJECT": "AT_TEST"})
    fits.PrimaryHDU(np.full((64, 64), 100.0, dtype=np.float32), header=header).writeto(path)
    state, context = initialize_pipeline([path], settings=NO_PLOTS, instrument_name=None,
                                         run_directory=tmp_path / "run")
    run_pipeline_stage(state, context, "read", save=False)
    image = context["images"][path.name]
    assert image["settings"]["instrument"]["profile"] == "keplercam"
    assert image["record"]["metadata"]["saturation"] == pytest.approx(50000.0)
    assert image["record"]["metadata"]["gain"] == pytest.approx(4.45)


def test_relative_alignment_accepts_wcs_with_different_frames():
    """Regression: FK5/obstime WCS (IMACS) vs ICRS reference made astropy refuse offsets."""

    from redphot.alignment import _relative_match_table

    shape = (100, 100)
    reference_header = _wcs_header(shape)
    reference_header["RADESYS"] = "ICRS"
    other_header = _wcs_header(shape)
    other_header["RADESYS"] = "FK5"
    other_header["EQUINOX"] = 2000.0
    other_header["MJD-OBS"] = 58848.0
    rows = Table({"persistent_id": ["a", "b", "c"], "x": [10.0, 50.0, 80.0],
                  "y": [20.0, 60.0, 30.0]})
    table = _relative_match_table(rows, rows, WCS(other_header), WCS(reference_header))
    assert len(table) == 3
    assert np.all(np.abs(np.asarray(table["residual_ra_original_arcsec"])) < 0.1)


def test_sip_terms_without_sip_ctype_are_declared(tmp_path):
    from redphot.image import _sip_consistent_header

    header = _wcs_header((50, 50))
    header.update({"A_ORDER": 2, "B_ORDER": 2, "A_2_0": 1e-7, "B_0_2": 1e-7})
    fixed = _sip_consistent_header(header.copy())
    assert fixed["CTYPE1"] == "RA---TAN-SIP" and fixed["CTYPE2"] == "DEC--TAN-SIP"
    assert _sip_consistent_header(_wcs_header((50, 50)))["CTYPE1"] == "RA---TAN"


def _bright_star_field(seed=11):
    """Sky plane + noise, a saturated star with a broad halo, and faint stars."""

    rng = np.random.default_rng(seed)
    ny, nx = 768, 768
    yy, xx = np.indices((ny, nx))
    sky = 1000.0 + 0.02 * xx - 0.01 * yy
    star_x, star_y = 420.0, 400.0
    radius = np.hypot(xx - star_x, yy - star_y)
    halo = 400.0 / (1.0 + (radius / 25.0) ** 2)
    core = 3.0e6 / (2 * np.pi * 2.0 ** 2) * np.exp(-0.5 * (radius / 2.0) ** 2)
    data = sky + halo + core
    for x, y in rng.uniform(40, 728, (25, 2)):
        r2 = (xx - x) ** 2 + (yy - y) ** 2
        data += 3000.0 / (2 * np.pi * 1.5 ** 2) * np.exp(-0.5 * r2 / 1.5 ** 2)
    data += rng.normal(0.0, 20.0, data.shape)
    saturated = data >= 50000.0
    data = np.minimum(data, 65000.0)
    # The masks stage masks the saturated core and its surroundings.
    from scipy import ndimage

    mask = ndimage.binary_dilation(saturated, iterations=6)
    return data, mask, sky, (int(star_x), int(star_y))


def test_bright_star_halo_is_masked_so_the_background_does_not_absorb_it():
    data, mask, sky, (star_x, star_y) = _bright_star_field()
    settings = get_default_settings()
    settings["background"].update({"box_size": [64, 64], "filter_size": [3, 3]})
    settings["source_detection"]["fwhm_guess_pixels"] = 4.0
    metadata = {"saturation": 50000.0}
    ccd = CCDData(data, unit="adu", mask=mask)

    settings["background"]["bright_star_mask_enabled"] = False
    _, old_products, old_info = model_background(ccd, metadata, settings)
    settings["background"]["bright_star_mask_enabled"] = True
    _, products, info = model_background(ccd, metadata, settings)

    truth = sky[star_y, star_x]
    old_bias = old_products["background"][star_y, star_x] - truth
    new_bias = products["background"][star_y, star_x] - truth
    assert old_bias > 20.0                      # the halo leaks into the model (1 RMS)
    # what is left is the halo just outside the mask (~0.25 RMS there)
    assert abs(new_bias) < 0.3 * 20.0 and abs(new_bias) < 0.3 * old_bias

    stars = info["source_mask"]["bright_stars"]
    assert len(stars) == settings["background"]["bright_star_count"]
    main = stars[0]
    assert main["saturated"] and main["converged"]
    assert abs(main["x"] - star_x) < 2 and abs(main["y"] - star_y) < 2
    # halo(r) = 400 / (1 + (r/25)^2) drops below 0.25 x 20 ADU at r ~ 220 px
    assert 170.0 < main["halo_radius_pixels"] < 280.0
    # faint stars keep (almost) the ordinary growth
    for star in stars[1:]:
        assert star["stop"] == "inside_brighter_halo" or \
            star["halo_radius_pixels"] < star["segment_radius_pixels"] + 30.0
    halo_mask = products["bright_star_mask"]
    assert halo_mask[star_y, star_x + 150] and not halo_mask[star_y, star_x + 300]
    # other sources are grown exactly as before
    assert np.array_equal(products["detected_source_mask"],
                          old_products["detected_source_mask"])
    assert info["source_mask"]["combined_mask_fraction"] < \
        old_info["source_mask"]["combined_mask_fraction"] + 0.25
    # sky residuals still look like noise
    keep = ~products["background_mask"]
    normalized = (products["background_subtracted"] / products["background_rms"])[keep]
    median = np.median(normalized)
    width = 1.4826 * np.median(np.abs(normalized - median))
    assert abs(median) < 0.1 and abs(width - 1.0) < 0.1


def test_bright_star_halo_mask_can_be_switched_off_and_ignores_faint_fields():
    rng = np.random.default_rng(5)
    data = rng.normal(500.0, 10.0, (256, 256))
    yy, xx = np.indices(data.shape)
    data += 800.0 * np.exp(-0.5 * ((xx - 128) ** 2 + (yy - 128) ** 2) / 2.0 ** 2)
    settings = get_default_settings()
    settings["background"]["box_size"] = [64, 64]
    _, products, info = model_background(CCDData(data, unit="adu"), {}, settings)
    star = info["source_mask"]["bright_stars"][0]
    # a modest star has no halo above 0.25 RMS: nothing beyond its segment
    assert star["halo_radius_pixels"] <= star["segment_radius_pixels"] + 10.0
    settings["background"]["bright_star_mask_enabled"] = False
    _, products, info = model_background(CCDData(data, unit="adu"), {}, settings)
    assert not products["bright_star_mask"].any()
    assert info["source_mask"]["bright_stars"] == []


def test_date_obs_without_time_defers_to_the_mjd_card(tmp_path):
    """Regression (IMACS): DATE-OBS = '2019-12-31' was read as midnight."""

    path = tmp_path / "imacs.fits"
    header = _wcs_header((64, 64))
    header.update({"DATE-OBS": "2019-12-31", "MJD": 58848.24338999996, "EXPTIME": 10.0,
                   "FILTER": "r", "OBJECT": "AT_TEST"})
    fits.PrimaryHDU(np.full((64, 64), 100.0, dtype=np.float32), header=header).writeto(path)
    settings = resolve_settings()
    _, metadata = read_fits_image(path, settings)
    assert metadata["mjd_start"] == pytest.approx(58848.24339, abs=1e-6)
    assert metadata["mjd_mid"] == pytest.approx(58848.24339 + 5.0 / 86400.0, abs=1e-6)
    assert metadata["date_mid_utc"].startswith("2019-12-31T05:50:3")
    assert "TIME_CONFLICT" in metadata["quality_flags"]
    conflict = metadata["metadata_conflicts"][0]
    assert conflict["kind"] == "date_obs_vs_mjd_card" and conflict["resolution"] == "used MJD"
    # switched off, DATE-OBS keeps deciding (and the conflict is still reported)
    settings["metadata"]["prefer_mjd_on_time_conflict"] = False
    _, metadata = read_fits_image(path, settings)
    assert metadata["mjd_start"] == pytest.approx(58848.0)
    assert "TIME_CONFLICT" in metadata["quality_flags"]


def test_images_without_a_saturation_keyword_default_to_50000(tmp_path):
    path = tmp_path / "ldss3.fits"
    header = _wcs_header((64, 64))
    header.update({"MJD": 58898.1, "EXPTIME": 300.0, "FILTER": "g", "OBJECT": "AT_TEST",
                   "GAIN": 1.65})
    fits.PrimaryHDU(np.full((64, 64), 100.0, dtype=np.float32), header=header).writeto(path)
    settings = resolve_settings(run_settings={"metadata": {"instrument_override": "LDSS3c"}})
    _, metadata = read_fits_image(path, settings)
    assert metadata["saturation"] == pytest.approx(50000.0)
    assert "fallback" in metadata["_sources"]["saturation"]
    assert "SATURATION_MISSING" not in metadata["quality_flags"]
    # a header value still wins
    header["SATURATE"] = 65000.0
    fits.PrimaryHDU(np.full((64, 64), 100.0, dtype=np.float32), header=header).writeto(
        path, overwrite=True)
    _, metadata = read_fits_image(path, settings)
    assert metadata["saturation"] == pytest.approx(65000.0)


def test_fwhm_rise_with_brightness_is_found_and_flat_seeing_is_not():
    from redphot.image import fwhm_brightness_onset

    rng = np.random.default_rng(4)
    flux = np.geomspace(1e3, 1e7, 200)
    flat = 3.0 + rng.normal(0.0, 0.08, flux.size)
    peak = flux / 30.0
    result = fwhm_brightness_onset(flux, flat, peak, np.ones(flux.size, bool),
                                   get_default_settings(), sky=1000.0)
    assert result["onset_flux"] is None and result["reference_fwhm"] == pytest.approx(3.0, abs=0.05)
    # stars brighter than 5e5 grow wider with flux (nonlinear cores)
    rising = flat * np.where(flux > 5e5, (flux / 5e5) ** 0.25, 1.0)
    result = fwhm_brightness_onset(flux, rising, peak, np.ones(flux.size, bool),
                                   get_default_settings(), sky=1000.0)
    assert 5e5 < result["onset_flux"] < 3e6
    assert result["broadened_count"] >= 3
    assert result["onset_level"] == pytest.approx(result["onset_peak"] + 1000.0)
    # a single bright outlier is not a trend
    single = flat.copy()
    single[-1] = 6.0
    result = fwhm_brightness_onset(flux, single, peak, np.ones(flux.size, bool),
                                   get_default_settings())
    assert result["onset_flux"] is None


def test_broadened_stars_are_kept_out_of_the_zeropoint_and_psf():
    from redphot.catalogs import _role_candidates, _screen_measurements

    rows = []
    for index, broadened in enumerate((False, True)):
        rows.append({
            "persistent_id": "s{}".format(index), "image_id": "a", "filter": "g",
            "magnitude": 17.0, "magnitude_error": 0.01, "edge_distance_pixels": 500.0,
            "near_edge": False, "saturated": False, "broadened": broadened,
            "saturation_distance_pixels": np.inf, "masked": False, "trail_overlap": False,
            "trail_distance_pixels": np.inf, "snr": 500.0, "ellipticity": 0.05,
            "fwhm_pixels": 3.0, "neighbor_distance_pixels": 100.0, "image_fwhm_pixels": 3.0,
            "detector_x_fraction": 0.5, "detector_y_fraction": 0.5,
        })
    from redphot.catalogs import _measurement_table

    measurements = _measurement_table(rows)
    master = Table({"persistent_id": ["s0", "s1"], "catalog_rejection_reasons": ["", ""]})
    screened = _screen_measurements(master, measurements, get_default_settings())
    assert list(screened["rejection_reasons"]) == ["", "NONLINEAR"]
    settings = get_default_settings()
    for role in ("calibration", "psf"):
        assert list(_role_candidates(screened, role, settings)) == [True, False]
    assert list(_role_candidates(screened, "astrometry", settings)) == [True, True]


def test_zeropoint_is_not_set_by_the_brightest_stars_alone():
    from redphot.photometry import _solve_zeropoint_group

    settings = get_default_settings()
    records = []
    rng = np.random.default_rng(8)
    for index in range(60):
        bright = index < 6
        records.append({
            "image_id": "a", "method": "psf", "filter": "g", "routed_catalog": "ps1",
            "magnitude_band": "g", "magnitude_system": "AB", "source_id": str(index),
            "input_accepted": True, "rejection_reason": "",
            # bright stars: tiny formal errors and a +0.1 mag offset (nonlinear)
            "individual_zeropoint": 27.0 + (0.10 if bright else rng.normal(0.0, 0.05)),
            "instrumental_magnitude_uncertainty": 0.001 if bright else 0.02,
            "catalog_magnitude_error": 0.002 if bright else 0.01,
            "snr": 1000.0 if bright else 50.0, "airmass": 1.2,
        })
    solution = _solve_zeropoint_group(records, list(range(60)), set(), settings)
    assert solution["zeropoint_mag"] == pytest.approx(27.0, abs=0.03)
    assert solution["intrinsic_scatter_mag"] > 0.03
    settings["calibration"]["add_intrinsic_scatter"] = False
    old = _solve_zeropoint_group(records, list(range(60)), set(), settings)
    assert old["zeropoint_mag"] > 27.05      # the old weighting followed the bright stars


def test_template_footprints_report_how_much_of_each_image_has_template_data():
    from redphot.subtraction import template_footprint_coverage

    template_shape = (200, 200)
    template_wcs = WCS(_wcs_header(template_shape))
    data = np.ones(template_shape)
    data[:, 150:] = np.nan                       # right quarter has no data
    data[95:105, 95:105] = np.nan                # a masked star core: not a gap
    template = {"data": data, "mask": ~np.isfinite(data), "wcs": template_wcs}
    inside = WCS(_wcs_header((40, 40)))          # centered on the template
    shifted_header = _wcs_header((40, 40))
    shifted_header["CRPIX1"] -= 50               # half of it over the gap
    records = [
        {"image_id": "inside", "data": np.zeros((40, 40)), "wcs": inside,
         "metadata": {"filter": "r"}},
        {"image_id": "edge", "data": np.zeros((40, 40)), "wcs": WCS(shifted_header),
         "metadata": {"filter": "r"}},
    ]
    coverage = template_footprint_coverage(template, records)
    assert coverage[0]["coverage_fraction"] == pytest.approx(1.0)
    assert 0.3 < coverage[1]["coverage_fraction"] < 0.7
    assert len(coverage[0]["outline_x"]) == 100
    from redphot.diagnostics import plot_template_footprints
    import matplotlib.pyplot as plt

    template["science_footprints"] = coverage
    figure = plot_template_footprints({"r": template}, status="WARN",
                                      flags=["TEMPLATE_FOOTPRINT_INCOMPLETE"])
    plt.close(figure)


def test_difference_comparison_reports_magnitudes_and_limits():
    from redphot.photometry import _difference_comparison

    def table(fluxes, errors):
        return Table({
            "image_id": ["a"] * len(fluxes), "source_type": ["target"] * len(fluxes),
            "method": ["psf", "large_aperture"][: len(fluxes)], "flux": fluxes,
            "flux_uncertainty": errors, "exposure_time": [100.0] * len(fluxes),
        })

    zeropoints = Table({"image_id": ["a", "a"], "method": ["psf", "large_aperture"],
                        "zeropoint_mag": [25.0, 25.1], "zeropoint_uncertainty_mag": [0.01, 0.01]})
    comparison = _difference_comparison(table([10000.0, 12000.0], [100.0, 150.0]),
                                        table([5000.0, 200.0], [100.0, 150.0]), zeropoints)
    rows = {str(row["method"]): row for row in comparison}
    assert rows["psf"]["science_magnitude"] == pytest.approx(25.0 - 2.5 * np.log10(100.0))
    assert rows["psf"]["difference_magnitude"] == pytest.approx(25.0 - 2.5 * np.log10(50.0))
    # 200 ± 150 is below 3σ: no magnitude, a 3σ limit instead
    assert np.ma.is_masked(rows["large_aperture"]["difference_magnitude"])
    assert rows["large_aperture"]["difference_limit_mag"] == pytest.approx(
        25.1 - 2.5 * np.log10(3 * 150.0 / 100.0))


def test_cached_template_is_reused_when_the_field_center_moves_slightly(tmp_path):
    """Regression: a 0.04" shift of the refined WCS changed the cache name."""

    from astropy.coordinates import SkyCoord
    from redphot.subtraction import _find_cached_template, _template_cache_path

    footprint = {"center": SkyCoord(103.60011, 17.49246, unit="deg"),
                 "width_arcmin": 15.65, "height_arcmin": 15.64}
    cached = _template_cache_path(tmp_path, "ps1", "g", footprint)
    cached.write_bytes(b"")
    moved = dict(footprint, center=SkyCoord(103.60010, 17.49246, unit="deg"))
    assert _template_cache_path(tmp_path, "ps1", "g", moved) != cached
    assert _find_cached_template(tmp_path, "ps1", "g", moved) == cached
    # another band, or a field that is not covered, is not reused
    assert _find_cached_template(tmp_path, "ps1", "r", moved) is None
    far = dict(footprint, center=SkyCoord(103.70, 17.49246, unit="deg"))
    assert _find_cached_template(tmp_path, "ps1", "g", far) is None
    bigger = dict(footprint, width_arcmin=20.0)
    assert _find_cached_template(tmp_path, "ps1", "g", bigger) is None


def test_stages_know_which_settings_they_read():
    from redphot.pipeline import first_stage_reading, stage_reads_setting

    assert first_stage_reading({"psf": {"maximum_stars": 30}}) == "star_selection"
    assert first_stage_reading({"psf": {"box_size_pixels": 31}}) == "psf"
    assert first_stage_reading({"masks": {"cosmic_rays": {"objlim": 10}}}) == "cosmic_rays"
    assert first_stage_reading({"masks": {"saturation_level": 45000}}) == "masks"
    assert first_stage_reading(
        {"image_quality": {"usability": {"zeropoint_scatter_warn_mag": 0.2}}}) == "usability"
    assert first_stage_reading(
        {"catalogs": {"comparison_stars": {"minimum_snr": 20}}}) == "star_selection"
    assert first_stage_reading({"background": {"box_size": [64, 64]}}) == "background"
    assert not stage_reads_setting("masks", "masks.cosmic_rays.enabled")
    assert stage_reads_setting("masks", "masks")
    assert not stage_reads_setting("astrometry", "catalogs.comparison_stars.minimum_snr")
    assert stage_reads_setting("astrometry", "catalogs.search_radius_arcmin")


def test_step_parameters_are_found_by_name():
    from redphot.steps import find_setting, step_name, step_overrides

    assert step_name(9) == "star_selection"
    assert step_name("09_star_selection") == "star_selection"
    assert step_name("sources") == "source_quality"
    assert find_setting(9, "minimum_snr") == "catalogs.comparison_stars.minimum_snr"
    assert find_setting(9, "maximum_stars") == "psf.maximum_stars"
    assert find_setting("cosmic_rays", "enabled") == "masks.cosmic_rays.enabled"
    assert find_setting("subtraction", "enabled") == "subtraction.enabled"
    assert find_setting("subtraction", "kernel_order") == "subtraction.hotpants.kernel_order"
    assert find_setting("usability", "enabled") == "image_quality.usability.enabled"
    assert find_setting("read", "saturation_override") == "metadata.saturation_override"
    with pytest.raises(ValueError, match="background.box_size"):
        find_setting("star_selection", "box_size")
    with pytest.raises(ValueError, match="minimum_snr"):
        find_setting("star_selection", "minimum_sn")
    assert step_overrides("background", {"box_size": [64, 64]}) == {
        "background": {"box_size": [64, 64]}}
    assert step_overrides(9, {"psf": {"maximum_stars": 12}, "minimum_snr": 20}) == {
        "psf": {"maximum_stars": 12},
        "catalogs": {"comparison_stars": {"minimum_snr": 20}}}


def _per_image_functions():
    """Synthetic stages whose batch results follow each image's own settings."""

    functions = _stage_functions()

    def active(context, required):
        state = context["_state"]
        return [
            image_id for image_id, image in state["images"].items()
            if image.get("status") != "REJECTED"
            and image.get("stages", {}).get(required, {}).get("status")
            in {"PASS", "WARN", "APPROVED"}
        ]

    def star_selection(context, image_id, settings):
        rows = []
        for identifier in active(context, "astrometry"):
            local = context["images"][identifier]["settings"]
            cut = local["catalogs"]["comparison_stars"]["minimum_snr"]
            rows.append({"image_id": identifier, "persistent_id": "star", "cut": float(cut)})
        summaries = [{"image_id": row["image_id"], "status": "PASS"} for row in rows]
        return {"status": "PASS", "measurements": Table(rows=rows), "summaries": summaries}

    def usability(context, image_id, settings):
        decisions = [{"image_id": identifier, "status": "PASS"}
                     for identifier in active(context, "astrometry")]
        return {"status": "PASS", "decisions": decisions}

    def alignment(context, image_id, settings):
        alignments = [{"image_id": identifier, "status": "PASS"}
                      for identifier in active(context, "usability")]
        return {"status": "PASS", "alignments": alignments,
                "target_solution": {"ra_deg": 10.0, "dec_deg": 20.0}}

    def calibration(context, image_id, settings):
        rows = [{"image_id": identifier, "zeropoint_mag": 25.0}
                for identifier in active(context, "science_photometry")]
        return {"status": "PASS", "zeropoints": Table(rows=rows)}

    def templates(context, image_id, settings):
        return {"status": "PASS", "templates": {"r": {"data": np.zeros((4, 4))}},
                "flags": []}

    functions.update(star_selection=star_selection, usability=usability,
                     alignment=alignment, calibration=calibration, templates=templates)
    return functions


def test_rerun_from_changes_only_the_selected_images(tmp_path, monkeypatch, capsys):
    import redphot.pipeline as pipeline
    from redphot.steps import rerun_from, run_step, start_run

    functions = _per_image_functions()
    monkeypatch.setattr(pipeline, "_default_stage_functions", lambda: dict(functions))
    files = []
    for name in ("one.fits", "two.fits", "three.fits"):
        _write_lco(tmp_path / name)
        files.append(tmp_path / name)
    state, context = start_run(files, tmp_path / "run", settings=NO_PLOTS)
    pipeline.run_pipeline_through(state, context)
    one, two, three = "one.fits", "two.fits", "three.fits"
    assert list(state["images"]) == [one, three, two]  # sorted by name
    calls = context["shared"]["calls"]
    before = dict(calls)

    # Step 9 with a new comparison-star cut for image two only (number 3).
    rerun_from(state, context, 9, images=[3], minimum_snr=20)
    assert state["images"][two]["overrides"] == {
        "catalogs": {"comparison_stars": {"minimum_snr": 20}}}
    assert state["images"][one]["overrides"] == {}
    rows = context["shared"]["star_selection"]["measurements"]
    assert {row["image_id"]: row["cut"] for row in rows} == {one: 10.0, two: 20.0, three: 10.0}
    changed = {key for key in calls if calls[key] != before.get(key)}
    # Image 2 redid the image steps after step 9; images 1 and 3 kept theirs,
    # and no image redid steps 1-8.
    assert changed == {two + ":psf", two + ":science_photometry", two + ":subtraction",
                       two + ":difference_photometry"}
    assert state["batch_stages"]["outputs"]["status"] == "PASS"

    # Nothing is out of date afterwards: a full pass reruns nothing.
    before = dict(calls)
    pipeline.run_pipeline_through(state, context)
    assert dict(calls) == before
    state, context = pipeline.load_pipeline_state(tmp_path / "run")
    pipeline.refresh_pipeline_staleness(state, context)
    pipeline.run_pipeline_through(state, context)
    assert context["shared"]["calls"] == before

    # One image step with a changed setting runs only for that image.
    calls = context["shared"]["calls"]
    before = dict(calls)
    run_step(state, context, "background", images="two", box_size=[64, 64])
    assert {key for key in calls if calls[key] != before.get(key)} == {two + ":background"}
    assert state["images"][two]["stages"]["astrometry"]["status"] == "STALE"
    assert state["images"][one]["stages"]["astrometry"]["status"] == "PASS"
    assert state["batch_stages"]["star_selection"]["status"] == "STALE"
    assert "out of date now" in capsys.readouterr().out

    # Settings of steps that use one value for all images cannot be per image.
    with pytest.raises(ValueError, match="same settings for every image"):
        run_step(state, context, "alignment", images=[1], relative_alignment_sigma_clip=4.0)


def test_reloaded_run_keeps_the_image_order(tmp_path):
    for name in ("b_image.fits", "A_image.fits", "c_image.fits"):
        _write_lco(tmp_path / name)
    state, context = initialize_pipeline(
        [tmp_path / "c_image.fits", tmp_path / "b_image.fits", tmp_path / "A_image.fits"],
        settings=NO_PLOTS, run_directory=tmp_path / "run")
    order = list(state["images"])
    from redphot.pipeline import save_pipeline_state

    save_pipeline_state(state, context)
    state, context = load_pipeline_state(tmp_path / "run")
    assert list(state["images"]) == order
    # Runs saved before the order was stored follow the checkpoint's order.
    state.pop("image_order")
    state["images"] = dict(sorted(state["images"].items()))
    save_pipeline_state(state, context)
    state, context = load_pipeline_state(tmp_path / "run")
    assert list(state["images"]) == order


def test_checkpoint_keeps_wcs_numbers_exact():
    import io
    import pickle

    from astropy.wcs import Sip
    from redphot.pipeline import _CheckpointPickler

    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN-SIP", "DEC--TAN-SIP"]
    wcs.wcs.crval = [103.59625693146123456, 17.491836103548765]
    wcs.wcs.crpix = [468.680954065171234, 512.123456789012345]
    wcs.wcs.cd = np.array([[-1.2345678901234567e-4, 2.345678901234567e-7],
                           [3.456789012345678e-7, 1.2345678901234567e-4]])
    a = np.zeros((3, 3))
    b = np.zeros((3, 3))
    a[2, 0], b[0, 2] = 1.234567890123456e-6, -2.345678901234567e-6
    wcs.sip = Sip(a, b, None, None, wcs.wcs.crpix)
    wcs.pixel_shape = (1000, 1100)
    wcs.wcs.set()
    buffer = io.BytesIO()
    _CheckpointPickler(buffer, protocol=pickle.HIGHEST_PROTOCOL).dump({"wcs": wcs})
    copy = pickle.loads(buffer.getvalue())["wcs"]
    x, y = np.meshgrid(np.linspace(0, 999, 7), np.linspace(0, 1099, 5))
    for first, second in zip(wcs.all_pix2world(x, y, 0), copy.all_pix2world(x, y, 0)):
        assert np.array_equal(first, second)
    assert copy.pixel_shape == (1000, 1100)
    # Astropy's own pickling rounds the numbers (the reason for the pickler).
    plain = pickle.loads(pickle.dumps(wcs))
    assert not np.array_equal(plain.wcs.crpix, wcs.wcs.crpix)


def test_wcs_refinement_applies_rotation_for_pc_headers():
    """A similarity fit changes the matrix whatever form the header uses."""

    from redphot.catalogs import _compose_similarity_wcs

    angle = np.deg2rad(0.2)
    transform = 1.001 * np.array([[np.cos(angle), -np.sin(angle)],
                                  [np.sin(angle), np.cos(angle)]])
    translation = np.array([1.5, -2.0])
    matrix = np.array([[-1.0e-4, 1.0e-7], [2.0e-7, 1.0e-4]])
    results = {}
    for form in ("cd", "pc", "both"):
        wcs = WCS(naxis=2)
        wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        wcs.wcs.crval = [103.6, 17.5]
        wcs.wcs.crpix = [500.0, 480.0]
        if form in ("pc", "both"):
            wcs.wcs.pc = matrix / 1.0e-4
            wcs.wcs.cdelt = [1.0e-4, 1.0e-4]
        if form in ("cd", "both"):
            wcs.wcs.cd = matrix
        wcs.wcs.set()
        refined = _compose_similarity_wcs(wcs, transform, translation)
        x, y = np.meshgrid(np.linspace(0, 999, 5), np.linspace(0, 999, 5))
        results[form] = np.array(refined.pixel_to_world_values(x, y))
        # The refined WCS puts each pixel where the original puts the
        # transformed pixel.
        moved = transform @ np.vstack([x.ravel(), y.ravel()]) + translation[:, None]
        expected = np.array(wcs.pixel_to_world_values(moved[0], moved[1]))
        assert np.allclose(results[form].reshape(2, -1), expected, atol=1e-9, rtol=0)
    assert np.allclose(results["pc"], results["cd"], atol=1e-10, rtol=0)
    assert np.allclose(results["both"], results["cd"], atol=1e-10, rtol=0)


def test_settings_resolved_by_the_read_stage_do_not_redo_the_run(tmp_path):
    """The read stage picks the instrument profile; a second pass keeps everything."""

    from redphot.config import merge_settings

    first, second, functions, state, context = _two_image_run(tmp_path)
    plain_read = functions["read"]

    def read_with_profile(context, image_id, settings):
        # Like _run_read: settings re-resolved for the instrument in the header.
        image = context["images"][image_id]
        image["settings"] = merge_settings(image["settings"],
                                           {"metadata": {"gain_override": 1.5}})
        image["record"]["settings"] = image["settings"]
        return plain_read(context, image_id, settings)

    functions["read"] = read_with_profile
    run_pipeline_through(state, context, stage_functions=functions)
    calls = dict(context["shared"]["calls"])
    run_pipeline_through(state, context, stage_functions=functions)
    assert context["shared"]["calls"] == calls


def test_star_screening_uses_each_images_settings():
    from redphot.catalogs import _screen_measurements
    from redphot.config import merge_settings

    master = Table({"persistent_id": ["star"], "catalog_rejection_reasons": [""]})
    row = {"persistent_id": "star", "image_fwhm_pixels": 3.0, "magnitude": 17.0,
           "magnitude_error": 0.01, "edge_distance_pixels": 200.0, "near_edge": False,
           "saturated": False, "saturation_distance_pixels": 500.0, "masked": False,
           "trail_distance_pixels": 500.0, "trail_overlap": False, "snr": 15.0,
           "ellipticity": 0.05, "fwhm_pixels": 3.0, "neighbor_distance_pixels": 100.0,
           "detector_x_fraction": 0.5, "detector_y_fraction": 0.5}
    measurements = Table(rows=[dict(row, image_id="a"), dict(row, image_id="b")], masked=True)
    settings = get_default_settings()
    strict = merge_settings(settings, {"catalogs": {"comparison_stars": {"minimum_snr": 20}}})
    screened = _screen_measurements(master, measurements, settings, {"b": strict})
    assert list(screened["image_accepted"]) == [True, False]
    assert "SNR_LOW" in screened["rejection_reasons"][1]


def test_target_moves_below_the_tolerance_keep_other_images():
    from redphot.pipeline import _shared_digest_part

    def part(ra_deg, tolerance, previous=None):
        result = {"target_solution": {"ra_deg": ra_deg, "dec_deg": 17.49198, "frozen": True,
                                      "version": "target-v1", "uncertainty_arcsec": 0.06}}
        settings = {"pipeline": {"rerun_position_tolerance_mas": tolerance}}
        return _shared_digest_part("alignment", result, settings, ["target_solution"],
                                   previous)

    base = 103.5962980
    first, position = part(base, 10.0)
    assert position == [base, 17.49198]
    mas = 1.0 / 3.6e6 / np.cos(np.deg2rad(17.49198))
    assert part(base + 2 * mas, 10.0, position)[0] == first      # 2 mas: kept
    assert part(base + 9 * mas, 10.0, position)[0] == first      # 9 mas: kept
    assert part(base + 12 * mas, 10.0, position)[0] != first     # 12 mas: redone
    assert part(base + 2 * mas, 0.0, position)[0] != first       # exact comparison
