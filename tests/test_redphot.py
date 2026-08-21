"""Integration and regression tests that do not require network access.

Only the named KeplerCam file is treated as observational test data.  Other
conditions are deterministic synthetic proxies and are not substitutes for
the release-validation observations listed in ``docs/validation.rst``.
"""

from hashlib import sha256
from pathlib import Path

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
    run_pipeline_through,
    set_image_overrides,
    skip_pipeline_stage,
)
from redphot.subtraction import evaluate_subtraction


DATA = Path(__file__).parent / "data"
FLWO_FILE = DATA / "AT_2024rmj_r_FLWO_2024.1012.fits"


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
    state, context = initialize_pipeline([first, second], run_directory=run_directory)
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
