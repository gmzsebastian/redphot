"""
Configuration defaults and conventions for redphot.

Configuration order is:

    general defaults
    instrument defaults
    run settings
    filter settings
    individual-image overrides

All functions return independent dictionaries. Changing the resolved settings
for one image will not modify the defaults or the settings for another image.
"""

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path


MISSING_VALUE = None

MISSING_VALUE_POLICY = {
    "configuration": None,
    "table_numeric": "masked",
    "table_string": "masked",
    "floating_array": "nan",
    "boolean": None,
    "never_use": [-999, -1],
}


STANDARD_UNITS = {
    "angle": "deg",
    "sky_separation": "arcsec",
    "pixel_scale": "arcsec/pixel",
    "exposure_time": "s",
    "time": "MJD-UTC",
    "gain": "electron/adu",
    "read_noise": "electron",
    "image": "adu",
    "background": "adu/pixel",
    "flux": "adu",
    "flux_rate": "adu/s",
    "magnitude": "mag",
    "temperature": "deg_C",
    "distance": "m",
}


FILTER_ALIASES = {
    "u": "u",
    "up": "u",
    "us": "u",
    "u-sloan": "u",
    "sloan-u": "u",
    "sdss-u": "u",
    "g": "g",
    "gp": "g",
    "gs": "g",
    "g-sloan": "g",
    "sloan-g": "g",
    "sdss-g": "g",
    "g-ztf": "g",
    "ztf-g": "g",
    "g-skymapper": "g",
    "r": "r",
    "rp": "r",
    "rs": "r",
    "r-sloan": "r",
    "sloan-r": "r",
    "sdss-r": "r",
    "r-ztf": "r",
    "ztf-r": "r",
    "r-skymapper": "r",
    "i": "i",
    "ip": "i",
    "is": "i",
    "i-sloan": "i",
    "sloan-i": "i",
    "sdss-i": "i",
    "i-ztf": "i",
    "ztf-i": "i",
    "i-skymapper": "i",
    "z": "z",
    "zp": "z",
    "zs": "z",
    "z-sloan": "z",
    "sloan-z": "z",
    "sdss-z": "z",
    "z-ztf": "z",
    "ztf-z": "z",
    "z-skymapper": "z",
    "y": "y",
    "yp": "y",
    "ys": "y",
    "y-sloan": "y",
    "sloan-y": "y",
    "B": "B",
    "Johnson-B": "B",
    "Bessell-B": "B",
    "V": "V",
    "Johnson-V": "V",
    "Bessell-V": "V",
    "R": "R",
    "Rc": "R",
    "R-Cousins": "R",
    "Cousins-R": "R",
    "Bessell-R": "R",
    "I": "I",
    "Ic": "I",
    "I-Cousins": "I",
    "Cousins-I": "I",
    "Bessell-I": "I",
    "J": "J",
    "H": "H",
    "K": "K",
    "Ks": "Ks",
    "clear": "clear",
    "open": "clear",
    "unfiltered": "clear",
}


QUALITY_FLAGS = {
    "input": [
        "FITS_UNREADABLE",
        "FITS_CHECKSUM_FAILED",
        "NO_IMAGE_DATA",
        "MULTIPLE_IMAGE_HDUS",
        "DUPLICATE_IMAGE",
        "UNSUPPORTED_RAW_IMAGE",
    ],
    "metadata": [
        "METADATA_MISSING",
        "METADATA_CONFLICT",
        "EXPOSURE_TIME_CONFLICT",
        "TIME_CONFLICT",
        "FILTER_UNKNOWN",
        "GAIN_MISSING",
        "READ_NOISE_MISSING",
        "SATURATION_MISSING",
    ],
    "image": [
        "TARGET_OUTSIDE_IMAGE",
        "TARGET_NEAR_EDGE",
        "BAD_EDGES",
        "TOO_MANY_MASKED_PIXELS",
        "SATURATION_HIGH",
        "BAD_ROWS_OR_COLUMNS",
        "TRAIL_PRESENT",
        "TARGET_TRAIL",
        "TARGET_COSMIC_RAY",
        "BACKGROUND_UNRELIABLE",
        "BACKGROUND_GRADIENT_HIGH",
        "FRINGE_CORRECTION_FAILED",
    ],
    "astrometry": [
        "WCS_MISSING",
        "WCS_INVALID",
        "WCS_POOR",
        "WCS_TOO_FEW_MATCHES",
        "WCS_REFINEMENT_FAILED",
        "RELATIVE_ALIGNMENT_FAILED",
        "RELATIVE_ALIGNMENT_POOR",
    ],
    "quality": [
        "TOO_FEW_SOURCES",
        "SEEING_POOR",
        "SEEING_SCATTER_HIGH",
        "ELLIPTICITY_HIGH",
        "ELLIPTICITY_SCATTER_HIGH",
        "TRACKING_POOR",
        "BACKGROUND_HIGH",
        "BACKGROUND_RMS_HIGH",
        "QUALITY_BATCH_OUTLIER",
        "UPSTREAM_QUALITY_MISMATCH",
        "CATALOG_RECOVERY_LOW",
        "QC_STAR_NOT_RECOVERED",
        "TRANSPARENCY_LOW",
        "TRANSPARENCY_NONUNIFORM",
        "ZEROPOINT_OUTLIER",
        "ZEROPOINT_SCATTER_HIGH",
        "IMAGE_TOO_SHALLOW",
    ],
    "photometry": [
        "TOO_FEW_PSF_STARS",
        "TOO_FEW_CALIBRATION_STARS",
        "PSF_MODEL_FAILED",
        "PSF_RESIDUAL_HIGH",
        "PSF_REVIEW_REQUIRED",
        "TARGET_MASKED",
        "TARGET_FIT_FAILED",
        "TARGET_CENTROID_OFFSET",
        "TARGET_POSITION_UNCERTAIN",
        "TARGET_FILTER_SHIFT",
        "TARGET_CENTROID_HOST_DOMINATED",
        "APERTURE_INCOMPLETE",
        "INSUFFICIENT_UNMASKED_PIXELS",
        "MASKED_PIXELS",
        "BAD_LOCAL_BACKGROUND",
        "COSMIC_RAY_OVERLAP",
        "TRAIL_OVERLAP",
        "CALIBRATION_FAILED",
        "CALIBRATION_STAR_REJECTED",
        "APERTURE_CORRECTION_FAILED",
        "EMPTY_APERTURE_LIMIT_FAILED",
        "NONDETECTION",
        "DIFFERENCE_PHOTOMETRY_FAILED",
        "DIFFERENCE_UNCERTAINTY_UNDERESTIMATED",
        "DIFFERENCE_DIPOLE",
        "DIFFERENCE_INVERTED_RESIDUAL",
        "DIFFERENCE_SUBTRACTION_REJECTED",
        "PREFERRED_RESULT_UNAVAILABLE",
        "COMPARISON_STAR_UNSTABLE",
        "BATCH_MEASUREMENT_OUTLIER",
        "BATCH_EPOCH_PROBLEM",
        "BATCH_GROUP_PROBLEM",
    ],
    "subtraction": [
        "TEMPLATE_MISSING",
        "TEMPLATE_COVERAGE_INCOMPLETE",
        "TEMPLATE_FILTER_MISMATCH",
        "TEMPLATE_DEPTH_INSUFFICIENT",
        "TEMPLATE_SEEING_POOR",
        "TEMPLATE_SATURATION_HIGH",
        "TEMPLATE_POST_TRANSIENT",
        "TEMPLATE_WCS_INVALID",
        "TEMPLATE_ALIGNMENT_FAILED",
        "SUBTRACTION_BACKEND_MISSING",
        "SUBTRACTION_FAILED",
        "SUBTRACTION_RESIDUAL_HIGH",
        "SUBTRACTION_DIPOLE",
        "SUBTRACTION_FLUX_LOSS",
        "SUBTRACTION_NOISE_HIGH",
    ],
}


INSTRUMENT_ALIASES = {
    "lco": "lco",
    "lcogt": "lco",
    "las-cumbres": "lco",
    "las-cumbres-observatory": "lco",
    "sq37": "lco",
    "qhy600m": "lco",
    "keplercam": "keplercam",
    "kepcam": "keplercam",
    "flwo": "keplercam",
    "flwo-1.2m": "keplercam",
}


DEFAULT_SETTINGS = {
    "instrument": {
        "name": None,
        "profile": None,
    },
    "input": {
        "paths": [],
        "recursive": False,
        "patterns": [
            "*.fits",
            "*.fit",
            "*.fts",
            "*.fits.fz",
            "*.fit.fz",
            "*.fits.gz",
        ],
        "data_hdu": None,
        "header_hdu": None,
        "mask_hdu": None,
        "variance_hdu": None,
        "preferred_extnames": ["SCI", "IMAGE", "IM1", "IM2"],
        "hdu_search_order": [0, 1, 2],
        "search_remaining_hdus": True,
        "prefer_target_hdu": True,
        "read_only": True,
        "memmap": False,
        "verify_checksum": True,
        "allow_compressed": True,
        "minimum_finite_fraction": 0.50,
        "detect_duplicate_files": True,
    },
    "metadata": {
        "keywords": {
            "object": ["OBJECT", "OBJNAME", "TARGNAME"],
            "telescope": ["TELESCOP", "TELESCOPE", "TELID"],
            "instrument": ["INSTRUME", "INSTRUMENT", "DETECTOR"],
            "detector": ["DETECTOR", "DETECTID", "CCDNAME"],
            "site": ["SITE", "SITEID", "SITENAME", "OBSERVAT"],
            "exposure_time": ["EXPTIME", "EXPOSURE"],
            "date_obs": ["DATE-OBS", "DATEOBS"],
            "date_end": ["DATE-END", "DATEEND", "UTEND", "UTSTOP"],
            "mjd": ["MJD-OBS", "MJD", "OBSMJD", "JD"],
            "filter": ["FILTER", "FILTER1", "FILTER2"],
            "gain": ["GAIN", "EGAIN", "GAINDL"],
            "read_noise": ["RDNOISE", "READNOIS", "READNOI", "ENOISE"],
            "saturation": ["MAXLIN", "SATURATE", "SATLEVEL"],
            "nonlinearity": ["MAXLIN", "LINLIMIT", "NONLIN", "NONLINEAR"],
            "airmass": ["AIRMASS", "AIR", "SECZ"],
            "pointing_ra": ["RA"],
            "pointing_dec": ["DEC"],
            "target_ra": ["CAT-RA", "OBJRA"],
            "target_dec": ["CAT-DEC", "OBJDEC"],
            "pixel_scale": ["PIXSCALE", "SECPIX1"],
            "binning": ["CCDSUM", "BINNING"],
            "reduction_level": ["RLEVEL"],
            "pipeline_version": ["PIPEVER"],
        },
        "fallback_values": {
            "telescope": None,
            "instrument": None,
            "detector": None,
            "site": None,
            "gain": None,
            "read_noise": None,
            "saturation": None,
            "nonlinearity": None,
            "pixel_scale": None,
        },
        "filter_override": None,
        "object_override": None,
        "exposure_time_override": None,
        "airmass_override": None,
        "gain_override": None,
        "read_noise_override": None,
        "saturation_override": None,
        "nonlinearity_override": None,
        "check_duplicate_cards": True,
        "check_conflicts": True,
        "warn_on_validation": True,
        "exposure_time_tolerance_s": 1.0,
        "time_tolerance_s": 2.0,
        "time_reference": "start",
        "convert_time_to_mid_exposure": True,
        "resolve_exposure_from_times": True,
        "derive_exposure_from_times": True,
        "canonical_time_scale": "utc",
        "canonical_time_format": "mjd",
        "strip_string_values": True,
        "normalize_filter": True,
        "require_reduced_input": True,
        "required_fields": ["object", "exposure_time", "mjd", "filter"],
        "conflict_fields": [
            "exposure_time",
            "gain",
            "read_noise",
            "saturation",
            "nonlinearity",
            "pixel_scale",
            "binning",
        ],
        "header_coordinate_unit": ["hourangle", "deg"],
        "diagnostic_keywords": {
            "pipeline_fwhm_arcsec": ["L1FWHM"],
            "pipeline_ellipticity": ["L1ELLIP"],
            "pipeline_background": [
                "L1SKYBKG",
                "L1MEDIAN",
                "SKYBKG",
                "BACKGRND",
            ],
            "pipeline_background_rms": ["L1SKYRMS", "SKYRMS", "BKGRMS"],
            "pipeline_zeropoint_mag": ["L1ZP"],
            "pipeline_saturated_fraction": ["SATFRAC"],
            "pipeline_wcs_error": ["WCSERR"],
        },
    },
    "crop": {
        "enabled": True,
        "center_on": "target",
        "size_arcmin": None,
        "shape": "square",
        "crop_mode": "trim",
        "use_datasec": True,
        "use_trimsec": True,
        "use_ccdsec": False,
        "use_detsec": False,
        "section_keywords": {
            "trimsec": ["TRIMSEC"],
            "datasec": ["DATASEC"],
            "ccdsec": ["CCDSEC"],
            "detsec": ["DETSEC"],
            "biassec": ["BIASSEC"],
        },
        "edge_crop_pixels": 0,
        "edge_buffer_fwhm": 2.0,
        "detect_empirical_edges": True,
        "edge_scan_fraction": 0.15,
        "edge_min_finite_fraction": 0.50,
        "edge_max_constant_fraction": 0.50,
        "edge_sigma": 5.0,
        "edge_grow_pixels": 2,
        "target_edge_distance_fwhm": 5.0,
        "valid_section": None,
        "require_target_inside": True,
    },
    "masks": {
        "use_existing_mask": True,
        "mask_nonfinite": True,
        "mask_invalid_edges": True,
        "mask_saturated": True,
        "mask_nonlinear": True,
        "saturation_grow_pixels": 5,
        "saturation_halo_fwhm": 5.0,
        "detect_bad_rows": True,
        "detect_bad_columns": True,
        "detect_amplifier_boundaries": False,
        "detect_trails": True,
        "trail_sigma": 5.0,
        "trail_min_length_pixels": 50,
        "trail_max_width_pixels": 20,
        "trail_grow_pixels": 5,
        "trail_min_elongation": 4.0,
        "trail_min_pixels": 20,
        "manual_regions": [],
        "saturation_level": None,
        "nonlinearity_level": None,
        "prefer_nonlinearity_limit": True,
        "saturation_high_fraction": 0.02,
        "bad_line_sigma": 6.0,
        "bad_line_grow_pixels": 0,
        "amplifier_boundaries": [],
        "amplifier_seam_sigma": 8.0,
        "amplifier_seam_grow_pixels": 1,
        "amplifier_seam_edge_margin": 20,
        "target_overlap_radius_fwhm": None,
        "star_overlap_radius_fwhm": 3.0,
        "cosmic_rays": {
            "enabled": False,
            "mode": "mask",
            "backend": "astroscrappy",
            "sigclip": 4.5,
            "sigfrac": 0.3,
            "objlim": 5.0,
            "niter": 4,
            "cleantype": "meanmask",
            "use_image_gain": True,
            "use_image_read_noise": True,
            "use_image_saturation": True,
            "gain": None,
            "read_noise": None,
            "saturation": None,
            "grow_pixels": 1,
            "reject_if_target_overlap": True,
            "target_radius_fwhm": 3.0,
            "psf_core_radius_fwhm": 1.0,
        },
    },
    "fringe": {
        "enabled": False,
        "eligible": False,
        "filters": ["i", "z"],
        "instruments": None,
        "map_path": None,
        "control_points_path": None,
        "control_points": None,
        "scale_method": "lstsq",
        "minimum_control_pairs": 10,
        "source_sigma": 3.0,
        "sigma_clip": 3.0,
        "maximum_iterations": 3,
        "scale_minimum": 0.0,
        "scale_maximum": None,
        "reject_invalid_scale": True,
        "validate_shape": True,
        "validate_binning": True,
        "check_filter": True,
        "save_model": True,
        "save_corrected": True,
    },
    "background": {
        "enabled": True,
        "mode": "broad_plus_local",
        "method": "Background2D",
        "box_size": [128, 128],
        "filter_size": [3, 3],
        "enforce_broad_scale": True,
        "minimum_mesh_fwhm": 10.0,
        "sigma_clip": 3.0,
        "maximum_iterations": 10,
        "exclude_percentile": 20.0,
        "estimator": "SExtractorBackground",
        "rms_estimator": "StdBackgroundRMS",
        "fallback_to_global": True,
        "source_mask_enabled": True,
        "source_mask_sigma": 3.0,
        "source_mask_min_pixels": 5,
        "source_mask_grow_fwhm": 3.0,
        "source_mask_kernel_fwhm": 1.0,
        "protect_target": True,
        "target_protection_fwhm": 5.0,
        "protect_host": False,
        "host_protection_fwhm": 10.0,
        "host_protection_region": None,
        "measure_residual_gradient": True,
        "gradient_max_samples": 50000,
        "measure_source_preservation": True,
        "source_preservation_max_sources": 50,
        "source_preservation_tolerance": 0.02,
        "save_background": True,
        "save_background_rms": True,
        "save_corrected": True,
        "save_source_mask": False,
    },
    "source_detection": {
        "enabled": True,
        "method": "segmentation",
        "threshold_sigma": 5.0,
        "fwhm_guess_pixels": 4.0,
        "kernel_fwhm_factor": 1.0,
        "minimum_separation_fwhm": 1.0,
        "exclude_border_fwhm": 3.0,
        "deblend": True,
        "minimum_pixels": 5,
        "connectivity": 8,
        "deblend_nlevels": 32,
        "deblend_contrast": 0.001,
        "minimum_snr": 5.0,
        "minimum_fwhm_pixels": 1.0,
        "maximum_fwhm_pixels": None,
        "maximum_ellipticity_for_seeing": 0.50,
        "maximum_sources": 1000,
        "reject_saturated": True,
        "reject_masked": True,
    },
    "astrometry": {
        "enabled": True,
        "catalog": "gaia",
        "verify_existing_wcs": True,
        "refine_wcs": True,
        "minimum_matches": 6,
        "maximum_match_separation_arcsec": 5.0,
        "require_unique_matches": True,
        "target_rms_arcsec": 0.5,
        "warning_rms_arcsec": 1.0,
        "fit_translation": True,
        "fit_rotation": True,
        "fit_scale": True,
        "fit_distortion": False,
        "sigma_clip": 4.0,
        "maximum_iterations": 3,
        "minimum_improvement_fraction": 0.05,
        "maximum_translation_pixels": 50.0,
        "maximum_rotation_degrees": 10.0,
        "maximum_scale_change_fraction": 0.10,
        "reject_unsafe_solution": True,
        "plate_solve_fallback": False,
        "plate_solver_command": "solve-field",
        "plate_solver_timeout_seconds": 300,
        "plate_solver_search_radius_degrees": 2.0,
        "plate_solver_scale_tolerance_fraction": 0.20,
        "reproject_science_image": False,
        "save_refined_header": True,
        "save_match_table": True,
    },
    "catalogs": {
        "cache_enabled": True,
        "cache_directory": "catalogs",
        "force_new_query": False,
        "save_catalog": True,
        "cache_format": "ecsv",
        "query_service": "vizier",
        "query_timeout_seconds": 120,
        "row_limit": -1,
        "use_object_name": True,
        "search_radius_arcmin": 10.0,
        "astrometry_catalog": "gaia",
        "catalog_ids": {
            "gaia": "I/355/gaiadr3",
            "ps1": "II/349/ps1",
            "sdss": "V/154/sdss16",
            "apass": "II/336/apass9",
            "skymapper": "II/379/smssdr4",
        },
        "photometry_catalog": "auto",
        "photometry_catalog_by_filter": {
            "u": "sdss",
            "g": "ps1",
            "r": "ps1",
            "i": "ps1",
            "z": "ps1",
            "y": "ps1",
            "B": "apass",
            "V": "apass",
            "R": "apass",
            "I": "apass",
        },
        "local_catalog_path": None,
        "user_column_map": {},
        "propagate_gaia_proper_motion": True,
        "comparison_stars": {
            "minimum_catalog_stars": 3,
            "maximum_catalog_stars": 300,
            "minimum_magnitude": 10.0,
            "maximum_magnitude": 22.0,
            "maximum_magnitude_error": 0.10,
            "allow_missing_magnitude_error": True,
            "minimum_snr": 10.0,
            "minimum_edge_distance_fwhm": 5.0,
            "minimum_saturation_distance_fwhm": 8.0,
            "minimum_mask_distance_fwhm": 2.0,
            "minimum_trail_distance_fwhm": 5.0,
            "minimum_neighbor_distance_fwhm": 3.0,
            "minimum_catalog_separation_arcsec": 2.0,
            "maximum_ellipticity": 0.35,
            "maximum_fwhm_deviation_fraction": 0.50,
            "maximum_proper_motion_mas_per_year": 200.0,
            "maximum_ruwe": 1.4,
            "require_point_source": True,
            "allow_unknown_morphology": True,
            "reject_known_variables": True,
            "require_catalog_quality": True,
            "allow_unknown_catalog_quality": True,
            "color_bands": ["g", "r"],
            "minimum_color": -0.5,
            "maximum_color": 2.5,
            "require_color": False,
            "prefer_target_color": False,
            "maximum_color_difference": None,
            "persistent_match_arcsec": 0.5,
            "check_batch_stability": True,
            "maximum_batch_rms_mag": 0.05,
            "minimum_epoch_fraction": 0.50,
            "psf_minimum_snr": 30.0,
            "psf_maximum_ellipticity": 0.25,
            "calibration_minimum_snr": 10.0,
            "ensemble_minimum_snr": 15.0,
            "qc_minimum_snr": 50.0,
            "maximum_calibration_stars": 100,
            "maximum_ensemble_stars": 50,
            "maximum_qc_anchors": 1,
            "spatial_grid": [3, 3],
            "excluded_detector_regions": [],
            "global_include": [],
            "global_exclude": [],
            "global_role_add": {},
            "global_role_remove": {},
            "image_overrides": {},
            "user_include_overrides_safety": False,
        },
    },
    "image_quality": {
        "enabled": True,
        "minimum_finite_fraction": 0.90,
        "maximum_masked_fraction_warn": 0.20,
        "maximum_masked_fraction_fail": 0.50,
        "minimum_sources_warn": 8,
        "minimum_sources_fail": 3,
        "fwhm_warn_arcsec": 4.0,
        "fwhm_fail_arcsec": 8.0,
        "ellipticity_warn": 0.30,
        "ellipticity_fail": 0.60,
        "fwhm_scatter_warn_fraction": 0.35,
        "fwhm_scatter_fail_fraction": 0.70,
        "ellipticity_scatter_warn": 0.15,
        "ellipticity_scatter_fail": 0.30,
        "elongated_source_ellipticity": 0.35,
        "elongated_fraction_warn": 0.35,
        "elongated_fraction_fail": 0.65,
        "orientation_concentration_minimum": 0.60,
        "maximum_saturated_sources_warn": None,
        "maximum_saturated_sources_fail": None,
        "maximum_background_warn": None,
        "maximum_background_fail": None,
        "maximum_background_rms_warn": None,
        "maximum_background_rms_fail": None,
        "target_background_inner_fwhm": 5.0,
        "target_background_outer_fwhm": 8.0,
        "upstream_fwhm_difference_warn_fraction": 0.50,
        "upstream_fwhm_difference_fail_fraction": 1.00,
        "upstream_ellipticity_difference_warn": 0.15,
        "upstream_ellipticity_difference_fail": 0.30,
        "upstream_background_difference_warn_fraction": 0.50,
        "upstream_background_difference_fail_fraction": 1.00,
        "upstream_background_rms_difference_warn_fraction": 0.50,
        "upstream_background_rms_difference_fail_fraction": 1.00,
        "batch_minimum_images": 3,
        "batch_fwhm_ratio_warn": 1.50,
        "batch_fwhm_ratio_fail": 2.50,
        "batch_ellipticity_offset_warn": 0.15,
        "batch_ellipticity_offset_fail": 0.30,
        "batch_background_ratio_warn": 2.00,
        "batch_background_ratio_fail": 5.00,
        "batch_background_rms_ratio_warn": 2.00,
        "batch_background_rms_ratio_fail": 5.00,
        "wcs_rms_warn_arcsec": 1.0,
        "wcs_rms_fail_arcsec": 3.0,
        "zeropoint_offset_warn_mag": 0.50,
        "zeropoint_offset_fail_mag": 1.50,
        "zeropoint_scatter_warn_mag": 0.10,
        "zeropoint_scatter_fail_mag": 0.30,
        "minimum_catalog_recovery_warn": 0.40,
        "minimum_catalog_recovery_fail": 0.15,
        "maximum_trail_fraction_warn": 0.05,
        "maximum_trail_fraction_fail": 0.20,
        "minimum_useful_depth_mag": None,
        "expected_target_magnitude": None,
        "reject_target_artifact_overlap": True,
        "allow_user_approval": True,
        "usability": {
            "enabled": True,
            "minimum_calibration_stars_warn": 5,
            "minimum_calibration_stars_fail": 2,
            "minimum_catalog_recovery_warn": 0.40,
            "minimum_catalog_recovery_fail": 0.15,
            "require_qc_anchor": True,
            "missing_qc_anchor_status": "WARN",
            "zeropoint_sigma_clip": 3.0,
            "zeropoint_maximum_iterations": 5,
            "zeropoint_scatter_warn_mag": 0.10,
            "zeropoint_scatter_fail_mag": 0.30,
            "transparency_minimum_images": 2,
            "transparency_attenuation_warn_mag": 0.50,
            "transparency_attenuation_fail_mag": 1.50,
            "cloud_minimum_stars": 8,
            "cloud_spatial_grid": [3, 3],
            "cloud_spatial_amplitude_warn_mag": 0.15,
            "cloud_spatial_amplitude_fail_mag": 0.35,
            "limiting_sigma_levels": [3.0, 5.0],
            "limiting_aperture_radius_fwhm": 1.0,
            "noise_correlation_factor": 1.0,
            "minimum_limit_mag": None,
            "expected_target_magnitude": None,
            "target_depth_margin_warn_mag": 0.50,
            "target_depth_margin_fail_mag": 0.0,
            "local_depth_loss_warn_mag": 0.50,
            "local_depth_loss_fail_mag": 1.50,
            "target_artifact_radius_fwhm": 1.0,
            "target_mask_status": "FAIL",
            "target_trail_status": "FAIL",
            "target_cosmic_ray_status": "WARN",
            "require_manual_review": False,
            "manual_decisions": {},
            "save_star_residuals": True,
            "save_decision_table": True,
        },
    },
    "target_position": {
        "ra": None,
        "dec": None,
        "coordinate_unit": ["hourangle", "deg"],
        "metadata_precedence": [
            "user",
            "header_target",
            "wcs_center",
            "telescope_pointing",
        ],
        "user_position_required": False,
        "position_precedence": [
            "user",
            "detection_stack",
            "catalog_header",
            "object_header",
        ],
        "build_detection_stack": True,
        "build_per_filter_stacks": True,
        "build_multifilter_stack": True,
        "exclude_failed_images": True,
        "exclude_warned_images": False,
        "fixed_position_photometry": True,
        "diagnostic_recenter": True,
        "maximum_diagnostic_offset_arcsec": 1.0,
        "save_stack": True,
        "relative_alignment_enabled": True,
        "relative_alignment_minimum_common_stars": 6,
        "relative_alignment_sigma_clip": 4.0,
        "relative_alignment_maximum_iterations": 3,
        "relative_alignment_target_rms_arcsec": 0.20,
        "relative_alignment_warn_rms_arcsec": 0.50,
        "relative_alignment_fail_rms_arcsec": 1.50,
        "relative_alignment_maximum_translation_pixels": 20.0,
        "relative_alignment_maximum_rotation_degrees": 2.0,
        "relative_alignment_maximum_scale_change_fraction": 0.02,
        "coordinate_allowed_statuses": ["PASS", "WARN"],
        "stack_allowed_statuses": ["PASS", "WARN"],
        "require_usability_approval": True,
        "stack_minimum_images_per_filter": 1,
        "stack_maximum_images_per_filter": None,
        "stack_combine": "weighted_mean",
        "stack_sigma_clip": 4.0,
        "stack_reprojection_order": 1,
        "stack_reprojection_tile_rows": 256,
        "stack_normalization": "zeropoint",
        "stack_use_inverse_variance_weights": True,
        "multifilter_normalization": "background_rms",
        "resample_science_images": False,
        "user_position_mode": "prior",
        "prior_uncertainty_arcsec": 1.0,
        "centroid_search_radius_arcsec": 3.0,
        "centroid_aperture_radius_fwhm": 1.5,
        "centroid_background_inner_fwhm": 3.0,
        "centroid_background_outer_fwhm": 6.0,
        "centroid_minimum_snr": 3.0,
        "centroid_maximum_offset_arcsec": 2.0,
        "centroid_maximum_width_fwhm": 1.8,
        "centroid_maximum_ellipticity": 0.50,
        "maximum_individual_centroids": 5,
        "maximum_filter_shift_arcsec": 0.50,
        "maximum_coordinate_uncertainty_arcsec": 1.0,
        "minimum_coordinate_uncertainty_arcsec": 0.02,
        "target_coordinate_version": 1,
        "allow_difference_stack_candidates": True,
        "save_alignment_headers": True,
        "save_alignment_table": True,
        "save_target_candidates": True,
        "save_target_position": True,
    },
    "psf": {
        "enabled": True,
        "model": "empirical",
        "fallback_model": "moffat",
        "box_size_pixels": 25,
        "oversampling": 2,
        "minimum_stars": 5,
        "minimum_fallback_stars": 1,
        "maximum_stars": 20,
        "minimum_star_snr": 30.0,
        "minimum_edge_distance_fwhm": 5.0,
        "reject_saturated": True,
        "reject_masked": True,
        "reject_blended": True,
        "maximum_masked_fraction": 0.05,
        "local_background_border_pixels": 3,
        "normalization_radius_fwhm": 2.5,
        "sigma_clip": 3.0,
        "maximum_iterations": 3,
        "spatial_order": 0,
        "minimum_spatial_stars": 20,
        "minimum_spatial_cells": 6,
        "spatial_grid": [3, 3],
        "maximum_residual_fraction": 0.20,
        "residual_warn_fraction": 0.15,
        "residual_fail_fraction": 0.30,
        "minimum_correlation": 0.90,
        "analytic_beta": 2.5,
        "fit_analytic_beta": True,
        "minimum_fwhm_pixels": 0.8,
        "maximum_fwhm_pixels": 30.0,
        "require_manual_review": False,
        "approved_statuses": ["PASS", "WARN"],
        "manual_decisions": {},
        "model_version": 1,
        "fix_target_centroid": True,
        "save_model": True,
        "save_cutouts": True,
        "save_residuals": True,
        "save_star_table": True,
        "save_review": True,
    },
    "apertures": {
        "enabled": True,
        "perform_small_aperture": True,
        "perform_large_aperture": True,
        "perform_psf": True,
        "perform_optimal": False,
        "require_approved_psf": True,
        "small_radius_fwhm": 1.0,
        "large_radius_fwhm": 2.5,
        "sky_inner_radius_fwhm": 4.0,
        "sky_outer_radius_fwhm": 7.0,
        "minimum_sky_pixels": 50,
        "subpixel_method": "exact",
        "subpixels": 5,
        "minimum_unmasked_fraction": 0.80,
        "local_background_estimator": "median",
        "local_background_sigma_clip": 3.0,
        "local_background_maximum_iterations": 5,
        "add_poisson_noise_when_needed": True,
        "minimum_uncertainty": 1.0e-6,
        "diagnostic_cutout_radius_fwhm": 5.0,
        "apply_aperture_correction": True,
        "fixed_target_position": True,
        "local_background": True,
        "save_measurement_table": True,
        "save_target_cutouts": True,
    },
    "calibration": {
        "enabled": True,
        "catalog": "auto",
        "allow_catalog_fallback": True,
        "require_routed_catalog": False,
        "filter_column_map": {},
        "magnitude_system_by_catalog": {
            "ps1": "AB",
            "sdss": "AB",
            "skymapper": "AB",
            "gaia": "Vega",
            "apass": "mixed",
            "user": "user",
        },
        "minimum_stars": 3,
        "maximum_stars": 100,
        "minimum_star_snr": 10.0,
        "maximum_catalog_error_mag": 0.10,
        "allow_missing_catalog_error": True,
        "excluded_measurement_flags": [
            "APERTURE_INCOMPLETE",
            "INSUFFICIENT_UNMASKED_PIXELS",
            "BAD_LOCAL_BACKGROUND",
            "COSMIC_RAY_OVERLAP",
            "TRAIL_OVERLAP",
            "MASKED_PIXELS",
        ],
        "sigma_clip": 3.0,
        "maximum_iterations": 5,
        "maximum_star_rms_mag": 0.10,
        "minimum_epochs_for_stability": 2,
        "maximum_catalog_separation_arcsec": 2.0,
        "calculate_psf_zeropoint": True,
        "calculate_aperture_zeropoint": True,
        "reference_aperture_method": "large_aperture",
        "minimum_aperture_correction_stars": 3,
        "aperture_correction_sigma_clip": 3.0,
        "calculate_color_term": False,
        "apply_color_term": False,
        "apply_atmospheric_extinction": False,
        "apply_galactic_extinction": False,
        "zeropoint_scatter_warn_mag": 0.10,
        "zeropoint_scatter_fail_mag": 0.30,
        "trend_slope_warn_mag": 0.05,
        "detection_sigma": 3.0,
        "retain_instrumental_measurements": True,
        "save_calibrated_table": True,
        "save_zeropoints": True,
        "save_calibration_stars": True,
        "save_aperture_corrections": True,
        "save_summary": True,
    },
    "subtraction": {
        "enabled": False,
        "method": "hotpants",
        "fallback_method": None,
        "template_path": None,
        "template_source": "auto",
        "template_survey_priority": ["ps1", "legacy", "decam"],
        "survey_names": {
            "ps1": "PanSTARRS DR1",
            "legacy": "DESI Legacy Imaging Surveys",
            "decam": "DECaLS DR5",
        },
        "survey_filter_map": {},
        "download_pixel_scale_arcsec": None,
        "download_timeout_s": 120,
        "maximum_mosaic_pixels": 100000000,
        "cache_directory": "redphot_cache/templates",
        "use_cached_templates": True,
        "save_downloaded_templates": True,
        "template_margin_arcmin": 2.0,
        "require_pretransient_template": True,
        "transient_epoch_mjd": None,
        "require_filter_match": True,
        "allow_approximate_filter_match": False,
        "approximate_filter_matches": {},
        "minimum_coverage_fraction": 0.99,
        "minimum_template_depth_margin_mag": 0.5,
        "maximum_template_fwhm_ratio": 2.0,
        "maximum_template_saturated_fraction": 0.01,
        "keep_science_grid": True,
        "resample_template_only": True,
        "resampling_order": 3,
        "resampling_tile_rows": 256,
        "background_match": True,
        "photometric_scale": True,
        "scale_sigma_clip": 3.0,
        "scale_maximum_iterations": 5,
        "minimum_scale_pixels": 1000,
        "automatic_parameters": True,
        "hotpants_executable": "hotpants",
        "execution_timeout_s": 300,
        "hotpants": {
            "kernel_order": "auto",
            "background_order": "auto",
            "stamp_count": "auto",
            "stamp_grid": "auto",
            "kernel_radius": "auto",
            "lower_threshold": "auto",
            "upper_threshold": "auto",
            "extra_arguments": [],
        },
        "pyzogy": {
            "enabled": False,
            "require_variance": True,
            "require_psf": True,
        },
        "maximum_alignment_rms_pixels": 0.5,
        "maximum_residual_fraction": 0.10,
        "maximum_dipole_fraction": 0.20,
        "maximum_flux_bias_fraction": 0.10,
        "maximum_noise_ratio": 2.0,
        "minimum_quality_stars": 3,
        "blank_aperture_count": 50,
        "blank_aperture_radius_fwhm": 1.0,
        "blank_aperture_seed": 12345,
        "save_aligned_template": True,
        "save_difference": True,
        "save_logs": True,
        "save_parameters": True,
        "save_quality_table": True,
        "photometry": {
            "enabled": True,
            "require_accepted_subtraction": True,
            "validate_with_empty_apertures": True,
            "minimum_empty_apertures": 20,
            "uncertainty_warn_ratio": 1.25,
            "uncertainty_fail_ratio": 2.0,
            "inflate_underestimated_uncertainties": True,
            "dipole_sigma": 3.0,
            "dipole_ratio_threshold": 0.25,
            "dipole_minimum_separation_pixels": 0.5,
            "inverted_residual_sigma": 3.0,
            "detection_sigma": 3.0,
            "preferred_difference_methods": [
                "psf",
                "small_aperture",
                "large_aperture",
            ],
            "preferred_science_methods": [
                "psf",
                "small_aperture",
                "large_aperture",
            ],
            "prefer_difference_when_valid": True,
            "save_measurements": True,
            "save_comparison": True,
            "save_limits": True,
            "save_summary": True,
        },
    },
    "upper_limits": {
        "enabled": True,
        "sigma_levels": [3.0, 5.0],
        "analytic": True,
        "empty_apertures": True,
        "number_empty_apertures": 100,
        "minimum_empty_apertures": 20,
        "maximum_empty_aperture_attempts_factor": 30,
        "empty_aperture_methods": ["small_aperture", "large_aperture", "psf"],
        "empty_aperture_radius_fwhm": 1.0,
        "empty_aperture_local_radius_arcsec": 60.0,
        "empty_aperture_source_exclusion_fwhm": 3.0,
        "empty_aperture_random_seed": 12345,
        "exclude_sources": True,
        "exclude_masked_regions": True,
        "injection_recovery": False,
        "number_injected_sources": 100,
        "minimum_recovery_fraction": 0.50,
        "save_limit_table": True,
        "calculate_on_science": True,
        "calculate_on_difference": True,
    },
    "batch_consistency": {
        "enabled": True,
        "comparison_methods": ["small_aperture", "large_aperture", "psf"],
        "minimum_comparison_epochs": 3,
        "comparison_star_rms_warn_mag": 0.05,
        "comparison_star_rms_fail_mag": 0.15,
        "comparison_star_reduced_chi2_warn": 3.0,
        "comparison_star_reduced_chi2_fail": 10.0,
        "minimum_stable_comparison_stars": 3,
        "metric_outlier_warn_sigma": 3.5,
        "metric_outlier_fail_sigma": 6.0,
        "problem_group_warn_fraction": 0.25,
        "problem_group_fail_fraction": 0.50,
        "minimum_group_images": 2,
        "ensemble_correction": {
            "enabled": False,
            "components": ["telescope", "epoch"],
            "minimum_stars": 3,
            "sigma_clip": 3.0,
            "maximum_iterations": 5,
            "maximum_absolute_correction_mag": 0.50,
            "apply_to_target": True,
        },
        "method_disagreement_warn_sigma": 3.0,
        "method_disagreement_fail_sigma": 5.0,
        "method_disagreement_floor_mag": 0.05,
        "temporal_outlier_sigma": 5.0,
        "temporal_maximum_gap_days": 30.0,
        "preferred_order": [
            "difference:psf",
            "difference:small_aperture",
            "difference:large_aperture",
            "science:psf",
            "science:small_aperture",
            "science:large_aperture",
        ],
        "accepted_image_statuses": ["PASS", "WARN"],
        "retain_rejected_measurements": True,
        "save_comparison_light_curves": True,
        "save_stability_table": True,
        "save_epoch_metrics": True,
        "save_group_summary": True,
        "save_method_comparison": True,
        "save_preferred_light_curve": True,
        "save_all_flagged_measurements": True,
        "save_summary": True,
    },
    "diagnostics": {
        "enabled": True,
        "show_plots": False,
        "save_stage_plots": True,
        "make_image_pdf": True,
        "make_batch_pdf": True,
        "include_failed_images": True,
        "plot_original_image": True,
        "plot_masks": True,
        "plot_background": True,
        "plot_astrometry": True,
        "plot_comparison_stars": True,
        "plot_psf": True,
        "plot_psf_3d": True,
        "plot_calibration": True,
        "plot_target": True,
        "plot_subtraction": True,
        "plot_upper_limits": True,
        "image_format": "png",
        "dpi": 150,
    },
    "output": {
        "directory": "redphot_output",
        "overwrite": False,
        "save_intermediate_fits": True,
        "save_masks": True,
        "save_background": True,
        "save_cleaned_image": False,
        "save_fringe_corrected_image": True,
        "save_psf": True,
        "save_templates": True,
        "save_difference": True,
        "table_format": "ascii.ecsv",
        "image_format": "fits",
        "write_resolved_config": True,
        "write_log": True,
        "log_level": "INFO",
    },
}


FILTER_DEFAULTS = {
    "i": {
        "fringe": {
            "eligible": True,
        },
    },
    "z": {
        "fringe": {
            "eligible": True,
        },
    },
}


INSTRUMENT_DEFAULTS = {
    "lco": {
        "instrument": {
            "name": "LCO",
            "profile": "lco",
        },
        "input": {
            "preferred_extnames": ["SCI", "IMAGE"],
            "hdu_search_order": [0, 1, 2],
            "allow_compressed": True,
            "verify_checksum": True,
        },
        "metadata": {
            "keywords": {
                "object": ["OBJECT", "GROUPID"],
                "telescope": ["TELESCOP", "TELID"],
                "instrument": ["INSTRUME"],
                "detector": ["DETECTOR", "DETECTID"],
                "site": ["SITE", "SITEID"],
                "exposure_time": ["EXPTIME"],
                "date_obs": ["DATE-OBS"],
                "date_end": ["DATE-END", "UTSTOP"],
                "mjd": ["MJD-OBS", "MJD", "OBSMJD", "JD"],
                "filter": ["FILTER", "FILTER1"],
                "gain": ["GAIN"],
                "read_noise": ["RDNOISE"],
                "saturation": ["MAXLIN", "SATURATE"],
                "nonlinearity": ["MAXLIN", "LINLIMIT", "NONLIN"],
                "airmass": ["AIRMASS", "AIR"],
                "pointing_ra": ["RA"],
                "pointing_dec": ["DEC"],
                "target_ra": ["CAT-RA", "OFST-RA"],
                "target_dec": ["CAT-DEC", "OFST-DEC"],
                "pixel_scale": ["PIXSCALE"],
                "binning": ["CCDSUM"],
                "reduction_level": ["RLEVEL"],
                "pipeline_version": ["PIPEVER"],
            },
            "fallback_values": {
                "telescope": "LCO",
                "instrument": None,
                "detector": None,
                "site": None,
                "gain": None,
                "read_noise": None,
                "saturation": None,
                "pixel_scale": None,
            },
            "time_reference": "start",
            "convert_time_to_mid_exposure": True,
            "require_reduced_input": True,
        },
        "crop": {
            "enabled": True,
            "center_on": "target",
            "size_arcmin": 15.0,
            "use_datasec": True,
            "use_trimsec": True,
        },
        "background": {
            "box_size": [128, 128],
            "filter_size": [3, 3],
        },
        "source_detection": {
            "fwhm_guess_pixels": 5.0,
        },
    },
    "keplercam": {
        "instrument": {
            "name": "KeplerCam",
            "profile": "keplercam",
        },
        "input": {
            "preferred_extnames": ["SCI", "IM1", "IM2", "IM3", "IM4"],
            "hdu_search_order": [0, 1, 2],
            "allow_compressed": True,
        },
        "metadata": {
            "keywords": {
                "object": ["OBJECT"],
                "telescope": ["TELESCOP", "SITENAME"],
                "instrument": ["INSTRUME", "DETECTOR"],
                "detector": ["DETECTOR"],
                "site": ["SITENAME"],
                "exposure_time": ["EXPTIME"],
                "date_obs": ["DATE-OBS"],
                "date_end": ["DATE-END", "UTEND"],
                "mjd": ["MJD-OBS", "MJD", "OBSMJD", "JD"],
                "filter": ["FILTER"],
                "gain": ["GAIN", "EGAIN", "GAINDL"],
                "read_noise": ["RDNOISE", "READNOI", "ENOISE"],
                "saturation": ["MAXLIN", "SATURATE"],
                "nonlinearity": ["MAXLIN", "LINLIMIT", "NONLIN"],
                "airmass": ["AIR", "AIRMASS", "SECZ"],
                "pointing_ra": ["RRA", "RA"],
                "pointing_dec": ["RDEC", "DEC"],
                "target_ra": ["RA"],
                "target_dec": ["DEC"],
                "pixel_scale": ["SECPIX1", "SECPIX2"],
                "binning": ["CCDSUM"],
                "reduction_level": ["RLEVEL"],
                "pipeline_version": ["PIPEVER"],
            },
            "fallback_values": {
                "telescope": "FLWO 1.2m",
                "instrument": "KeplerCam",
                "detector": "KeplerCam",
                "site": "FLWO",
                "gain": 4.45,
                "read_noise": 7.18,
                "saturation": 50000.0,
                "nonlinearity": 50000.0,
                "pixel_scale": 0.672,
            },
            "time_reference": "start",
            "convert_time_to_mid_exposure": True,
            "exposure_time_tolerance_s": 1.0,
            "require_reduced_input": True,
        },
        "crop": {
            "enabled": True,
            "center_on": "target",
            "size_arcmin": None,
            "use_datasec": True,
            "use_trimsec": True,
        },
        "background": {
            "box_size": [64, 64],
            "filter_size": [3, 3],
        },
        "source_detection": {
            "fwhm_guess_pixels": 4.0,
        },
        "fringe": {
            "filters": ["i", "z"],
        },
    },
}


REQUIRED_SECTIONS = [
    "instrument",
    "input",
    "metadata",
    "crop",
    "masks",
    "fringe",
    "background",
    "source_detection",
    "astrometry",
    "catalogs",
    "image_quality",
    "target_position",
    "psf",
    "apertures",
    "calibration",
    "subtraction",
    "upper_limits",
    "batch_consistency",
    "diagnostics",
    "output",
]


def normalize_filter_name(filter_name):
    """
    Convert a raw filter name to a standard redphot filter name.

    Unrecognized filter names are returned unchanged so that the caller can
    decide whether to accept, override, or reject them.
    """
    if filter_name is None:
        return None

    value = str(filter_name).strip()

    if value in FILTER_ALIASES:
        return FILTER_ALIASES[value]

    normalized = value.replace("_", "-").replace(" ", "-")

    if normalized in FILTER_ALIASES:
        return FILTER_ALIASES[normalized]

    if len(normalized) > 1:
        lower_value = normalized.lower()
        if lower_value in FILTER_ALIASES:
            return FILTER_ALIASES[lower_value]

    return value


def normalize_instrument_name(instrument_name):
    """
    Convert an instrument, telescope, or site name to a profile name.

    Unknown names are returned in normalized lowercase form. This allows a
    caller to use general defaults even when no instrument profile exists.
    """
    if instrument_name is None:
        return None

    value = str(instrument_name).strip().lower()
    normalized = value.replace("_", "-").replace(" ", "-")

    if normalized in INSTRUMENT_ALIASES:
        return INSTRUMENT_ALIASES[normalized]

    if "lcogt" in normalized or "las-cumbres" in normalized:
        return "lco"

    if normalized.startswith("sq") or "qhy600" in normalized:
        return "lco"

    if "kepcam" in normalized or "keplercam" in normalized:
        return "keplercam"

    if "flwo" in normalized:
        return "keplercam"

    return normalized


def merge_settings(base_settings, override_settings):
    """
    Recursively merge two settings dictionaries without modifying either one.

    Nested dictionaries are merged recursively. Lists, tuples, scalars, and
    None values replace the corresponding value from the base dictionary.
    """
    if not isinstance(base_settings, Mapping):
        raise TypeError("base_settings must be a mapping")

    if override_settings is None:
        return deepcopy(dict(base_settings))

    if not isinstance(override_settings, Mapping):
        raise TypeError("override_settings must be a mapping or None")

    merged = deepcopy(dict(base_settings))

    for key, value in override_settings.items():
        if (
            key in merged
            and isinstance(merged[key], Mapping)
            and isinstance(value, Mapping)
        ):
            merged[key] = merge_settings(merged[key], value)
        else:
            merged[key] = deepcopy(value)

    return merged


def get_default_settings():
    """
    Return an independent copy of the general redphot settings.
    """
    return deepcopy(DEFAULT_SETTINGS)


def get_instrument_settings(instrument_name):
    """
    Return an independent instrument-settings dictionary.

    An empty dictionary is returned when no matching instrument profile exists.
    """
    profile = normalize_instrument_name(instrument_name)

    if profile not in INSTRUMENT_DEFAULTS:
        return {}

    return deepcopy(INSTRUMENT_DEFAULTS[profile])


def get_filter_settings(filter_name, filter_settings=None):
    """
    Return the built-in and user-supplied settings for one filter.

    User filter settings must be a mapping whose keys are raw or normalized
    filter names and whose values are nested override dictionaries.
    """
    normalized_filter = normalize_filter_name(filter_name)
    resolved = {}

    if normalized_filter in FILTER_DEFAULTS:
        resolved = merge_settings(
            resolved,
            FILTER_DEFAULTS[normalized_filter],
        )

    if filter_settings is None:
        return resolved

    if not isinstance(filter_settings, Mapping):
        raise TypeError("filter_settings must be a mapping or None")

    for raw_filter, overrides in filter_settings.items():
        if normalize_filter_name(raw_filter) == normalized_filter:
            resolved = merge_settings(resolved, overrides)

    return resolved


def get_image_settings(image_name, image_overrides=None):
    """
    Return overrides assigned to one image.

    Overrides may be keyed by the complete filename or by basename. A basename
    override is applied first, followed by a complete-path override when both
    are present.
    """
    if image_name is None or image_overrides is None:
        return {}

    if not isinstance(image_overrides, Mapping):
        raise TypeError("image_overrides must be a mapping or None")

    image_string = str(image_name)
    image_basename = Path(image_string).name
    normalized_overrides = {
        str(key): value for key, value in image_overrides.items()
    }

    resolved = {}

    if image_basename in normalized_overrides:
        resolved = merge_settings(
            resolved,
            normalized_overrides[image_basename],
        )

    if (
        image_string in normalized_overrides
        and image_string != image_basename
    ):
        resolved = merge_settings(
            resolved,
            normalized_overrides[image_string],
        )

    return resolved


def validate_settings(settings):
    """
    Perform lightweight validation of a resolved settings dictionary.

    This function checks only configuration structure and basic value choices.
    Scientific validation remains the responsibility of the processing stage
    that uses each setting.
    """
    if not isinstance(settings, Mapping):
        raise TypeError("settings must be a mapping")

    missing_sections = [
        section for section in REQUIRED_SECTIONS if section not in settings
    ]

    if missing_sections:
        raise KeyError(
            "Missing required settings sections: "
            + ", ".join(missing_sections)
        )

    crop_size = settings["crop"]["size_arcmin"]
    if crop_size is not None and crop_size <= 0:
        raise ValueError("crop.size_arcmin must be positive or None")

    crop_center = settings["crop"]["center_on"]
    if crop_center not in {"target", "field"}:
        raise ValueError("crop.center_on must be 'target' or 'field'")

    crop_mode = settings["crop"].get("crop_mode", "trim")
    if crop_mode not in {"trim", "partial"}:
        raise ValueError("crop.crop_mode must be 'trim' or 'partial'")

    cosmic_mode = settings["masks"]["cosmic_rays"]["mode"]
    if cosmic_mode not in {"off", "detect_only", "mask", "clean"}:
        raise ValueError(
            "masks.cosmic_rays.mode must be off, detect_only, mask, or clean"
        )

    fringe_method = settings["fringe"].get("scale_method", "lstsq")
    if fringe_method not in {"lstsq", "control_pairs"}:
        raise ValueError(
            "fringe.scale_method must be 'lstsq' or 'control_pairs'"
        )

    background_mode = settings["background"]["mode"]
    if background_mode not in {
        "off",
        "measure_only",
        "subtract_broad",
        "local_only",
        "broad_plus_local",
    }:
        raise ValueError(
            "background.mode must be off, measure_only, subtract_broad, "
            "local_only, or broad_plus_local"
        )

    background_box = settings["background"].get("box_size", [128, 128])
    if isinstance(background_box, (int, float)):
        background_box = [background_box, background_box]
    if len(background_box) != 2 or any(int(value) <= 0 for value in background_box):
        raise ValueError("background.box_size must contain two positive integers")

    background_filter = settings["background"].get("filter_size", [3, 3])
    if isinstance(background_filter, (int, float)):
        background_filter = [background_filter, background_filter]
    if (
        len(background_filter) != 2
        or any(int(value) <= 0 for value in background_filter)
        or any(int(value) % 2 == 0 for value in background_filter)
    ):
        raise ValueError(
            "background.filter_size must contain two positive odd integers"
        )

    exclude_percentile = settings["background"].get("exclude_percentile", 20.0)
    if not 0 <= float(exclude_percentile) <= 100:
        raise ValueError("background.exclude_percentile must be between 0 and 100")

    source_settings = settings["source_detection"]
    if float(source_settings.get("threshold_sigma", 5.0)) <= 0:
        raise ValueError("source_detection.threshold_sigma must be positive")
    if int(source_settings.get("minimum_pixels", 5)) <= 0:
        raise ValueError("source_detection.minimum_pixels must be positive")
    if int(source_settings.get("connectivity", 8)) not in {4, 8}:
        raise ValueError("source_detection.connectivity must be 4 or 8")
    contrast = float(source_settings.get("deblend_contrast", 0.001))
    if not 0 <= contrast <= 1:
        raise ValueError(
            "source_detection.deblend_contrast must be between 0 and 1"
        )

    astrometry_settings = settings["astrometry"]
    if int(astrometry_settings.get("minimum_matches", 6)) < 3:
        raise ValueError("astrometry.minimum_matches must be at least 3")
    if float(
        astrometry_settings.get("maximum_match_separation_arcsec", 5.0)
    ) <= 0:
        raise ValueError(
            "astrometry.maximum_match_separation_arcsec must be positive"
        )
    if astrometry_settings.get("fit_distortion", False):
        raise ValueError(
            "astrometry.fit_distortion is not supported; WCS refinement intentionally "
            "fits only translation, rotation, and scale"
        )
    if float(astrometry_settings.get("plate_solver_timeout_seconds", 300)) <= 0:
        raise ValueError("astrometry.plate_solver_timeout_seconds must be positive")
    plate_scale_tolerance = float(
        astrometry_settings.get("plate_solver_scale_tolerance_fraction", 0.20)
    )
    if not 0 <= plate_scale_tolerance < 1:
        raise ValueError(
            "astrometry.plate_solver_scale_tolerance_fraction must be in [0, 1)"
        )

    catalog_settings = settings["catalogs"]
    if float(catalog_settings.get("search_radius_arcmin", 10.0)) <= 0:
        raise ValueError("catalogs.search_radius_arcmin must be positive")
    if catalog_settings.get("cache_format", "ecsv") not in {"ecsv", "fits"}:
        raise ValueError("catalogs.cache_format must be 'ecsv' or 'fits'")
    star_settings = catalog_settings.get("comparison_stars", {})
    minimum_magnitude = float(star_settings.get("minimum_magnitude", 10.0))
    maximum_magnitude = float(star_settings.get("maximum_magnitude", 22.0))
    if minimum_magnitude >= maximum_magnitude:
        raise ValueError(
            "catalogs.comparison_stars.minimum_magnitude must be below "
            "maximum_magnitude"
        )
    spatial_grid = star_settings.get("spatial_grid", [3, 3])
    if len(spatial_grid) != 2 or any(int(value) <= 0 for value in spatial_grid):
        raise ValueError(
            "catalogs.comparison_stars.spatial_grid must contain two positive integers"
        )
    color_bands = star_settings.get("color_bands", ["g", "r"])
    if len(color_bands) != 2:
        raise ValueError(
            "catalogs.comparison_stars.color_bands must contain two filter names"
        )

    quality_settings = settings["image_quality"]
    threshold_pairs = (
        ("maximum_masked_fraction_warn", "maximum_masked_fraction_fail"),
        ("fwhm_warn_arcsec", "fwhm_fail_arcsec"),
        ("ellipticity_warn", "ellipticity_fail"),
        ("fwhm_scatter_warn_fraction", "fwhm_scatter_fail_fraction"),
        ("ellipticity_scatter_warn", "ellipticity_scatter_fail"),
        ("elongated_fraction_warn", "elongated_fraction_fail"),
        ("batch_fwhm_ratio_warn", "batch_fwhm_ratio_fail"),
        ("batch_ellipticity_offset_warn", "batch_ellipticity_offset_fail"),
        ("batch_background_ratio_warn", "batch_background_ratio_fail"),
        ("batch_background_rms_ratio_warn", "batch_background_rms_ratio_fail"),
    )
    for warn_name, fail_name in threshold_pairs:
        warn_value = quality_settings.get(warn_name)
        fail_value = quality_settings.get(fail_name)
        if warn_value is not None and fail_value is not None:
            if float(warn_value) > float(fail_value):
                raise ValueError(
                    "image_quality.{} cannot exceed image_quality.{}".format(
                        warn_name, fail_name
                    )
                )

    inner_target = float(
        quality_settings.get("target_background_inner_fwhm", 5.0)
    )
    outer_target = float(
        quality_settings.get("target_background_outer_fwhm", 8.0)
    )
    if inner_target < 0 or outer_target <= inner_target:
        raise ValueError(
            "image_quality target-background radii must be non-negative and "
            "outer must exceed inner"
        )

    usability_settings = quality_settings.get("usability", {})
    recovery_warn = float(
        usability_settings.get("minimum_catalog_recovery_warn", 0.40)
    )
    recovery_fail = float(
        usability_settings.get("minimum_catalog_recovery_fail", 0.15)
    )
    if not 0 <= recovery_fail <= recovery_warn <= 1:
        raise ValueError(
            "image_quality.usability catalog-recovery limits must satisfy "
            "0 <= fail <= warn <= 1"
        )
    calibration_warn = int(
        usability_settings.get("minimum_calibration_stars_warn", 5)
    )
    calibration_fail = int(
        usability_settings.get("minimum_calibration_stars_fail", 2)
    )
    if calibration_fail < 0 or calibration_warn < calibration_fail:
        raise ValueError(
            "image_quality.usability calibration-star limits must satisfy "
            "0 <= fail <= warn"
        )
    for name in (
        "zeropoint_scatter",
        "transparency_attenuation",
        "cloud_spatial_amplitude",
        "local_depth_loss",
    ):
        warn_value = usability_settings.get("{}_warn_mag".format(name))
        fail_value = usability_settings.get("{}_fail_mag".format(name))
        if (
            warn_value is not None
            and fail_value is not None
            and float(warn_value) > float(fail_value)
        ):
            raise ValueError(
                "image_quality.usability.{}_warn_mag cannot exceed the "
                "corresponding fail value".format(name)
            )
    target_margin_warn = usability_settings.get(
        "target_depth_margin_warn_mag"
    )
    target_margin_fail = usability_settings.get(
        "target_depth_margin_fail_mag"
    )
    if (
        target_margin_warn is not None
        and target_margin_fail is not None
        and float(target_margin_fail) > float(target_margin_warn)
    ):
        raise ValueError(
            "image_quality.usability target-depth fail margin cannot exceed "
            "the warn margin"
        )
    sigma_levels = usability_settings.get("limiting_sigma_levels", [3.0, 5.0])
    if not sigma_levels or any(float(value) <= 0 for value in sigma_levels):
        raise ValueError(
            "image_quality.usability.limiting_sigma_levels must be positive"
        )
    cloud_grid = usability_settings.get("cloud_spatial_grid", [3, 3])
    if len(cloud_grid) != 2 or any(int(value) <= 0 for value in cloud_grid):
        raise ValueError(
            "image_quality.usability.cloud_spatial_grid must contain two "
            "positive integers"
        )
    if float(usability_settings.get("noise_correlation_factor", 1.0)) <= 0:
        raise ValueError(
            "image_quality.usability.noise_correlation_factor must be positive"
        )
    for name in (
        "missing_qc_anchor_status",
        "target_mask_status",
        "target_trail_status",
        "target_cosmic_ray_status",
    ):
        if usability_settings.get(name, "WARN") not in {"PASS", "WARN", "FAIL"}:
            raise ValueError(
                "image_quality.usability.{} must be PASS, WARN, or FAIL".format(
                    name
                )
            )

    aperture_settings = settings["apertures"]
    aperture_radii = (
        float(aperture_settings.get("small_radius_fwhm", 1.0)),
        float(aperture_settings.get("large_radius_fwhm", 2.5)),
        float(aperture_settings.get("sky_inner_radius_fwhm", 4.0)),
        float(aperture_settings.get("sky_outer_radius_fwhm", 7.0)),
    )
    if not 0 < aperture_radii[0] < aperture_radii[1] < aperture_radii[2] < aperture_radii[3]:
        raise ValueError(
            "Aperture radii must satisfy 0 < small < large < sky inner < sky outer"
        )
    if int(aperture_settings.get("minimum_sky_pixels", 50)) < 1:
        raise ValueError("apertures.minimum_sky_pixels must be positive")
    if int(aperture_settings.get("subpixels", 5)) < 1:
        raise ValueError("apertures.subpixels must be positive")
    if aperture_settings.get("subpixel_method", "exact") not in {
        "exact", "subpixel", "center"
    }:
        raise ValueError(
            "apertures.subpixel_method must be exact, subpixel, or center"
        )
    if not 0 < float(aperture_settings.get("minimum_unmasked_fraction", 0.80)) <= 1:
        raise ValueError("apertures.minimum_unmasked_fraction must be in (0, 1]")
    if aperture_settings.get("local_background_estimator", "median") not in {
        "median", "mean"
    }:
        raise ValueError("apertures.local_background_estimator must be median or mean")
    if float(aperture_settings.get("local_background_sigma_clip", 3.0)) <= 0:
        raise ValueError("apertures.local_background_sigma_clip must be positive")
    if float(aperture_settings.get("minimum_uncertainty", 1.0e-6)) <= 0:
        raise ValueError("apertures.minimum_uncertainty must be positive")

    psf_settings = settings["psf"]
    if psf_settings.get("model", "empirical") not in {"empirical", "moffat", "gaussian"}:
        raise ValueError("psf.model must be empirical, moffat, or gaussian")
    if psf_settings.get("fallback_model", "moffat") not in {"moffat", "gaussian"}:
        raise ValueError("psf.fallback_model must be moffat or gaussian")
    box_size = int(psf_settings.get("box_size_pixels", 25))
    if box_size < 7 or box_size % 2 == 0:
        raise ValueError("psf.box_size_pixels must be an odd integer of at least 7")
    if int(psf_settings.get("oversampling", 2)) < 1:
        raise ValueError("psf.oversampling must be at least 1")
    minimum_psf_stars = int(psf_settings.get("minimum_stars", 5))
    fallback_psf_stars = int(psf_settings.get("minimum_fallback_stars", 1))
    maximum_psf_stars = int(psf_settings.get("maximum_stars", 20))
    if not 1 <= fallback_psf_stars <= minimum_psf_stars <= maximum_psf_stars:
        raise ValueError(
            "PSF star counts must satisfy 1 <= fallback minimum <= empirical "
            "minimum <= maximum"
        )
    if not 0 <= float(psf_settings.get("maximum_masked_fraction", 0.05)) < 1:
        raise ValueError("psf.maximum_masked_fraction must be in [0, 1)")
    residual_warn = float(psf_settings.get("residual_warn_fraction", 0.15))
    residual_fail = float(psf_settings.get("residual_fail_fraction", 0.30))
    if not 0 <= residual_warn <= residual_fail:
        raise ValueError("PSF residual limits must satisfy 0 <= warn <= fail")
    if not 0 <= float(psf_settings.get("minimum_correlation", 0.90)) <= 1:
        raise ValueError("psf.minimum_correlation must be in [0, 1]")
    spatial_grid = psf_settings.get("spatial_grid", [3, 3])
    if len(spatial_grid) != 2 or any(int(value) <= 0 for value in spatial_grid):
        raise ValueError("psf.spatial_grid must contain two positive integers")
    if int(psf_settings.get("spatial_order", 0)) < 0:
        raise ValueError("psf.spatial_order cannot be negative")
    if int(psf_settings.get("model_version", 1)) <= 0:
        raise ValueError("psf.model_version must be positive")
    psf_statuses = set(psf_settings.get("approved_statuses", ["PASS", "WARN"]))
    if not psf_statuses or not psf_statuses.issubset({"PASS", "WARN", "FAIL"}):
        raise ValueError("psf.approved_statuses must contain PASS, WARN, or FAIL")

    calibration_settings = settings["calibration"]
    if int(calibration_settings.get("minimum_stars", 3)) < 1:
        raise ValueError("calibration.minimum_stars must be positive")
    if int(calibration_settings.get("maximum_stars", 100)) < int(
        calibration_settings.get("minimum_stars", 3)
    ):
        raise ValueError(
            "calibration.maximum_stars cannot be below calibration.minimum_stars"
        )
    if float(calibration_settings.get("minimum_star_snr", 10.0)) <= 0:
        raise ValueError("calibration.minimum_star_snr must be positive")
    if float(calibration_settings.get("maximum_catalog_error_mag", 0.10)) <= 0:
        raise ValueError("calibration.maximum_catalog_error_mag must be positive")
    if float(calibration_settings.get("sigma_clip", 3.0)) <= 0:
        raise ValueError("calibration.sigma_clip must be positive")
    if int(calibration_settings.get("minimum_aperture_correction_stars", 3)) < 1:
        raise ValueError(
            "calibration.minimum_aperture_correction_stars must be positive"
        )
    if calibration_settings.get("reference_aperture_method", "large_aperture") not in {
        "small_aperture", "large_aperture", "psf"
    }:
        raise ValueError("calibration.reference_aperture_method is not recognized")
    if float(calibration_settings.get("detection_sigma", 3.0)) <= 0:
        raise ValueError("calibration.detection_sigma must be positive")

    subtraction_settings = settings["subtraction"]
    subtraction_method = subtraction_settings["method"]
    if subtraction_method not in {"hotpants", "pyzogy"}:
        raise ValueError(
            "subtraction.method must be hotpants or pyzogy"
        )
    fallback_method = subtraction_settings.get("fallback_method")
    if fallback_method not in {None, "hotpants", "pyzogy"}:
        raise ValueError(
            "subtraction.fallback_method must be None, hotpants, or pyzogy"
        )
    for name in (
        "minimum_coverage_fraction",
        "maximum_template_saturated_fraction",
        "maximum_residual_fraction",
        "maximum_dipole_fraction",
        "maximum_flux_bias_fraction",
    ):
        value = float(subtraction_settings.get(name, 0.0))
        if not 0 <= value <= 1:
            raise ValueError("subtraction.{} must be in [0, 1]".format(name))
    for name in (
        "template_margin_arcmin",
        "maximum_template_fwhm_ratio",
        "maximum_alignment_rms_pixels",
        "maximum_noise_ratio",
        "execution_timeout_s",
    ):
        if float(subtraction_settings.get(name, 0.0)) <= 0:
            raise ValueError("subtraction.{} must be positive".format(name))
    if int(subtraction_settings.get("resampling_order", 3)) not in {0, 1, 2, 3}:
        raise ValueError("subtraction.resampling_order must be between 0 and 3")
    if int(subtraction_settings.get("resampling_tile_rows", 256)) <= 0:
        raise ValueError("subtraction.resampling_tile_rows must be positive")
    if int(subtraction_settings.get("maximum_mosaic_pixels", 100000000)) <= 0:
        raise ValueError("subtraction.maximum_mosaic_pixels must be positive")
    if int(subtraction_settings.get("minimum_scale_pixels", 1000)) < 10:
        raise ValueError("subtraction.minimum_scale_pixels must be at least 10")
    if int(subtraction_settings.get("blank_aperture_count", 50)) < 1:
        raise ValueError("subtraction.blank_aperture_count must be positive")
    if int(subtraction_settings.get("minimum_quality_stars", 3)) < 1:
        raise ValueError("subtraction.minimum_quality_stars must be positive")
    hotpants_settings = subtraction_settings.get("hotpants", {})
    stamp_count = hotpants_settings.get("stamp_count", "auto")
    if stamp_count != "auto" and int(stamp_count) < 1:
        raise ValueError("subtraction.hotpants.stamp_count must be positive or auto")
    stamp_grid = hotpants_settings.get("stamp_grid", "auto")
    if stamp_grid != "auto" and (
        len(stamp_grid) != 2 or any(int(value) < 1 for value in stamp_grid)
    ):
        raise ValueError(
            "subtraction.hotpants.stamp_grid must contain two positive integers"
        )
    difference_settings = subtraction_settings.get("photometry", {})
    if int(difference_settings.get("minimum_empty_apertures", 20)) < 1:
        raise ValueError(
            "subtraction.photometry.minimum_empty_apertures must be positive"
        )
    uncertainty_warn = float(
        difference_settings.get("uncertainty_warn_ratio", 1.25)
    )
    uncertainty_fail = float(
        difference_settings.get("uncertainty_fail_ratio", 2.0)
    )
    if not 1 <= uncertainty_warn <= uncertainty_fail:
        raise ValueError(
            "difference uncertainty ratios must satisfy 1 <= warn <= fail"
        )
    for name in (
        "dipole_sigma",
        "dipole_minimum_separation_pixels",
        "inverted_residual_sigma",
        "detection_sigma",
    ):
        if float(difference_settings.get(name, 1.0)) <= 0:
            raise ValueError("subtraction.photometry.{} must be positive".format(name))
    dipole_ratio = float(difference_settings.get("dipole_ratio_threshold", 0.25))
    if not 0 < dipole_ratio <= 1:
        raise ValueError(
            "subtraction.photometry.dipole_ratio_threshold must be in (0, 1]"
        )
    methods = {"small_aperture", "large_aperture", "psf"}
    for name in ("preferred_difference_methods", "preferred_science_methods"):
        values = difference_settings.get(name, [])
        if not values or not set(values).issubset(methods):
            raise ValueError(
                "subtraction.photometry.{} contains an unknown method".format(name)
            )

    target_ra = settings["target_position"]["ra"]
    target_dec = settings["target_position"]["dec"]
    target_settings = settings["target_position"]
    if target_settings.get("resample_science_images", False):
        raise ValueError(
            "target_position.resample_science_images must remain False; only "
            "derived detection-stack inputs may be reprojected"
        )
    if target_settings.get("user_position_mode", "prior") not in {"prior", "fixed"}:
        raise ValueError(
            "target_position.user_position_mode must be 'prior' or 'fixed'"
        )
    if target_settings.get("stack_combine", "weighted_mean") not in {
        "weighted_mean", "median"
    }:
        raise ValueError(
            "target_position.stack_combine must be weighted_mean or median"
        )
    if target_settings.get("stack_normalization", "zeropoint") not in {
        "none", "exposure", "zeropoint"
    }:
        raise ValueError(
            "target_position.stack_normalization must be none, exposure, or zeropoint"
        )
    if target_settings.get("multifilter_normalization", "background_rms") not in {
        "none", "background_rms"
    }:
        raise ValueError(
            "target_position.multifilter_normalization must be none or background_rms"
        )
    if int(target_settings.get("relative_alignment_minimum_common_stars", 6)) < 3:
        raise ValueError(
            "target_position.relative_alignment_minimum_common_stars must be at least 3"
        )
    if int(target_settings.get("stack_reprojection_order", 1)) not in {0, 1, 2, 3}:
        raise ValueError(
            "target_position.stack_reprojection_order must be between 0 and 3"
        )
    if int(target_settings.get("stack_reprojection_tile_rows", 256)) <= 0:
        raise ValueError(
            "target_position.stack_reprojection_tile_rows must be positive"
        )
    allowed_statuses = {"PASS", "WARN", "FAIL"}
    for name in ("coordinate_allowed_statuses", "stack_allowed_statuses"):
        values = target_settings.get(name, ["PASS", "WARN"])
        if not values or not set(values).issubset(allowed_statuses):
            raise ValueError(
                "target_position.{} must contain PASS, WARN, or FAIL".format(name)
            )
    positive_names = (
        "relative_alignment_target_rms_arcsec",
        "relative_alignment_warn_rms_arcsec",
        "relative_alignment_fail_rms_arcsec",
        "prior_uncertainty_arcsec",
        "centroid_search_radius_arcsec",
        "centroid_minimum_snr",
        "maximum_filter_shift_arcsec",
        "maximum_coordinate_uncertainty_arcsec",
        "minimum_coordinate_uncertainty_arcsec",
    )
    for name in positive_names:
        if float(target_settings.get(name, 1.0)) <= 0:
            raise ValueError("target_position.{} must be positive".format(name))
    relative_target = float(
        target_settings.get("relative_alignment_target_rms_arcsec", 0.20)
    )
    relative_warn = float(
        target_settings.get("relative_alignment_warn_rms_arcsec", 0.50)
    )
    relative_fail = float(
        target_settings.get("relative_alignment_fail_rms_arcsec", 1.50)
    )
    if not relative_target <= relative_warn <= relative_fail:
        raise ValueError(
            "target_position relative-alignment RMS limits must satisfy "
            "target <= warn <= fail"
        )
    centroid_inner = float(
        target_settings.get("centroid_background_inner_fwhm", 3.0)
    )
    centroid_outer = float(
        target_settings.get("centroid_background_outer_fwhm", 6.0)
    )
    if centroid_inner < 0 or centroid_outer <= centroid_inner:
        raise ValueError(
            "target_position centroid background radii must satisfy "
            "0 <= inner < outer"
        )
    if float(target_settings.get("centroid_aperture_radius_fwhm", 1.5)) <= 0:
        raise ValueError(
            "target_position.centroid_aperture_radius_fwhm must be positive"
        )
    if int(target_settings.get("stack_minimum_images_per_filter", 1)) <= 0:
        raise ValueError(
            "target_position.stack_minimum_images_per_filter must be positive"
        )
    if float(
        target_settings.get("minimum_coordinate_uncertainty_arcsec", 0.02)
    ) > float(target_settings.get("maximum_coordinate_uncertainty_arcsec", 1.0)):
        raise ValueError(
            "target_position minimum coordinate uncertainty cannot exceed maximum"
        )
    if int(target_settings.get("target_coordinate_version", 1)) <= 0:
        raise ValueError(
            "target_position.target_coordinate_version must be positive"
        )

    if (target_ra is None) != (target_dec is None):
        raise ValueError(
            "target_position.ra and target_position.dec must both be set "
            "or both be None"
        )

    sigma_levels = settings["upper_limits"]["sigma_levels"]
    if not sigma_levels or any(level <= 0 for level in sigma_levels):
        raise ValueError(
            "upper_limits.sigma_levels must contain positive values"
        )
    limit_settings = settings["upper_limits"]
    if int(limit_settings.get("number_empty_apertures", 100)) < 1:
        raise ValueError("upper_limits.number_empty_apertures must be positive")
    if not 1 <= int(limit_settings.get("minimum_empty_apertures", 20)) <= int(
        limit_settings.get("number_empty_apertures", 100)
    ):
        raise ValueError(
            "upper_limits.minimum_empty_apertures must be between 1 and the "
            "requested empty-aperture count"
        )
    allowed_limit_methods = {"small_aperture", "large_aperture", "psf"}
    if not set(limit_settings.get("empty_aperture_methods", [])).issubset(
        allowed_limit_methods
    ):
        raise ValueError("upper_limits.empty_aperture_methods contains an unknown method")

    batch_settings = settings["batch_consistency"]
    if int(batch_settings.get("minimum_comparison_epochs", 3)) < 2:
        raise ValueError("batch_consistency.minimum_comparison_epochs must be at least 2")
    if int(batch_settings.get("minimum_stable_comparison_stars", 3)) < 1:
        raise ValueError(
            "batch_consistency.minimum_stable_comparison_stars must be positive"
        )
    rms_warn = float(batch_settings.get("comparison_star_rms_warn_mag", 0.05))
    rms_fail = float(batch_settings.get("comparison_star_rms_fail_mag", 0.15))
    if not 0 < rms_warn <= rms_fail:
        raise ValueError("comparison-star RMS limits must satisfy 0 < warn <= fail")
    chi_warn = float(batch_settings.get("comparison_star_reduced_chi2_warn", 3.0))
    chi_fail = float(batch_settings.get("comparison_star_reduced_chi2_fail", 10.0))
    if not 1 <= chi_warn <= chi_fail:
        raise ValueError("comparison-star chi-square limits must satisfy 1 <= warn <= fail")
    metric_warn = float(batch_settings.get("metric_outlier_warn_sigma", 3.5))
    metric_fail = float(batch_settings.get("metric_outlier_fail_sigma", 6.0))
    if not 0 < metric_warn <= metric_fail:
        raise ValueError("batch metric limits must satisfy 0 < warn <= fail")
    fraction_warn = float(batch_settings.get("problem_group_warn_fraction", 0.25))
    fraction_fail = float(batch_settings.get("problem_group_fail_fraction", 0.50))
    if not 0 <= fraction_warn <= fraction_fail <= 1:
        raise ValueError("batch group fractions must satisfy 0 <= warn <= fail <= 1")
    if int(batch_settings.get("minimum_group_images", 2)) < 1:
        raise ValueError("batch_consistency.minimum_group_images must be positive")
    allowed_methods = {"small_aperture", "large_aperture", "psf"}
    if not set(batch_settings.get("comparison_methods", [])).issubset(allowed_methods):
        raise ValueError("batch_consistency.comparison_methods contains an unknown method")
    allowed_preferences = {
        "{}:{}".format(kind, method)
        for kind in ("science", "difference") for method in allowed_methods
    }
    preferred = batch_settings.get("preferred_order", [])
    if not preferred or not set(preferred).issubset(allowed_preferences):
        raise ValueError("batch_consistency.preferred_order contains an unknown result")
    statuses = set(batch_settings.get("accepted_image_statuses", ["PASS", "WARN"]))
    if not statuses or not statuses.issubset({"PASS", "WARN", "FAIL"}):
        raise ValueError(
            "batch_consistency.accepted_image_statuses must contain PASS, WARN, or FAIL"
        )
    ensemble = batch_settings.get("ensemble_correction", {})
    if not set(ensemble.get("components", [])).issubset({"epoch", "telescope"}):
        raise ValueError("ensemble correction components must be epoch or telescope")
    if int(ensemble.get("minimum_stars", 3)) < 1:
        raise ValueError("ensemble_correction.minimum_stars must be positive")
    if float(ensemble.get("maximum_absolute_correction_mag", 0.50)) <= 0:
        raise ValueError("ensemble maximum correction must be positive")


def resolve_settings(instrument_name=None, run_settings=None, filter_name=None, filter_settings=None,
                     image_name=None, image_overrides=None, validate=True):
    """
    Resolve the complete settings for one image.

    Settings are applied in the following order:

        general defaults
        instrument defaults
        run settings
        filter settings
        individual-image overrides

    The returned dictionary is fully independent of all input dictionaries.
    """
    resolved = get_default_settings()
    profile = normalize_instrument_name(instrument_name)

    if profile in INSTRUMENT_DEFAULTS:
        resolved = merge_settings(
            resolved,
            get_instrument_settings(profile),
        )

    if run_settings is not None:
        resolved = merge_settings(resolved, run_settings)

    normalized_filter = normalize_filter_name(filter_name)

    if normalized_filter is None:
        normalized_filter = normalize_filter_name(
            resolved["metadata"].get("filter_override")
        )

    if normalized_filter is not None:
        resolved = merge_settings(
            resolved,
            get_filter_settings(
                normalized_filter,
                filter_settings=filter_settings,
            ),
        )

    resolved = merge_settings(
        resolved,
        get_image_settings(
            image_name,
            image_overrides=image_overrides,
        ),
    )

    if profile in INSTRUMENT_DEFAULTS:
        resolved["instrument"]["profile"] = profile

    if instrument_name is not None:
        resolved["instrument"]["name"] = str(instrument_name)

    if validate:
        validate_settings(resolved)

    return resolved


__all__ = [
    "DEFAULT_SETTINGS",
    "FILTER_ALIASES",
    "FILTER_DEFAULTS",
    "INSTRUMENT_ALIASES",
    "INSTRUMENT_DEFAULTS",
    "MISSING_VALUE",
    "MISSING_VALUE_POLICY",
    "QUALITY_FLAGS",
    "STANDARD_UNITS",
    "get_default_settings",
    "get_filter_settings",
    "get_image_settings",
    "get_instrument_settings",
    "merge_settings",
    "normalize_filter_name",
    "normalize_instrument_name",
    "resolve_settings",
    "validate_settings",
]
