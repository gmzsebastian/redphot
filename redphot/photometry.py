import pathlib
import warnings
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
import subprocess
import glob
from pyraf import iraf
from pyraf.iraf import noao
from pyraf.iraf import imred
from pyraf.iraf import digiphot
from pyraf.iraf import daophot
from astropy.modeling import models, fitting
import suds
from astropy.coordinates import SkyCoord
from astropy import units as u
import numpy as np
from astropy import table
from astropy import wcs
from photutils import CircularAperture
import matplotlib.pyplot as plt
from photutils import DAOStarFinder
import os
from scipy.optimize import curve_fit, minimize
from photutils import aperture_photometry
import matplotlib.patheffects as PathEffects
import sys
import mastcasjobs
import matplotlib as mpl
import logging
from pyraf.iraf import phot
mpl.rcParams['figure.figsize'] = 16,16
mpl.rcParams['font.size'] = 15

########################################################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################
############################################################# IMAGE SUBTRACTION FUNCTIONS ##############################################################################
########################################################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################

from astropy.nddata import Cutout2D
from reproject import reproject_interp
from astropy.wcs import WCS, _wcs
import requests
from astropy.nddata import CCDData, NDData
from photutils import psf, EPSFBuilder
from io import BytesIO

def read_with_datasec(filename, hdu=0):
    ccddata = CCDData.read(filename, format='fits', unit='adu', hdu=hdu)
    if 'datasec' in ccddata.meta:
        jmin, jmax, imin, imax = eval(ccddata.meta['datasec'].replace(':', ','))
        ccddata = ccddata[imin-1:imax, jmin-1:jmax]
    return ccddata

def get_ccd_bbox(ccddata):
    corners = [[0.], [0.5], [1.]] * np.array(ccddata.shape)[::-1]
    (ra_min, dec_min), (ra_ctr, dec_ctr), (ra_max, dec_max) = ccddata.wcs.all_pix2world(corners, 0.)
    if ra_min > ra_max:
        ra_min, ra_max = ra_max, ra_min
    if dec_min > dec_max:
        dec_min, dec_max = dec_max, dec_min
    max_size_dec = 0.199
    if dec_max - dec_min > max_size_dec:
        dec_min = dec_ctr - max_size_dec / 2.
        dec_max = dec_ctr + max_size_dec / 2.
    return ra_min, dec_min, ra_max, dec_max

def make_psf(data, catalog, show=False, boxsize=25.):
    catalog = catalog.copy()
    catalog['x'], catalog['y'] = data.wcs.all_world2pix(flot(catalog['RA']), flot(catalog['DEC']), 0)
    bkg = np.nanmedian(data)
    nddata = NDData(data - bkg)

    stars = psf.extract_stars(nddata, catalog, size=boxsize)
    try:
        epsf_builder = EPSFBuilder(oversampling=2.)
        epsf, fitted_stars = epsf_builder(stars)
    except:
        epsf_builder = EPSFBuilder(oversampling=4.)
        epsf, fitted_stars = epsf_builder(stars)

    #if show:
    #    plt.figure()
    #    plt.imshow(epsf.data)
    #    plot_stars(fitted_stars)

    return epsf, fitted_stars

def plot_stars(stars):
    nrows = int(np.ceil(len(stars) ** 0.5))
    fig, axarr = plt.subplots(nrows, nrows, figsize=(20, 20), squeeze=True)
    for ax, star in zip(axarr.ravel(), stars):
        ax.imshow(star)
        ax.plot(star.cutout_center[0], star.cutout_center[1], 'r+')

def update_wcs(wcs, p):
    wcs.wcs.crval += p[:2]
    c, s = np.cos(p[2]), np.sin(p[2])
    if wcs.wcs.has_cd():
        wcs.wcs.cd = wcs.wcs.cd @ np.array([[c, -s], [s, c]]) * p[3]
    if wcs.wcs.has_pc():
        wcs.wcs.pc = wcs.wcs.pc @ np.array([[c, -s], [s, c]]) * p[3]

def wcs_offset(p, radec, xy, origwcs):
    wcs = origwcs.deepcopy()
    update_wcs(wcs, p)
    test_xy = wcs.all_world2pix(radec, 0)
    rms = (np.sum((test_xy - xy)**2) / len(radec))**0.5
    return rms

def refine_wcs(wcs, stars, catalog, use_sep=False):
    if use_sep:
        xy = np.array([[star['x'], star['y']] for star in stars])
        t_match = catalog[stars['i']]
    else:
        xy = np.array([star.center for star in stars.all_good_stars])
        t_match = catalog[[star.id_label - 1 for star in stars.all_good_stars]]
    radec = np.array([flot(t_match['RA']), flot(t_match['DEC'])]).T

    res = minimize(wcs_offset, [0., 0., 0., 1.], args=(radec, xy, wcs),
                                  bounds=[(-0.01, 0.01), (-0.01, 0.01), (-0.1, 0.1), (0.9, 1.1)])

    orig_rms = wcs_offset(res.x, radec, xy, wcs)
    print(' orig_fun: {}'.format(orig_rms))
    print(res)
    update_wcs(wcs, res.x)

def get_ps1_filename(ra, dec, filt):
    """
    Download Image from PS1 and correct luptitudes back to a linear scale.

    Parameters
    ---------------
    ra, dec : Coordinates in degrees
    filt    : Filter color 'g', 'r', 'i', 'z', or 'y'

    Output
    ---------------
    filename : PS1 image filename
    """

    # Query a center RA and DEC from PS1 in a specified color
    res = requests.get('http://ps1images.stsci.edu/cgi-bin/ps1filenames.py',
                 params={'ra': ra, 'dec': dec, 'filters': filt})
    t = table.Table.read(res.text, format='ascii')

    return t['filename'][0]

def download_ps1_image(filename, saveas=None):
    """
    Download image from PS1 and correct luptitudes back to a linear scale.

    Parameters
    ---------------
    filename : PS1 image filename (from `get_ps1_filename`)
    saveas   : Path to save template file (default: do not save)

    Output
    ---------------
    ccddata : CCDData format of data with WCS
    """
    res = requests.get('http://ps1images.stsci.edu' + filename)
    hdulist = fits.open(BytesIO(res.content))

    # Linearize from luptitudes
    boffset = hdulist[1].header['boffset']
    bsoften = hdulist[1].header['bsoften']
    data_linear = boffset + bsoften * 2 * np.sinh(hdulist[1].data * np.log(10.) / 2.5)
    warnings.simplefilter('ignore')  # ignore warnings from nonstandard PS1 header keywords
    ccddata = CCDData(data_linear, wcs=WCS(hdulist[1].header), unit='adu')

    # Save the template to file
    if saveas is not None:
        ccddata.write(saveas, overwrite=True)

    return ccddata

def download_references(ra_min, dec_min, ra_max, dec_max, mag_filter, template_basename=None, catalog=None, boxsize = 25):
    """
    Download 1 to 4 references from PS1 as necessary to cover full RA & dec range

    Parameters
    ---------------
    ra_min, ra_max   : Minimum and Maximum RA and DEC
    dec_min, dec_max   in units of degrees
    mag_filter       : Filter color 'g', 'r', 'i', 'z', or 'y'
    template_basename: Filename of the output(s), to be suffixed by 0.fits, 1.fits, ...
    catalog          : Catalog to which to align the reference image WCS (default: do not align)
    boxsize          : Boxsize for PSF fitter

    Output
    ---------------
    refdatas   : List of CCDData objects containing the reference images

    """

    filename0 = get_ps1_filename(ra_min, dec_min, mag_filter)
    filename1 = get_ps1_filename(ra_max, dec_max, mag_filter)
    filename2 = get_ps1_filename(ra_min, dec_max, mag_filter)
    filename3 = get_ps1_filename(ra_max, dec_min, mag_filter)

    filenames = {filename0, filename1, filename2, filename3}
    refdatas = []
    for i, fn in enumerate(filenames):
        if template_basename is not None:
            saveas = template_basename + '{:d}.fits'.format(i)
            print('downloading', saveas)
        else:
            saveas = None
            print('downloading', fn)
        refdata = download_ps1_image(fn, saveas)
        if catalog is not None:
            _, stars = make_psf(refdata, catalog, boxsize)
            #try:
            #    #refine_wcs(refdata.wcs, stars, catalog)
            #except _wcs.InvalidTransformError:
            #    print('WARNING: unable to refine wcs')
        refdatas.append(refdata)

    return refdatas

def assemble_reference(refdatas, wcs, shape):
    """Reproject and stack the reference images to match the science image"""
    refdatas_reprojected = []
    refdata_foot = np.zeros(shape, float)
    for data in refdatas:
        reprojected, foot = reproject_interp((data.data, data.wcs), wcs, shape)
        refdatas_reprojected.append(reprojected)
        refdata_foot += foot

    refdata_reproj = np.nanmean(refdatas_reprojected, axis=0)
    refdata_reproj[np.isnan(refdata_reproj)] = 0.
    refdata = CCDData(refdata_reproj, wcs=wcs, mask=refdata_foot == 0., unit='adu')
    return refdata

def generate_hotpants_image(image_name, parameters_list, cat_in, best_fwhm = 3.5, max_mag=16.0, min_mag=21.0, show = False, upper_limit_image = 0.95, upper_limit_template = 0.95, edge_crop = 0, force_download = False, boxsize = 25):
    '''
    Generate a subtracted image using hotpants

    Parameters
    ---------------
    image_name          : Name of the image to process
    parameters_list     : List of relevant parameters form loadimage()
    cat_in              : Catalogue file with stars coordiantes
    best_fwhm           : FWHM value of stars as determined by estimate_fwhm() in inputs of pixels.
    min_mag             : Dimmest magnitude allowed to be matched in the process
    max_mag             : Birghtest stars allowed to be matched in the process
    show                : Show image subtraction results
    upper_limits        : Maximum counts in images
    edge_crop           : Crop these many pixels from the edges
    force_download      : Force download a new template?
    boxsize             : Boxsize for PSF fitter
    '''

    # Read the science image with PyZOGY
    scidata0 = read_with_datasec(image_name)
    ccd_bbox = get_ccd_bbox(scidata0)

    # Crop catalog by magnitude
    color = parameters_list['filter']
    cat_in.sort(color + 'mag')
    catalogue_magnitudes = cat_in[color + 'mag'].astype(float)
    all_stars = np.where((catalogue_magnitudes < min_mag) & (catalogue_magnitudes > max_mag))[0]
    catalog = cat_in[all_stars]

    # Crop The Catalog before aligning
    header_data     = fits.open(image_name)
    wcs_data        = wcs.WCS(header_data[0].header)
    pix_RA, pix_DEC = wcs_data.wcs_world2pix(flot(catalog['RA']), flot(catalog['DEC']), 1)
    catalog_crop    = catalog[np.where((pix_RA > edge_crop) & (pix_RA < (scidata0.shape[0] - edge_crop)) & (pix_DEC > edge_crop) & (pix_DEC < (scidata0.shape[1] - edge_crop)))]

    # Pretend to make the PSF for the science image
    # This is just to find the centroids of the stars so we can update the WCS in the next step
    _, sci_stars = make_psf(scidata0, catalog_crop, show=show)

    # Update the WCS for the science image
    scidata = scidata0.copy()
    #refine_wcs(scidata.wcs, sci_stars, catalog_crop)
    scidata_crop = CCDData(scidata.data, wcs = scidata.wcs, unit = 'adu', header = scidata.header)
    # Write the cutout to a new FITS file
    cutout_filename = image_name[:-5] + '_science.fits'
    scidata_crop.write(cutout_filename, overwrite=True)

    science_shape = np.array([np.array(scidata.shape) * 1.2][0]).astype(int)
    science_wcs   = scidata.wcs
    science_wcs.wcs.crpix = science_wcs.wcs.crpix * 1.2

    # Get object name and template name
    object_name   = parameters_list['object']
    template_name = object_name + '_' + color + '_template.fits'

    # Download the reference image
    exists   = check_existence(template_name, '', verbose = False)
    # If it doesn't exist, query 3PI and get the stars near the target.
    if exists and force_download == False:
        print('Opening Existing Template')
        refdata = read_with_datasec(template_name)
    elif force_download == True:
        template_name = image_name[:-5][image_name[:-5].find('/') + 1 :] + '_' + color + '_template_temp.fits'
        exists = check_existence(template_name, '', verbose = False)
        if exists:
            refdata = read_with_datasec(template_name)
        else:
            print('Downloading Template')
            refdatas = download_references(*ccd_bbox, color, catalog=catalog_crop, boxsize = boxsize)
            refdata  = assemble_reference(refdatas, science_wcs, science_shape)
            refdata.write(template_name, overwrite=True)
    else:
        print('Downloading Template')
        refdatas = download_references(*ccd_bbox, color, catalog=catalog_crop, boxsize = boxsize)
        refdata  = assemble_reference(refdatas, science_wcs, science_shape)
        refdata.write(template_name, overwrite=True)

    # Crop The images
    # Position and Size of the cutout
    #position = (scidata.shape[0] / 2, scidata.shape[1] / 2)
    #size     = (scidata.shape[0] - edge_crop * 2, scidata.shape[1] - edge_crop * 2)
    # Crop the science image 
    #cutout_science = Cutout2D(scidata.data, position=position, size=size, wcs=scidata.wcs)

    # Import Template
    hdu1 = fits.open(cutout_filename)[0]
    hdu2 = fits.open(template_name)[0]
    # Reproject and Save
    im2new, im2footprint = reproject_interp(hdu2, hdu1.header)

    # Add pedestal to the template
    min_template = np.min(im2new.data)
    if min_template <= 0:
        im2new.data += np.abs(min_template)

    template_name_output = image_name[:-5] + '_template.fits'
    fits.writeto(template_name_output, im2new, hdu1.header, overwrite=True)

    # Crop the template image 
    #cutout_template = Cutout2D(refdata.data, position=position, size=size, wcs=refdata.wcs)
    #refdata_crop    = CCDData(cutout_template.data, wcs = cutout_template.wcs, unit = 'adu', header = refdata.header)

    # Write the cutout to a new FITS file
    #refdata_crop.write(template_name_output, overwrite=True)

    # Run hotpants on the generated images with these parameters
    output_image = image_name[:-5] + '_diff.fits'
    science_limit = np.nanmin((np.nanmax(hdu1.data) * upper_limit_image, 50000))
    template_limit = np.nanmin(np.nanmax(im2new.data) * upper_limit_template)
    gain = parameters_list['gain']

    # Determine width of gaussian
    sigma_gauss = best_fwhm / 2 * np.sqrt(2 * np.log(2))

    xmin, xmax, ymin, ymax = edge_crop, hdu1.shape[0] - edge_crop, edge_crop, hdu1.shape[1] - edge_crop
    #xmin, xmax, ymin, ymax = 100, 1400, 300, 1800

    # Run hotpants
    #os.system('hotpants -inim %s -tmplim %s -outim %s -n i -c t -tu %s -iu %s -ig %s -ng 3 6 %s 4 %s 2 %s -gd %s %s %s %s'%(cutout_filename, 
    #          template_name_output, output_image, template_limit, science_limit, gain, 0.5 * best_fwhm, 1.0 * best_fwhm, 2.0 * best_fwhm,
    #          xmin, xmax, ymin, ymax))
    os.system('hotpants -inim %s -tmplim %s -outim %s -n i -c t -tu %s -iu %s -ig %s -ng 3 6 %s 4 %s 2 %s'%(cutout_filename, 
              template_name_output, output_image, template_limit, science_limit, gain, 0.5 * best_fwhm, 1.0 * best_fwhm, 2.0 * best_fwhm))
    print('hotpants -inim %s -tmplim %s -outim %s -n i -c t -tu %s -iu %s -ig %s -ng 3 6 %s 4 %s 2 %s'%(cutout_filename, 
              template_name_output, output_image, template_limit, science_limit, gain, 0.5 * best_fwhm, 1.0 * best_fwhm, 2.0 * best_fwhm))

    return output_image

########################################################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################
############################################################# IMAGE SUBTRACTION FUNCTIONS ##############################################################################
########################################################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################

flot = lambda x : np.array(x).astype(float) 

def sigma_clipped_stats_version(data, sigma_low, sigma_hi, iterations):
    '''
    Same as sigma_clipped_stats, but checking for the Python version
    '''
    python_version = sys.version_info[0]
    if python_version == 3:
        mean, median, std = sigma_clipped_stats(data, sigma_lower=sigma_low, sigma_upper=sigma_hi, maxiters=iterations)
    elif python_version == 2:
        mean, median, std = sigma_clipped_stats(data, sigma_lower=sigma_low, sigma_upper=sigma_hi, iters=iterations)
    return mean, median, std

def header_value(header_data, names, output = '', vartype = float):
    '''
    Read in value from the header from the options are in names.
    If the output is not empty, then overwrite with that.
    The type of the header value must be same to the vartype variable.
    
    Parameters
    -------------
    header_data : Header file from .fits file
    names       : Array of names, i.e. ['GAIN', 'EGAIN']
    output      : Overwrite with this value if not empty
    vartype.    : Only return header value if it is this type

    Output
    --------------
    Float of header value
    '''

    for i in range(len(names)):
        if output == '':
            try:
                # Import Parameter Name
                parameter = header_data[0].header[names[i]]
                if vartype == float:
                    if type(float(parameter)) == vartype:
                        output = float(parameter)
                elif type(parameter) == vartype:
                    output = parameter
            except:
                pass

    return output

def loadimage(image_name, gain = '', rdnoise = '', RA = '', DEC = '', airmass = '', mjd = '', color = '', object_name = '', exptime = '', edge_crop = 0):
    '''
    Open a .fits image and read the picture data, WCS, coordinates, and header parameters.

    Parameters
    -------------
    image_name  : Name of the .fits iamge
    gain        : Image Gain
    rdnoise     : Read noise
    RA/DEC      : Coordinates
    airmass     : Airmass
    mjd         : Modified Julian Date
    color       : Filter of the image
    object_name : Name of the target
    exptime     : Exposure time
    edge_crop   : Crop these many pixels from the edges

    Output
    --------------
    image_data      : Numpy array of data
    wcs_data        : data in WCS format
    coord           : Coordiantes in SkyCoord format
    parameters_list : List of relevant parameters

    '''

    # Open image file
    header_data = fits.open(image_name)

    # FLWO needs to overwrite EPOCH label
    try:
        header_data[0].header.set('EPOCH', header_data[0].header['EQUINOX']) # Setting the value of EPOCH to the value of EQUINOX
    except:
        pass

    # Import Header Values
    gain_value    = header_value(header_data, ['GAIN', 'EGAIN', 'HIERARCH CELL.GAIN', 'GAINDL'], gain        )
    rdnoise_value = header_value(header_data, ['RDNOISE', 'ENOISE', 'HIERARCH CELL.READNOISE'] , rdnoise     )
    ra_value      = header_value(header_data, ['RA']                                           , RA          , vartype = str)
    dec_value     = header_value(header_data, ['DEC']                                          , DEC         , vartype = str)
    airmass_value = header_value(header_data, ['AIR', 'AIRMASS', 'SECZ']                       , airmass     )
    mjd_value     = header_value(header_data, ['MJD', 'MJD-OBS', 'JD']                         , mjd         )
    filter_value  = header_value(header_data, ['FILTER', 'HIERARCH FPA.FILTER', 'CCDFLTID']    , color       , vartype = str)
    object_value  = header_value(header_data, ['OBJECT']                                       , object_name , vartype = str)
    exptime_value = header_value(header_data, ['EXPTIME']                                      , exptime     )

    # MJD might be JD
    if mjd_value > 2000000:
        mjd_value -= 2400000.5

    # Replace Header values
    all_vals = np.array([gain, rdnoise, RA, DEC, airmass, mjd, color, object_name, exptime])
    if np.any(all_vals != ''):
        fits.setval(image_name, 'EXPTIME', value = exptime_value)
        fits.setval(image_name, 'AIR',     value = airmass_value)
        fits.setval(image_name, 'FILTER',  value = filter_value )
        fits.setval(image_name, 'MJD',     value = mjd_value    )
        fits.setval(image_name, 'RDNOISE', value = rdnoise_value)
        fits.setval(image_name, 'GAIN',    value = gain_value   )

    # Crop Image a little
    #xmin, xmax, ymin, ymax = edge_crop, header_data[0].data.shape[0] - edge_crop, edge_crop, header_data[0].data.shape[1] - edge_crop
    #fits.setval(image_name, 'DATASEC',   value = '[%s:%s,%s:%s]'%(xmin, xmax, ymin, ymax))
    #fits.setval(image_name, 'TRIMSEC',   value = '[%s:%s,%s:%s]'%(xmin, xmax, ymin, ymax))
    #fits.setval(image_name, 'ORIGSEC',   value = '[%s:%s,%s:%s]'%(xmin, xmax, ymin, ymax))
    #fits.setval(image_name, 'CCDSEC',    value = '[%s:%s,%s:%s]'%(xmin, xmax, ymin, ymax))

    # Import image data and WCS
    image_data = header_data[0].data
    wcs_data   = wcs.WCS(header_data[0].header)

    # Close image file
    header_data.close()

    # Convert targetcoords to SkyCoords format
    coord = SkyCoord(ra_value, dec_value, unit=(u.hourangle, u.deg))

    # Fix filter if not a normal filter
    g_filters = ['g-ZTF', 'g_filter', 'g-SM-SkyMapper', 'g-Sloan', 'g_Sloan', 'Sloan_g', 'g_sloan', 'g.00000', 'gS', 'gp']
    r_filters = ['r-ZTF', 'r_filter', 'R', 'R-Cousins', 'r-SM-SkyMapper', 'r_Sloan', 'Sloan_r', 'r_sloan', 'r.00000', 'rS', 'rp']
    i_filters = ['i-ZTF', 'i_filter', 'i-Sloan', 'i-sloan', 'I-Cousins', 'I', 'i_Sloan', 'Sloan_i', 'i_sloan', 'i.00000', 'iS', 'ip']
    z_filters = ['z-ZTF', 'z_filter', 'z-Sloan', 'z-sloan', 'Z', 'z_Sloan', 'Z_sloan', 'z.00000', 'zS', 'zp']
    y_filters = ['y-ZTF', 'y_filter', 'y-Sloan', 'y-sloan', 'Y', 'y_Sloan', 'Y_sloan', 'y.00000', 'yS', 'yp']

    if   filter_value in g_filters: filter_value = 'g'
    elif filter_value in r_filters: filter_value = 'r'
    elif filter_value in i_filters: filter_value = 'i'
    elif filter_value in z_filters: filter_value = 'z'
    elif filter_value in y_filters: filter_value = 'y'

    parameters_list = {'gain'    : gain_value    ,
                       'rdnoise' : rdnoise_value ,
                       'ra'      : ra_value      ,
                       'dec'     : dec_value     ,
                       'airmass' : airmass_value ,
                       'mjd'     : mjd_value + (exptime_value / 86400) / 2.0,
                       'filter'  : filter_value  ,
                       'object'  : object_value  ,
                       'exptime' : exptime_value }

    return image_data, wcs_data, coord, parameters_list

def get_psf_kron(cat, color):
    '''
    Extract the PSF and Kron magntiudes and make any values
    that are -999 not a number.

    Parameters
    -------------
    cat: Name of the catalogue being queried
    color: name of the filter color (i.e. g)

    Output
    ------------
    - PSF magnitude array
    - Kron magnitude array
    '''

    # Get the PSF and Kron magnitude of each star in the catalogue
    psf_mag  = np.array(cat[color + 'mag' ]).astype(float)
    kron_mag = np.array(cat[color + 'kron']).astype(float)

    # Make any invalid values equal to nan
    psf_mag[psf_mag   == -999] = 'nan'
    kron_mag[kron_mag == -999] = 'nan'

    return psf_mag, kron_mag

def get_positions(image_name, image_data, cat_in, wcs_data, coord, parameters_list, best_fwhm, average_background, background_high, background_low, min_mag = 21, max_mag = 16, pix_range = 20, detection_threshold = 2.0, initial_sigma_clip = 1.5, initial_snr_cut = 10.0, overexposed_limit = 50000):
    '''
    Use the coordinates of the catalogue to find stars in the image with name image_name.
    Save a file with the x,y coordinates of the stars in pixel units. And plot the image
    with the apertures of the selected stars.

    Parameters
    ---------------
    image_name          : Name of the image to process
    image_data          : Numpy array of data
    cat_in              : Catalogue file with stars coordiantes
    wcs_data            : Output form loadimage() with wcs data
    coord               : Coordiantes in SkyCoord format form loadimage()
    parameters_list     : List of relevant parameters form loadimage()
    best_fwhm           : FWHM value of stars as determined by estimate_fwhm() in inputs of pixels.
    average_background  : Average background value of stars as determined by estimate_fwhm()
    background_high     : Upper std sigma value of the background
    background_low      : Lower std sigma value of the background
    min_mag             : Dimmest magnitude allowed to be matched in the process
    max_mag             : Birghtest stars allowed to be matched in the process
    pix_range           : Search radius for the corresponding star in the image from
                          the catalogue.
    detection_threshold : threshold above the background that a star would be detected.
    initial_sigma_clip  : Sigma from the average magnitude difference from which to
                          remove stars from
    initial_snr_cut     : Minimum signal to noise that a star should have to be accepted
    overexposed_limit   : Maximum number of counts before discarding the star

    Output
    ---------------
    Catalogue with the stars that will be used for the comparison.
    '''   

    # Sort catalogue
    color = parameters_list['filter']
    cat_in.sort(color + 'mag')
    
    ## Select the stars that will go into the FWHM calculation
    # Get the magniudes in the right color and in float format
    catalogue_magnitudes = cat_in[color + 'mag'].astype(float)

    # Select the stars that will be used to estimate the FWHM
    all_stars = np.where((catalogue_magnitudes < min_mag) & (catalogue_magnitudes > max_mag))[0]

    # Get pixel positions of the catalogue stars in the image.
    cat_coords_arr    = np.transpose(np.array([cat_in['RA'][all_stars].astype(float),cat_in['DEC'][all_stars].astype(float)]))
    cat_coords_pix    = wcs_data.wcs_world2pix(cat_coords_arr, 1)
    target_coords_img = wcs_data.wcs_world2pix(coord.ra.deg, coord.dec.deg, 1)

    # selected stars that are inside the image
    datasize_y = image_data.shape[0]
    datasize_x = image_data.shape[1]
    selected   = np.where((cat_coords_pix.T[0] > pix_range) & (cat_coords_pix.T[0] < datasize_x - pix_range) & (cat_coords_pix.T[1] > pix_range) & (cat_coords_pix.T[1] < datasize_y - pix_range))

    # Stars to be used
    selected_coordinates = cat_coords_pix[selected]

    def get_catalog(initial_snr_cut):
        # Empty variable for future use
        first = 0
        # For each of the stars in the catalog, find the corresponding star in the image
        for i in range(len(selected_coordinates)):
            # Select a small area of size pix_range pixels around the catalogue star
            cat_star = selected_coordinates[i]

            # Select a small area of pix_range pixels around the catalogue star
            if np.isfinite(cat_star[0]):
                xmin = int(np.around(cat_star[0]-pix_range))
                xmax = int(np.around(cat_star[0]+pix_range))
                ymin = int(np.around(cat_star[1]-pix_range))
                ymax = int(np.around(cat_star[1]+pix_range))

                # Crop the data
                cropped_data = image_data[ymin:ymax,xmin:xmax]

                # Make sure there is data in the cropped region
                if len(cropped_data.flatten()) == 0:
                    print('No Data')
                    continue

                # Maximum value in the cropped image
                max_counts = np.max(cropped_data)

                # If the star is too close to the edge, skip it
                if cropped_data.shape != (pix_range*2, pix_range*2):
                    print('Selected star too close to the edge, picking a different one')
                    continue

                # If the star is too bright, skip it
                if max_counts > overexposed_limit:
                    print('Selected star is overexposed, picking a different one')
                    continue

                # Get the mean of the background and its standard deviation while attempting to remove bright stars
                mean_local, median_local, std_local = sigma_clipped_stats_version(cropped_data, sigma_low=2.0, sigma_hi=1.0, iterations=5)

                # Detect stars in using the DAOFIND (Stetson 1987) in an image for local density maxima
                # that have a peak amplitude greater than 'threshold' and have a size and shape similar
                # to the defined 2D Gaussian kernel
                daofind = DAOStarFinder(fwhm=best_fwhm, threshold=detection_threshold*background_high, sigma_radius = 3.0)
                sources = daofind(cropped_data - median_local)

                if sources:
                    # Continue if sources were found
                    if len(sources) > 0:
                        # Calculate chance coincidence from separation and magnitude of each star.
                        separation = np.sqrt((sources['xcentroid'] - pix_range) ** 2 + (sources['ycentroid'] - pix_range) ** 2)
                        magnitude  = sources['mag']
                        chance     = calculate_coincidence(separation, magnitude)

                        # And then select the best star if there was at least one found.
                        # And only if the star is not near the edges
                        if len(chance) > 0:
                            # Select the star with the lowest chance coincidence
                            print(cat_coords_pix[i])
                            best_match = np.argmin(chance)

                            # Get coordinates of star to match
                            pixel_centroid = np.array([sources[best_match]['xcentroid'], sources[best_match]['ycentroid']])

                            # Define the aperture for that star that's 2.5 X the FWHM value
                            single_aperture = CircularAperture(pixel_centroid, r=(best_fwhm/2.0)*2.5)

                            # Number of pixels in the aperture area
                            try:
                                n_pix = single_aperture.area()
                            except:
                                n_pix = single_aperture.area

                            # Perform aperture photometry to obtain sum of counts in aperture
                            do_phot = aperture_photometry(cropped_data, single_aperture)

                            # Calculate the noise based on https://arxiv.org/pdf/1701.04817v1.pdf
                            # Calculate the signal (Signal - Background) in ADU
                            sky_background = median_local*n_pix
                            signal         = (do_phot['aperture_sum'] - sky_background)

                            # Sigma_f is the standard deviation of the fractional count lost to digitization in a single pixel
                            sigma_f = 0.289
                            # Calculate the noise in ADU
                            # Not accounting for dark counts
                            noise = np.sqrt(signal + sky_background + n_pix * (parameters_list['rdnoise'] / parameters_list['gain']) ** 2 + sigma_f ** 2)

                            # Calculate signal to noise
                            signal_to_noise = np.array(signal / noise)[0]

                            #single_aperture.plot(color = 'g')
                            #plt.imshow(cropped_data, vmin = median_local-3.0*std_local, vmax = median_local+3.0*std_local, cmap='Greys', origin='lower',interpolation='none')
                            #plt.title(signal_to_noise)
                            #plt.xlabel(i)
                            #plt.show()

                            # Re-offset to the correct image position.
                            source_xcoord = sources[best_match]['xcentroid'] + cat_star[0] - pix_range
                            source_ycoord = sources[best_match]['ycentroid'] + cat_star[1] - pix_range
                            print(signal_to_noise)
                            # If the SNR is within the threshold
                            if signal_to_noise > initial_snr_cut:
                                print('     Source has signal')
                                # If it's the first star in the list
                                if first == 0:
                                    data = np.array([cat_in[all_stars][selected][i]['RA'], cat_in[all_stars][selected][i]['DEC'], cat_in[all_stars][selected][i]['gmag'], cat_in[all_stars][selected][i]['rmag'], cat_in[all_stars][selected][i]['imag'], cat_in[all_stars][selected][i]['zmag'], cat_in[all_stars][selected][i]['ymag'], source_xcoord, source_ycoord, sources[best_match]['flux'], sources[best_match]['mag'], signal_to_noise, cat_star[0], cat_star[1]])
                                    cat_matched = table.Table(data = data, names=('RA', 'DEC', 'gmag', 'rmag', 'imag', 'zmag', 'ymag', 'xcentroid', 'ycentroid', 'flux', 'mag', 'SNR','cat_x', 'cat_y'), dtype = ('f', 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'f'))
                                    first = 1

                                # If it's not the first star in the list
                                else:
                                    data = np.array([cat_in[all_stars][selected][i]['RA'], cat_in[all_stars][selected][i]['DEC'], cat_in[all_stars][selected][i]['gmag'], cat_in[all_stars][selected][i]['rmag'], cat_in[all_stars][selected][i]['imag'], cat_in[all_stars][selected][i]['zmag'], cat_in[all_stars][selected][i]['ymag'], source_xcoord, source_ycoord, sources[best_match]['flux'], sources[best_match]['mag'], signal_to_noise, cat_star[0], cat_star[1]])
                                    cat_matched.add_row(data)
        return cat_matched

    try:
        cat_matched = get_catalog(initial_snr_cut)
    except:
        print('Cutting minimum SNR')
        cat_matched = get_catalog(initial_snr_cut / 2)

    # Filter out stars that don't match the general trend (mismatched or unresolved)
    distance_in       = np.sqrt((cat_matched['xcentroid'] - cat_matched['cat_x'])**2 + (cat_matched['ycentroid'] - cat_matched['cat_y'])**2)
    mag_difference_in = (cat_matched['mag'] - cat_matched[color + 'mag'])

    # Delete the two best data points
    if len(distance_in) > 20:
        to_delete      = distance_in.argsort()[:2]
        mag_difference = np.delete(mag_difference_in, to_delete)
        distance       = np.delete(distance_in, to_delete)
    # Unless the list is already small
    else:
        distance       = distance_in
        mag_difference = mag_difference_in

    # Measure average weighted by the stars that are closest to their host
    #average_difference_init = np.average(mag_difference, weights = 1/distance)
    #std_difference_init     = np.std(mag_difference)

    # Stars that make the cut
    #good_one = np.where((mag_difference < average_difference_init + std_difference_init * 3) & (mag_difference > average_difference_init - std_difference_init * 3))

    # Measure the average again after sigma clipping
    #average_difference = np.average(mag_difference[good_one], weights = 1/distance[good_one])
    #std_difference     = np.std(mag_difference[good_one])

    # Stars that make the cut
    average_difference, median_difference, std_difference = sigma_clipped_stats_version(mag_difference, sigma_low=initial_sigma_clip, sigma_hi=initial_sigma_clip, iterations=3)
    good = np.where((mag_difference < average_difference + std_difference * initial_sigma_clip) & (mag_difference > average_difference - std_difference * initial_sigma_clip))

    # Plot Distance vs. Difference
    plt.subplot('331')
    plt.scatter(distance, mag_difference, color = 'r', s = 10)
    plt.scatter(distance[good], mag_difference[good], color = 'b', s = 12)
    plt.xlabel('Distance to Center')
    plt.ylabel('Magnitude Difference')
    plt.ylim(min(mag_difference[good])-3*std_difference, max(mag_difference[good])+3*std_difference)
    plt.axhline(y = average_difference, color = 'k')
    plt.axhline(y = average_difference + std_difference * initial_sigma_clip, color = 'k', linestyle = '--')
    plt.axhline(y = average_difference - std_difference * initial_sigma_clip, color = 'k', linestyle = '--')

    # Create apertures for image stars (not catalogue stars)
    star_list     = np.array([cat_matched['xcentroid'][good], cat_matched['ycentroid'][good]])
    apertures_img = CircularAperture(star_list, r=best_fwhm)

    # Create apertures for catalogue stars
    cat_list      = np.array([cat_matched['cat_x'][good], cat_matched['cat_y'][good]])
    apertures_cat = CircularAperture(cat_list, r=best_fwhm * 2.0)

    # Create apertue for the target star
    apertures_tar = CircularAperture(target_coords_img, r=best_fwhm * 2.0)

    # Save image with apertures
    plt.subplot('332')
    apertures_cat.plot(color='blue',  lw=0.2, alpha=0.3)
    apertures_img.plot(color='red',   lw=0.2, alpha=0.5)
    apertures_tar.plot(color='green', lw=0.7, alpha=0.4, linestyle = '--')
    plt.imshow(image_data, vmin = average_background-4.0*background_low, vmax = average_background+10.0*background_high, cmap='Greys', origin='lower',interpolation='none')

    # plot All stars
    all_coords_arr = np.transpose(np.array([cat_in['RA'].astype(float),cat_in['DEC'].astype(float)]))
    all_coords_pix = wcs_data.wcs_world2pix(all_coords_arr, 1)
    all_selected   = np.where((all_coords_pix.T[0] > pix_range) & (all_coords_pix.T[0] < datasize_x - pix_range) & (all_coords_pix.T[1] > pix_range) & (all_coords_pix.T[1] < datasize_y - pix_range))
    plt.scatter(all_coords_pix.T[0][all_selected], all_coords_pix.T[1][all_selected], marker = 'x', s = 150, alpha = 0.2, color = 'C1')
    plt.xlim(xmin = 0)
    plt.ylim(ymin = 0)

    # Save pixel coordinates to file
    np.savetxt(image_name[:-5] + '_coords.txt', np.transpose(star_list))

    # Return Catalogue of best stars
    good_cat = cat_matched[good]

    return good_cat

def check_existence(file_name, function, verbose = True):
    '''
    Check if some files with file_name already exist.
    If they exist return True, if they don't return False.
    Print the name of the function too.

    Parameters
    -------------
    file_name: name of the files to search.
    function: Name of the function, only for plotting purposes
    verbose: Print the name of the function?

    Output
    -------------
    True if the file already exists, False if it doesn't

    '''

    # Check that the files don't exist
    exists   = subprocess.Popen('ls ' + file_name, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = exists.communicate()

    # If files were returned:
    if out != b'':
        if verbose:
            print("%s -- %s already exists, skipping."%(function, file_name))
        return True
    else:
        return False

def query3pi(query,jobType):
    '''
    This program is meant as a client side example of querying a PSPS Database via Data Retrevial Layer DRL
    from a query in a file and writing the results to a file.
    Command line usage: python queryDRLClient.py query_file_name output_file_name [type: slow|fast]
    Example: python queryDRLClient.py myquery.sql output.csv fast
    '''

    # URLs to the PSPS SOAP WSDLs for authentication and queries.
    authWsdlUrl = "http://panstarrs.stsci.edu/DFetch/WSDL/AuthService.wsdl.php"
    jobsWsdlUrl = "http://panstarrs.stsci.edu/DFetch/WSDL/JobsService.wsdl.php"

    # Credentials
    schemaGroup = "PS1_SCHEMA"
    schema      = "PanSTARRS_3PI_PV3.1"

    # Get the PS1 MAST username and password from /Users/username/3PI_key.txt
    key_location = os.path.join(pathlib.Path.home(), '3PI_key.txt')
    user, password = np.genfromtxt(key_location, dtype = 'str')

    python_version = sys.version_info[0]

    if python_version == 3:
        from zeep import Client
        from pysimplesoap.client import SoapClient
        authClient = Client(authWsdlUrl)
        jobsClient = SoapClient(wsdl=jobsWsdlUrl, trace = False)

        sessionID =  authClient.service.login(user, password)
        task      = "Executing " + jobType + " query from client python script"

        # SOAP call to execute query
        if  jobType.lower() == 'fast':
            #executeQuickJob(xs:string sessionID, xs:string schemaGroup, xs:string query, xs:string context, xs:string taskname, xs:boolean isSystem, )
            queryResults = jobsClient.executeQuickJob( sessionID, schemaGroup, query, schema, task, False)
        elif  jobType.lower() == 'slow':
            #submitJob(xs:string sessionID, xs:string schemaGroup, xs:string query, xs:string context, xs:string taskname, xs:int TimeEstimate, )
            queryResults = 'Job ID:' + str(jobsClient.service.submitJob( sessionID, schemaGroup, query, schema, task, 10))
        else:
            print ("Error: unkown job type: " + jobType + ". Must be either fast or slow.")
        return queryResults['return']

    elif python_version == 2:
        from suds.client import Client
        logging.getLogger('suds.client').setLevel(logging.DEBUG)
        authClient = Client(authWsdlUrl)
        jobsClient = Client(jobsWsdlUrl)

        sessionID =  authClient.service.login(user, password)
        task      = "Executing " + jobType + " query from client python script"

        # SOAP call to execute query
        if  jobType.lower() == 'fast':
            #executeQuickJob(xs:string sessionID, xs:string schemaGroup, xs:string query, xs:string context, xs:string taskname, xs:boolean isSystem, )
            queryResults = jobsClient.service.executeQuickJob( sessionID, schemaGroup, query, schema, task, False)
        elif  jobType.lower() == 'slow':
            #submitJob(xs:string sessionID, xs:string schemaGroup, xs:string query, xs:string context, xs:string taskname, xs:int TimeEstimate, )
            queryResults = 'Job ID:' + str(jobsClient.service.submitJob( sessionID, schemaGroup, query, schema, task, 30))
        else:
            print ("Error: unkown job type: " + jobType + ". Must be either fast or slow.")
        return queryResults

def query3pi_mast(RA_deg, DEC_deg, search_radius, wsid = 1932612232, password = 'B127HAHA39'):
    '''
    This program is meant as a client side example of querying a PSPS Database 
    via Data Retrevial Layer DRL from a query in a file and writing the results to a file.
    The function will only return objects with at least one detection

    # The list of parameters you can query is in: 
    https://outerspace.stsci.edu/display/PANSTARRS/PS1+StackObjectAttributes+table+fields
    The default jobType is fast, slow is the other option.

    Parameters
    ---------------
    RA_deg, DEC_deg : Coordinates of the object in degrees.
    search_radius   : Search radius in arcminutes
    user            : WSID from the MAST query database
    password        : password for MAST query database

    Returns
    -------------
    Table with data outlined in the query_3pi variable below

    '''

    # 3PI query
    query_3pi = """select o.objID,            -- Object ID
                          o.raStack,          -- Right Ascension in degrees
                          o.decStack,         -- Declination in degress
                          o.nDetections,      -- Number of detections for this object
                          m.gKronMag,m.rKronMag,m.iKronMag,m.zKronMag,m.yKronMag,      -- Kron Magnitude in grizy
                          m.gPSFMag,m.rPSFMag,m.iPSFMag,m.zPSFMag,m.yPSFMag,           -- PSF Magnitude in grizy
                          m.gPSFMagErr,m.rPSFMagErr,m.iPSFMagErr,m.zPSFMagErr,m.yPSFMagErr,           -- PSF Magnitude Error in grizy
                          b.gKronRad,b.rKronRad,b.iKronRad,b.zKronRad,b.yKronRad,      -- Kron Radius in grizy [Arcsec]
                          nb.distance,                                                 -- Separation from querry center
                          m.primaryDetection                                           -- Is this the primary detection?
                    from fGetNearbyObjEq(%s, %s, %s) nb
                    inner join ObjectThin o on o.objid=nb.objid
                    inner join StackObjectThin m on o.objid=m.objid
                    inner join StackObjectAttributes b on o.objid=b.objid
                    where m.primaryDetection = 1
                """
    query = query_3pi%(RA_deg, DEC_deg, search_radius)

    # Query
    jobs    = mastcasjobs.MastCasJobs(userid=wsid, password=password, context="PanSTARRS_DR1")
    results = jobs.quick(query, task_name="python cone search")
  
    return results

def get3pimags(coord, cat_name, search_radius = 7.0):
    '''
    Querry 3pi data around the a given region and return the list of found stars.

    Parameters
    ----------------
    coord               : Coordinates in SkyCoord format
    cat_name            : Name of the catalogue to save
    search_radius       : Search radius in arcminutes
    '''

    # Querry the 3pi survey within some search radius, making sure the object has a minimum number of detections
    print("Querying 3PI ...")

    # Login to 3pi and extract the data
    data = query3pi_mast(coord.ra.deg, coord.dec.deg, search_radius)

    # If objects were found split each found target into rows
    # and format it in column x rows array
    try:
        table_rows = data.split()
        new_rows   = []
        for row in table_rows: new_rows.append(row.split(','))

        # Create table of stars
        if len(new_rows) <= 1:
            return '--'
        else:
            cat = table.Table(rows=new_rows[1:],names=['ID','RA','DEC','N_det','gkron','rkron','ikron','zkron','ykron','gmag','rmag','imag','zmag','ymag','gmagerr','rmagerr','imagerr','zmagerr','ymagerr','grad','rrad','irad','zrad','yrad','distance','pridet'])
    except:
        cat = table.Table(data, names=['ID','RA','DEC','N_det','gkron','rkron','ikron','zkron','ykron','gmag','rmag','imag','zmag','ymag','gmagerr','rmagerr','imagerr','zmagerr','ymagerr','grad','rrad','irad','zrad','yrad','distance','pridet'])

    cat_in = table.unique(cat, keys='ID')
    cat_in.write(cat_name, format='ascii', overwrite=True)

    return cat_in

def crop_3picatalog(cat_read, kron_psf_difference = 0.1):
    '''
    Querry 3pi data around the a given region and return the list of found stars.

    Parameters
    ----------------
    cat_read            : Input catalogue
    kron_psf_difference : Only return stars lower than this value
                          to remove the galaxies.
    '''

    # Select stars with more than 2 detections and Kron magnitude simliar to a star
    N_det = np.array(cat_read['N_det']).astype(float)

    # Get the psf and Kron magnitudes
    psf_g, kron_g = get_psf_kron(cat_read, 'g')
    psf_r, kron_r = get_psf_kron(cat_read, 'r')
    psf_i, kron_i = get_psf_kron(cat_read, 'i')
    psf_z, kron_z = get_psf_kron(cat_read, 'z')
    psf_y, kron_y = get_psf_kron(cat_read, 'y')

    # Get average magnitudes and difference for each star
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        average_psf  = np.nanmean([psf_g , psf_r , psf_i , psf_z , psf_y ], axis = 0)
        average_kron = np.nanmean([kron_g, kron_r, kron_i, kron_z, kron_y], axis = 0)
        difference   = average_psf - average_kron

        # Return only the good stars
        good = np.where((N_det > 1) & (difference < kron_psf_difference))

    cat_in = table.unique(cat_read[good], keys='ID')

    return cat_in

def estimate_fwhm(image_data, cat_in, wcs_data, coord, parameters_list, min_mag = 21, max_mag = 16, background_stars = 5, pix_range = 20, overexposed_limit = 50000, fwhm_guess = 3.5, sigma_l_fwhm = 2.0, sigma_h_fwhm = 2.0, iter_fwhm = 2, sigma_l_back = 2.0, sigma_h_back = 2.0, iter_back = 2):
    '''
    Estimate the FWHM, Background and Std of the stars in the image by fitting the region around
    a few stars using photutils. This value will be fed into IRAF.

    Parameters
    ---------------
    image_data        : Numpy array of data
    cat_in            : Catalogue file with stars coordiantes
    wcs_data          : Output form loadimage() with wcs data
    coord             : Coordiantes in SkyCoord format form loadimage()
    parameters_list   : List of relevant parameters form loadimage()
    min_mag           : Dimmest magnitude allowed to be matched in the process
    max_mag           : Birghtest stars allowed to be matched in the process
    background_stars  : Number of background stars to use
    pix_range         : Search radius for the corresponding star in the image from
                        the catalogue.
    overexposed_limit : Maximum number of counts before discarding the star
    fwhm_guess        : Initial guess for the FWHM value
    sigma_l_fwhm      : Low Sigma clipping for FWHM
    sigma_h_fwhm      : High Sigma clipping for FWHM
    iter_fwhm         : Number of iterations for FWHM sigma clipping
    sigma_l_back      : Low Sigma clipping for background
    sigma_h_back      : High Sigma clipping for background
    iter_back         : Number of iterations for background sigma clipping

    Output
    ---------------
    best_fwhm          : Average FWHM of stars
    average_background : Average value of backgrounf
    average_std_high   : Upper sigma for background
    average_std_low    : Lower sigma for background
    just_nearby        : Pixels all around nearby stars

    '''

    # Sort catalogue
    color = parameters_list['filter']
    cat_in.sort(color + 'mag')
    
    ## Select the stars that will go into the FWHM calculation
    # Get the magniudes in the right color and in float format
    catalogue_magnitudes = cat_in[color + 'mag'].astype(float)

    # Select the stars that will be used to estimate the FWHM
    all_stars = np.where((catalogue_magnitudes < min_mag) & (catalogue_magnitudes > max_mag))[0]

    # Get pixel positions of the catalogue stars in the image.
    cat_coords_arr    = np.transpose(np.array([cat_in['RA'][all_stars].astype(float),cat_in['DEC'][all_stars].astype(float)]))
    cat_coords_pix    = wcs_data.wcs_world2pix(cat_coords_arr, 1)
    target_coords_img = wcs_data.wcs_world2pix(coord.ra.deg, coord.dec.deg, 1)

    # selected stars that are inside the image
    datasize_y = image_data.shape[0]
    datasize_x = image_data.shape[1]
    selected   = np.where((cat_coords_pix.T[0] > pix_range) & (cat_coords_pix.T[0] < datasize_x - pix_range) & (cat_coords_pix.T[1] > pix_range) & (cat_coords_pix.T[1] < datasize_y - pix_range))

    # Stars to be used
    selected_coordinates = cat_coords_pix[selected]

    # Select the number of stars to calculate the background
    distance = np.sqrt((selected_coordinates.T[0] - target_coords_img[0])**2 + (selected_coordinates.T[1] - target_coords_img[1])**2)
    closest  = distance.argsort()[:background_stars]

    # Empty list for future list
    FWHM_list       = np.array([]) # List of stars FWHM
    Background_list = np.array([]) # List of average background values
    sigma_list      = np.array([]) # List of all background values
    just_nearby     = np.array([]) # List of background near closest stars

    # Calculate the FWHM for each star
    for i in range(len(selected_coordinates)):
        print(i)
        cat_star = selected_coordinates[i]

        # Select a small area of pix_range pixels around the catalogue star
        if np.isfinite(cat_star[0]):
            xmin = int(np.around(cat_star[0]-pix_range))
            xmax = int(np.around(cat_star[0]+pix_range))
            ymin = int(np.around(cat_star[1]-pix_range))
            ymax = int(np.around(cat_star[1]+pix_range))

            # Crop the data
            cropped_data = image_data[ymin:ymax,xmin:xmax]

            # Make sure there is data in the cropped region
            if len(cropped_data.flatten()) == 0:
                print('No Data')
                continue

            # Maximum value in the cropped image
            max_counts = np.max(cropped_data)

            # If the star is too close to the edge, skip it
            if cropped_data.shape != (pix_range*2, pix_range*2):
                print('Selected star too close to the edge, picking a different one')
                continue

            # If the star is too bright, skip it
            if max_counts > overexposed_limit:
                print('Selected star is overexposed, picking a different one')
                continue

            # Calculate sigma-clipped statistics on the specified cropped data
            mean, median, std = sigma_clipped_stats_version(cropped_data, sigma_low=2.0, sigma_hi=1.0, iterations=5)

            # Create model with initial guess
            star_model  = models.Gaussian2D(amplitude=max_counts-mean, x_mean=pix_range, y_mean=pix_range, x_stddev=fwhm_guess, y_stddev=fwhm_guess)
            back_model  = models.Const2D(amplitude=mean)
            total_model = star_model + back_model

            # Create grid for modeling
            y_grid, x_grid = np.mgrid[:pix_range*2, :pix_range*2]

            # Fit the data
            fitter  = fitting.LevMarLSQFitter()
            bestfit = fitter(total_model, x_grid, y_grid, cropped_data)

            # Calculate the sigma and return the FWHM
            sigma           = np.mean([bestfit.x_stddev_0.value, bestfit.y_stddev_0.value])
            FWHM            = 2 * np.sqrt(2 * np.log(2)) * sigma
            FWHM_list       = np.append(FWHM_list, FWHM)
            Background_list = np.append(Background_list, bestfit.amplitude_1.value)

            # Substract the model from the data to get an example of the background
            background_only = np.array(cropped_data - bestfit(x_grid,y_grid) + bestfit.amplitude_1.value).flatten()
            sigma_list      = np.append(sigma_list, background_only)

            # Test Fit Result
            #best_model = bestfit(x_grid, y_grid)
            #plt.subplot(131)
            #plt.imshow(cropped_data, vmin = np.average(background_only) - np.std(background_only) * 3, vmax = np.average(background_only) + np.std(background_only) * 3, cmap='Greys', origin='lower',interpolation='none')
            #plt.subplot(132)
            #plt.imshow(best_model, vmin = np.average(background_only) - np.std(background_only) * 3, vmax = np.average(background_only) + np.std(background_only) * 3, cmap='Greys', origin='lower',interpolation='none')
            #plt.subplot(133)
            #plt.imshow(cropped_data - best_model, vmin = 0 - np.std(background_only) * 3, vmax = 0 + np.std(background_only) * 3, cmap='Greys', origin='lower',interpolation='none')
            #plt.show()

            # Append the background value from only the nearby sources
            if i in closest:
                just_nearby = np.append(just_nearby, background_only)

    # Calculate the best FWHM, from reasonable values only
    good_fwhm = [np.where((FWHM_list > 0.5) & (FWHM_list < 12))][0]
    FWHM_good = FWHM_list[good_fwhm]

    # Calculate the weights
    magniude_values   = catalogue_magnitudes[all_stars][selected][good_fwhm]
    magnitude_errors  = cat_in[color + 'magerr'].astype(float)[all_stars][selected][good_fwhm]

    luminosity_values = 10 ** (-0.4 * (magniude_values - 26.74))
    luminosity_errors = np.abs(-0.4 * np.log(10) * 10 ** (-0.4 * (magniude_values - 26.74)) * magnitude_errors)

    # Calculate the average fwhm
    best_fwhm, good_fwhm = weighted_average(FWHM_good, 1 / magnitude_errors, sigma_lower = sigma_l_fwhm, sigma_upper = sigma_h_fwhm, iterations = int(iter_fwhm))

    # Calculate the average background
    average_background, good_back = weighted_average(Background_list, np.ones(len(Background_list)), sigma_lower = sigma_l_back, sigma_upper = sigma_h_back, iterations = int(iter_back))

    # Calculate the standard deviation
    average_std = np.std(Background_list[good_back])

    # Do a rough cut of 7 sigma to the quick calculation of background
    good_background = np.where((sigma_list > average_background - 7 * average_std) & (sigma_list < average_background + 7 * average_std))

    # Calculate the 1 sigma percentiles of the background
    sigma_good = np.array([sigma_list[good_background],sigma_list[good_background]]).T
    background_mcmc, background_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(sigma_good, [15.87, 50, 84.13], axis=0)))

    # Background Parameters
    average_background = background_mcmc[0]
    background_high    = background_mcmc[1]
    background_low     = background_mcmc[2]

    return best_fwhm, average_background, background_high, background_low, just_nearby

def weighted_average(values, weights, sigma_lower, sigma_upper, iterations):
    '''
    Given a distribution of values, do a sigma clipping on the average to
    purge the outliers. Then only select the values that are within sigma of
    the median. Then calculate a weighted average based on the provided weights.

    Parameters
    ---------------
    values: List of values as a numpy array to calculate the average on
    weights: Ordered weights of these values
    sigma_lower: sigma clipping for low values
    sigma_upper: sigma clipping for high values
    iterations: Number of iterations of the sigma clipping.

    Output
    ---------------
    - Weighted verage of the input values.
    - List of used values.
    '''

    # Do the sigma clipping to purge outliers
    mean, median, std = sigma_clipped_stats_version(values, sigma_low=sigma_lower, sigma_hi=sigma_upper, iterations=iterations)
    # Reject the points that do not make the sigma cuts
    good = np.where((values > median - sigma_lower * std) & (values < median + sigma_lower * std))

    # If there are no points, make the sigma larger
    repeats = 0
    while (len(good[0]) == 0) and (repeats < 15):
        print("Good")
        sigma_lower += sigma_lower + 0.5
        sigma_upper += sigma_upper + 0.5
        good = np.where((values > median - sigma_lower * std) & (values < median + sigma_lower * std))
        repeats += 1

    # Get the weighted average
    average = np.average(values[good], weights = weights[good])

    return average, good

def calculate_coincidence(separation, magnitude):
    '''
    Calculate the chance that a galaxy of magnitude M to fall
    within a separation R of a transient. The galaxies with the lowest
    chance probability will be selected as the best candidate hosts.

    Parameters
    ---------------
    separation: Separation between the host and transient [Arcseconds]
    Magnitude: Magnitude of the galaxy

    Output
    ---------------
    P_cc = Probability of chance coincidence
    '''

    # Observed number density of galaxies brighter than magnitude M (From Berger 2010)
    sigma = 10 ** (0.33 * (magnitude - 24) - 2.44) / (0.33 * np.log(10))

    # Probability of chance coincidence
    chance_coincidence = 1 - np.exp(-np.pi * (4 * separation) ** 2.0 * sigma)

    return chance_coincidence

def daophot_parameters(parameters_list, n_iterations = 10, best_fwhm = 3.5, pix_range = 20, varorder = 0, sky_annulus = 8.0, recenter = 'yes', centering_fwhm = 3.0, zmag = 30.0, overexposed_limit = 50000):
    '''
    Function that serves only to define the daophot parameters inside IRAF.

    Parameters
    ---------------
    parameters_list    : List of relevant parameters form loadimage()
    n_iterations       : Maximum Number of sky fitting iterations
    best_fwhm          : FWHM value of stars as determined by estimate_fwhm() in inputs of pixels.
    pix_range          : Search radius for the stars in the catalogue.
    varorder           : 1 will have 3 look up tables, 2 will have 6, etc. Set to -1 to 2
    sky_annulus        : Width of sky fitting annulus
    recenter           : Recenter PSF fitting model
    centering_fwhm     : Prefactor for the box of the centering algortithm.
    zmag               : Zero point of magnitude scale
    overexposed_limit  : Maximum number of counts before discarding the star

    Output
    ---------------
    Doesn't return anything, but sets the parameters for IRAF.

    '''

    # Edit the data dependent parameters
    daophot.datapars.scale        = 1.0                             # Image scale in units per pixel
    daophot.datapars.fwhmpsf      = best_fwhm                       # FWHM of the PSF in scale units
    daophot.datapars.emission     = 'yes'                           # Features are positive?
    daophot.datapars.sigma        = 'INDEF'                         # Standard deviation of background in counts
    daophot.datapars.datamin      = 'INDEF'                         # Minimum good data value
    daophot.datapars.datamax      =  overexposed_limit              # Maximum good data value
    daophot.datapars.noise        = 'poisson'                       # Noise model
    daophot.datapars.ccdread      = 'RDNOISE'                       # CCD readout noise image header keyword
    daophot.datapars.gain         = 'GAIN'                          # CCD gain image header keyword
    daophot.datapars.readnoise    = str(parameters_list['rdnoise']) # CCD readout noise in electrons
    daophot.datapars.epadu        = str(parameters_list['gain'])    # Gain in electrons per count
    daophot.datapars.exposure     = 'EXPTIME'                       # Exposure time image header keyword
    daophot.datapars.airmass      = 'AIR'                           # Airmass image header keyword
    daophot.datapars.filter       = 'FILTER'                        # Filter image header keyword
    daophot.datapars.obstime      = 'MJD'                           # Time of observation image header keyword
    daophot.datapars.itime        = 1.0                             # Exposure time
    daophot.datapars.xairmass     = parameters_list['airmass']      # Airmass
    daophot.datapars.filter       = ''                              # Filter
    daophot.datapars.otime        = ''                              # Time of Observation
    daophot.datapars.mode         = 'ql'                            # IRAF mode

    # Edit the centering algorithm parameters
    if recenter == 'yes':
        daophot.centerpars.calgorithm = 'centroid'
    elif recenter == 'no':
        daophot.centerpars.calgorithm = 'none'
    daophot.centerpars.cbox       = best_fwhm*centering_fwhm        # Centering box width in scale units
                                                                    # Reasonable values are 2.5-4.0 * FWHM of the PSF
    daophot.centerpars.cthreshold = 0.0                             # Centering threshold in sigma above background
    daophot.centerpars.minsnratio = 1.0                             # Minimum signal to noise ratio for centering algorithm
    daophot.centerpars.cmaxiter   = pix_range                       # Maximum iterations for centering algorithm
    daophot.centerpars.maxshift   = 0.5                             # Maximum center shift in scale units
    daophot.centerpars.clean      = 'no'                            # Symmetry clean before centering?
    daophot.centerpars.rclean     = 1.0                             # Cleaning radius in scale units
    daophot.centerpars.rclip      = 2.0                             # Clipping radius in scale units
    daophot.centerpars.kclean     = 3.0                             # K-sigma rejection criterion in skysigma
    daophot.centerpars.mkcenter   = 'no'                            # Mark the computed center?
    daophot.centerpars.mode       = 'ql'                            # IRAF mode

    # Edit the sky fitting algorithm parameters
    daophot.fitskypars.salgorithm = 'ofilter'                       # Sky fitting alforithm
    daophot.fitskypars.annulus    = 4*best_fwhm/2                   # Inner radius of sky annulus in scale units
    daophot.fitskypars.dannulus   = sky_annulus                     # Width of sky annulus in scale units
    daophot.fitskypars.skyvalue   = 0.0                             # User sky value
    daophot.fitskypars.smaxiter   = n_iterations                    # Maximum number of sky fitting iterations
    daophot.fitskypars.sloclip    = 0.0                             # Lower clipping factor in percent
    daophot.fitskypars.shiclip    = 0.0                             # Upper clipping factor in percent
    daophot.fitskypars.snreject   = 30                              # Maximum number of sky fitting rejection iterations
    daophot.fitskypars.sloreject  = 2.0                             # Lower K-sigma rejection limit in sky sigma
    daophot.fitskypars.shireject  = 2.0                             # Upper K-sigma rejection limit in sky sigma
    daophot.fitskypars.khist      = 3.0                             # Half width of histogram in sky sigma
    daophot.fitskypars.binsize    = 0.1                             # Binsize of histogram in sky sigma
    daophot.fitskypars.smooth     = 'no'                            # Boxcar smooth the histogram?
    daophot.fitskypars.rgrow      = 0.0                             # Region growing radius in scale units
    daophot.fitskypars.mksky      = 'no'                            # Mark asky annuli on the display?
    daophot.fitskypars.mode       = 'ql'                            # IRAF mode

    # Edit the photometry parameters
    daophot.photpars.weighting    = 'constant'                      # Photometric weighting scheme
    daophot.photpars.apertures    = (best_fwhm/2.0)*2.5             # List of aperture radii in scale units
    daophot.photpars.zmag         =  zmag                           # Zero point of magnitude scale
    daophot.photpars.mkapert      = 'no'                            # Draw apertures on the display?
    daophot.photpars.mode         = 'ql'                            # IRAF mode

    # Edit the daophot fitting parameters
    daophot.pstselect.daopars.function      = 'gaussian'            # Form of analytic component of psf model
    daophot.pstselect.daopars.varorder      = varorder              # Order of empirical component of psf model
    daophot.pstselect.daopars.nclean        = 1                     # Number of cleaning iterations for computing psf model
    daophot.pstselect.daopars.saturated     = 'no'                  # Use wings of saturated stars in psf model computation?
    daophot.pstselect.daopars.matchrad      = 3                     # Object matching radius in scale units
    daophot.pstselect.daopars.psfrad        = best_fwhm*4+1         # Radius of psf model in scale units
    daophot.pstselect.daopars.fitrad        = best_fwhm/2.0         # Fitting radius in scale units
    daophot.pstselect.daopars.recenter      = recenter              # Recent stars during fit?
    daophot.pstselect.daopars.fitsky        = 'yes'                 # Recompute group sky value during fit?
    daophot.pstselect.daopars.groupsky      = 'yes'                 # Use group rather than individual sky values?
    daophot.pstselect.daopars.sannulus      = best_fwhm*2.5         # Inner radius of sky fitting annulus in scale units
    daophot.pstselect.daopars.wsannulus     = sky_annulus           # Width of sky fitting annulus in scale units
    daophot.pstselect.daopars.flaterr       = 0.75                  # Flat field error in percent
    daophot.pstselect.daopars.proferr       = 5.0                   # Profile error in percent
    daophot.pstselect.daopars.maxiter       = 50                    # Maximum number of fitting iterations
    daophot.pstselect.daopars.clipexp       = 6                     # Bad data clipping exponent
    daophot.pstselect.daopars.cliprange     = 2.0                   # Bad data clipping range in sigma
    daophot.pstselect.daopars.mergerad      = 'INDEF'               # Critical object merging radius in scale units
    daophot.pstselect.daopars.critsnratio   = 1.0                   # Critical S/N ratio for group membership
    daophot.pstselect.daopars.maxnstar      = 10000                 # Maximum number off stars to fit
    daophot.pstselect.daopars.maxgroup      = 60                    # Maximum number of stars to fit per group
    daophot.pstselect.daopars.mode          = 'ql'                  # IRAF mode

def psf_photometry_iraf(image_name, image_data, average_background, background_high, background_low, parameters_list, n_iterations = 10, best_fwhm = 3.5, pix_range = 20, varorder = 0, sky_annulus = 8.0, recenter = 'yes', centering_fwhm = 3.0, zmag = 30.0, overexposed_limit = 50000, max_psf_stars = 15, delete_star = '', image_frame = ''):
    '''
    Do PSF photometry on the image with image_name with the coordinate file
    that has name image_name_coords.txt.

    Parameters
    ---------------
    image_name         : Name of the image to process
    image_data         : Numpy array of data
    average_background : Average background value of stars as determined by estimate_fwhm()
    background_high    : Upper std sigma value of the background
    background_low     : Lower std sigma value of the background
    parameters_list    : List of relevant parameters form loadimage()
    n_iterations       : Maximum Number of sky fitting iterations
    best_fwhm          : FWHM value of stars as determined by estimate_fwhm() in inputs of pixels.
    pix_range          : Search radius for the stars in the catalogue.
    varorder           : 1 will have 3 look up tables, 2 will have 6, etc. Set to -1 to 2
    sky_annulus        : Width of sky fitting annulus
    recenter           : Recenter PSF fitting model
    centering_fwhm     : Prefactor for the box of the centering algortithm.
    zmag               : Zero point of magnitude scale
    overexposed_limit  : Maximum number of counts before discarding the star
    max_psf_stars      : Number of stars to fit the PSF on
    delete_star        : Which star to remove from the PSF list? Comma separated values
    image_frame        : Fits frame, i.e. '[0]'

    Output
    ---------------
    Prints the best aperture and returns the magnitudes
    and errors for that aperture.
    '''

    # Check that the files don't exist
    exists1 = check_existence(image_name[:-5] + '.psf.1.fits',   'psf_phot')
    if exists1:
        subprocess.call('rm ' + image_name[:-5] + '.psf.1.fits', shell=True)

    exists2 = check_existence(image_name[:-5] + '.sub.1.fits',   'psf_phot')
    if exists2:
        subprocess.call('rm ' + image_name[:-5] + '.sub.1.fits', shell=True)

    exists3 = check_existence(image_name[:-5] + '.PSF_OUT.fits', 'psf_phot')
    if exists3:
        subprocess.call('rm ' + image_name[:-5] + '.PSF_OUT.fits', shell=True)

    exists4 = check_existence(image_name[:-5] + '.sub.2.fits',   'psf_phot')
    if exists4:
        subprocess.call('rm ' + image_name[:-5] + '.sub.2.fits', shell=True)

    exists5 = check_existence(image_name[:-5] + '.mag.out',   'psf_phot')
    if exists5:
        subprocess.call('rm ' + image_name[:-5] + '.mag.out', shell=True)

    exists6 = check_existence(image_name[:-5] + '.pst.1',   'psf_phot')
    if exists6:
        subprocess.call('rm ' + image_name[:-5] + '.pst.1', shell=True)

    exists7 = check_existence(image_name[:-5] + '.psg.1',   'psf_phot')
    if exists7:
        subprocess.call('rm ' + image_name[:-5] + '.psg.1', shell=True)

    exists8 = check_existence(image_name[:-5] + '.pst.2',   'psf_phot')
    if exists8:
        subprocess.call('rm ' + image_name[:-5] + '.pst.2', shell=True)

    exists9 = check_existence(image_name[:-5] + '.als.1',   'psf_phot')
    if exists9:
        subprocess.call('rm ' + image_name[:-5] + '.als.1', shell=True)

    exists10 = check_existence(image_name[:-5] + '.arj.1',   'psf_phot')
    if exists10:
        subprocess.call('rm ' + image_name[:-5] + '.arj.1', shell=True)

    # Set daophot parameters
    daophot_parameters(parameters_list, n_iterations, best_fwhm, pix_range, varorder, sky_annulus, recenter, centering_fwhm, zmag, overexposed_limit)

    # If delete_star is empty it means you don't want to remove any stars from the PSF
    if delete_star == '':
        # Run phot to get the aperture photometry magniutes of stars
        phot(image       = image_name + image_frame,                # The input image(s)
             skyfile     = '',                                      # The input sky file(s)
             coords      = image_name[:-5]+'_coords.txt',           # The input coordinate file(s) (default: image.coo.)
             output      = image_name[:-5]+'.mag.out',              # The output photometry file(s) (default: image.mag.)
             plotfile    = '',                                      # The output plots metacode file
             interactive = 'no',                                    # Interactive mode?
             radplots    = 'no',                                    # Plot the radial profiles in interactive mode?
             icommands   = '',                                      # Image cursor: [x y wcs] key [cmd]
             gcommands   = '',                                      # Graphics cursor: [x y wcs] key [cmd]
             wcsin       = ')_.wcsin',                              # Input coordinate system (logical, tv, physical, world)
             wcsout      = ')_.wcsout',                             # Output coordinate system (logical, tv, physical, world)
             cache       = 'no',                                    # Cache the input image pixels in memory?
             verify      = 'no',                                    # Verify critical parameters in non-interactive mode?
             update      = 'no',                                    # Update critical parameters in non-interactive mode?
             verbose     = 'no',                                    # Print messages in non-interactive mode?
             graphics    = ')_.graphics',                           # Graphics device
             display     = 'stdgraph',                              # Display device
             mode        = 'ql'                                     # IRAF mode
            )

        # Select candidate PSF stars from a photometry file
        daophot.pstselect(image       = image_name + image_frame,   # Image for which to build psf star list
                          photfile    = image_name[:-5]+'.mag.out', # Photometry file
                          pstfile     = image_name[:-5]+'.pst.1',   # Output psf star list file (Index, X, Y, Mag, Sky)
                          maxnpsf     = str(max_psf_stars),         # Maximum number of psf stars
                          mkstars     = 'no',                       # Mark deleted and accepted stars?
                          plotfile    = '',                         # Output plot metacode file
                          interactive = 'no',                       # Select psf stars interactively?
                          plottype    = 'mesh',                     # Default plot type (mesh, contour, radial)
                          icommands   = '',                         # Image cursor [x y wcs] key [cmd]
                          gcommands   = '',                         # Graphics cursor [x y wcs] key [cmd]
                          wcsin       = 'logical',                  # The input coordinate system
                          wcsout      = ')_.wcsout',                # The output coordinate system
                          cache       = 'no',                       # Cache the input image pixels in memory?
                          verify      = 'no',                       # Verify critical phot parameters?
                          update      = 'no',                       # Update critical phot parameters?
                          verbose     = 'yes',                      # Print phot messages?
                          graphics    = 'stdgraph',                 # Graphics device
                          display     = ')_.display',               # Display device
                          mode        = 'ql'                        # IRAF Mode
                          )

        # Name of output file
        out_number = 1

    # Otherwise remove thsoe stars when doing the PSF calculation
    else:

        # Run phot to get the aperture photometry magniutes of stars
        phot(image       = image_name,                              # The input image(s)
             skyfile     = '',                                      # The input sky file(s)
             coords      = image_name[:-5]+'_coords.txt',           # The input coordinate file(s) (default: image.coo.)
             output      = image_name[:-5]+'.mag.out',              # The output photometry file(s) (default: image.mag.)
             plotfile    = '',                                      # The output plots metacode file
             interactive = 'no',                                    # Interactive mode?
             radplots    = 'no',                                    # Plot the radial profiles in interactive mode?
             icommands   = '',                                      # Image cursor: [x y wcs] key [cmd]
             gcommands   = '',                                      # Graphics cursor: [x y wcs] key [cmd]
             wcsin       = ')_.wcsin',                              # Input coordinate system (logical, tv, physical, world)
             wcsout      = ')_.wcsout',                             # Output coordinate system (logical, tv, physical, world)
             cache       = 'no',                                    # Cache the input image pixels in memory?
             verify      = 'no',                                    # Verify critical parameters in non-interactive mode?
             update      = 'no',                                    # Update critical parameters in non-interactive mode?
             verbose     = 'no',                                    # Print messages in non-interactive mode?
             graphics    = ')_.graphics',                           # Graphics device
             display     = 'stdgraph',                              # Display device
             mode        = 'ql'                                     # IRAF mode
            )

        # Select candidate PSF stars from a photometry file
        daophot.pstselect(image       = image_name,                 # Image for which to build psf star list
                          photfile    = image_name[:-5]+'.mag.out', # Photometry file
                          pstfile     = image_name[:-5]+'.pst.1',   # Output psf star list file (Index, X, Y, Mag, Sky)
                          maxnpsf     = str(max_psf_stars),         # Maximum number of psf stars
                          mkstars     = 'no',                       # Mark deleted and accepted stars?
                          plotfile    = '',                         # Output plot metacode file
                          interactive = 'no',                       # Select psf stars interactively?
                          plottype    = 'mesh',                     # Default plot type (mesh, contour, radial)
                          icommands   = '',                         # Image cursor [x y wcs] key [cmd]
                          gcommands   = '',                         # Graphics cursor [x y wcs] key [cmd]
                          wcsin       = 'logical',                  # The input coordinate system
                          wcsout      = ')_.wcsout',                # The output coordinate system
                          cache       = 'no',                       # Cache the input image pixels in memory?
                          verify      = 'no',                       # Verify critical phot parameters?
                          update      = 'no',                       # Update critical phot parameters?
                          verbose     = 'yes',                      # Print phot messages?
                          graphics    = 'stdgraph',                 # Graphics device
                          display     = ')_.display',               # Display device
                          mode        = 'ql'                        # IRAF Mode
                          )

        # Select the stars that will be removed from the PSF calculation
        delete_star = str(delete_star).replace(' ', '')
        if delete_star[-1] == ',':
            delete_star = delete_star[:-1]
        to_delete   = [int(i) for i in delete_star.split(',')]

        # Create the command in IRAF format to delete them
        delete_command = 'id != %s'%to_delete[0]
        # If there's more than 1, append them
        if len(to_delete) > 1:
            for i in range(1, len(to_delete)):
                delete_command += ' && id != %s'%to_delete[i]

        daophot.pselect(infiles  = image_name[:-5]+'.pst.1', # Input apphot/daophot database(s)
                        outfiles = image_name[:-5]+'.pst.3', # Output apphot/daophot databases(s)
                        expr     = delete_command,           # Boolean expression for record selection
                        mode     = 'ql'                      # IRAF mode
                        )

        # Name of output file
        out_number = 3

    # Import Centers and ID's
    IDs        = np.array(iraf.txdump(textfile=image_name[:-5]+'.pst.%s'%out_number,fields='ID'     ,expr='yes',Stdout=1))
    x_position = np.array(iraf.txdump(textfile=image_name[:-5]+'.pst.%s'%out_number,fields='XCENTER',expr='yes',Stdout=1))
    y_position = np.array(iraf.txdump(textfile=image_name[:-5]+'.pst.%s'%out_number,fields='YCENTER',expr='yes',Stdout=1))
    positions  = np.array([x_position.astype(float), y_position.astype(float)])

    # Plot the image with PSF stars
    plt.subplot('333')
    psf_apertures = CircularAperture(positions, r=float(best_fwhm/2.0))
    psf_apertures.plot(color='red', lw=1.0, alpha=0.5)
    for i in range(len(IDs)):
        plt.annotate(IDs[i], xy = (positions[0][i],positions[1][i]), color = 'k', alpha = 0.5, path_effects=[PathEffects.withStroke(linewidth=3,foreground="w")])
    plt.imshow(image_data, vmin = average_background-4.0*background_low, vmax = average_background+10.0*background_high, cmap='Greys', origin='lower',interpolation='none')

    # Build the PSF for an image
    def daophot_psf(attempt):
        daophot.psf(image       = image_name + image_frame,      # Input image for which to build PSF
                    photfile    = image_name[:-5]+'.mag.out',    # Input photometry files
                    pstfile     = image_name[:-5]+'.pst.%s'%attempt, # Input psf star list (Index, X, Y, Mag, Sky)
                    psfimage    = image_name[:-5]+'.psf.1.fits', # Output PSF image
                    opstfile    = image_name[:-5]+'.pst.2',      # Output PSF star list
                    groupfile   = image_name[:-5]+'.psg.1',      # Output PSF star group file
                    plotfile    = '',                            # Output plot metacode file
                    matchbyid   = 'yes',                         # Match psf star list to photometry file by id number?
                    interactive = 'no',                          # Compute the psf interactively?
                    mkstars     = 'no',                          # Mark deleted and accepted psf stars?
                    showplots   = 'yes',                         # Show plots of PSF stars?
                    plottype    = 'mesh',                        # Default plot type (mesh, contour, radial)
                    icommands   = '',                            # Image cursor [x y wcs] key [cmd]
                    gcommands   = '',                            # Graphics cursor [x y wcs] key [cmd]
                    wcsin       = 'logical',                     # The input coordinate system
                    wcsout      = ')_.wcsout',                   # The output coordinate system
                    cache       = 'no',                          # Cache the input image pixels in memory?
                    verify      = 'no',                          # Verify critical phot parameters?
                    update      = 'no',                          # Update critical phot parameters?
                    verbose     = 'yes',                         # Print phot messages?
                    graphics    = 'stdgraph',                    # Graphics device
                    display     = ')_.display',                  # Display device
                    mode        = 'ql'                           # IRAF Mode
                    )

    # Run the function
    daophot_psf(attempt = out_number)

    # If the function failed, try again deleting the first four stars
    exists = check_existence(image_name[:-5]+'.psf.1.fits', 'psf_phot')
    if exists == False:
       daophot.pselect(infiles  = image_name[:-5]+'.pst.1',                   # Input apphot/daophot database(s)
                       outfiles = image_name[:-5]+'.pst.3',                   # Output apphot/daophot databases(s)
                       expr     = 'id != 1 && id != 2 && id != 3 && id != 4', # Boolean expression for record selection
                       mode     = 'ql'                                        # IRAF mode
                       )
       daophot_psf(attempt = 3)
       print("* Deleted first 4 stars and tried again.")

    # Convert a sampled PSF lookup table to a PSF image
    daophot.seepsf(psfimage  = image_name[:-5]+'.psf.1.fits',   # PSF image name
                   image     = image_name[:-5]+'.PSF_OUT.fits', # Output image name
                   dimension = 'INDEF',                         # Dimension of the output PSF image
                   xpsf      = 'INDEF',                         # X distance from the PSF star
                   ypsf      = 'INDEF',                         # Y distance from the PSF star
                   magnitude = 'INDEF',                         # Magnitude of the PSF star
                   mode      = 'ql'                             # IRAF mode
                   )

    # Group and fit PSF to multiple stars simultaneously
    daophot.allstar(image       = image_name + image_frame,      # Image corresponding to photometry
                    photfile    = image_name[:-5]+'.mag.out',    # Input photometry file
                    psfimage    = image_name[:-5]+'.psf.1.fits', # PSF image
                    allstarfile = image_name[:-5]+'.als.1',      # Output photometry file
                    rejfile     = image_name[:-5]+'.arj.1',      # Output rejections file
                    subimage    = image_name[:-5]+'.sub.1.fits', # Substracted image
                    wcsin       = ')_.wcsin',                    # Input coordiante system
                    wcsout      = ')_.wcsout',                   # Output coordinate system
                    wcspsf      = ')_.wcspsf',                   # PSF coordinate system
                    cache       = 'no',                          # Cache the input image pixels in memory?
                    verify      = 'no',                          # Verify critical phot parameters?
                    update      = 'no',                          # Update critical phot parameters?
                    verbose     = 'yes',                         # Print phot messages?
                    version     = '2',                           # Version
                    mode        = 'ql'                           # IRAF Mode
                    )

    # Import output data from IRAF
    IDs    = np.array(iraf.txdump(textfile=image_name[:-5]+'.als.1',fields='ID'  ,expr='yes',Stdout=1))
    rejIDs = np.array(iraf.txdump(textfile=image_name[:-5]+'.arj.1',fields='ID'  ,expr='yes',Stdout=1))
    mags   = np.array(iraf.txdump(textfile=image_name[:-5]+'.als.1',fields='MAG' ,expr='yes',Stdout=1))
    merrs  = np.array(iraf.txdump(textfile=image_name[:-5]+'.als.1',fields='MERR',expr='yes',Stdout=1))

    # Create a combined list of accepted + rejected stars
    IDs_full   = np.append(IDs  , rejIDs)
    mags_full  = np.append(mags , 99.0*np.ones(len(rejIDs)))
    merrs_full = np.append(merrs, 99.0*np.ones(len(rejIDs)))

    # Conver indefs to nan's
    mags_full[mags_full   == 'INDEF'] = 'nan'
    merrs_full[merrs_full == 'INDEF'] = 'nan'

    # Sort the list by ID
    mags_out  =  mags_full[np.argsort(IDs_full.astype(float))]
    merrs_out = merrs_full[np.argsort(IDs_full.astype(float))]

    # Convert to floats
    mags_out  = mags_out.astype(float)
    merrs_out = merrs_out.astype(float)

    # Get the starts that made the cut
    good_psf_mags   = np.where(mags_out < 90)
    psf_mags        = mags_out[good_psf_mags]
    psf_mags_err    = merrs_out[good_psf_mags]

    return psf_mags, psf_mags_err, good_psf_mags

def aperture_photometry_iraf(image_name, coordinate_suffix, parameters_list, n_iterations = 10, best_fwhm = 3.5, pix_range = 20, varorder = 0, sky_annulus = 8.0, recenter = 'yes', centering_fwhm = 3.0, zmag = 30.0, overexposed_limit = 50000, image_substracted_file = '', image_frame = ''):
    '''
    Do aperture photometry on the image with image_name with the coordinate file
    that has name image_name_coords.txt.

    Parameters
    ---------------
    image_name             : Name of the image to process
    coordinate_suffix      : Suffix for the file with the pixel coordinates in (x, y)
    parameters_list        : List of relevant parameters form loadimage()
    n_iterations           : Maximum Number of sky fitting iterations
    best_fwhm              : FWHM value of stars as determined by estimate_fwhm() in inputs of pixels.
    pix_range              : Search radius for the stars in the catalogue.
    varorder               : 1 will have 3 look up tables, 2 will have 6, etc. Set to -1 to 2
    sky_annulus            : Width of sky fitting annulus
    recenter               : Recenter PSF fitting model
    centering_fwhm         : Prefactor for the box of the centering algortithm.
    zmag                   : Zero point of magnitude scale
    overexposed_limit      : Maximum number of counts before discarding the star
    image_substracted_file : Specify a file that comes from an image subtraction routine.
                             To do photometry with the parameters of the main image but on this one.
    image_frame            : Fits frame, i.e. '[0]'

    Output
    ---------------
    Returns the magnitude and error from the aperture photometry, as well as the
    sum inside each aperture to be used in upper limit calculation

    '''

    # Use the alternative image if specified
    if image_substracted_file != '':
        image_name_in = image_substracted_file
    else:
        image_name_in = image_name

    # Set daophot parameters
    daophot_parameters(parameters_list, n_iterations, best_fwhm, pix_range, varorder, sky_annulus, recenter, centering_fwhm, zmag, overexposed_limit)

    if coordinate_suffix == '_recentered.txt':
        try:
            x_target = np.array(iraf.txdump(textfile=image_name[:-5]+'.als.2',fields='XCENTER',expr='yes',Stdout=1)).astype(float)[0]
            y_target = np.array(iraf.txdump(textfile=image_name[:-5]+'.als.2',fields='YCENTER',expr='yes',Stdout=1)).astype(float)[0]
            np.savetxt(image_name[:-5]+coordinate_suffix, (x_target, y_target), newline=" ")
            print("1")
        except:
            coordinate_suffix = '_coords_targ.txt'

    exists5 = check_existence(image_name[:-5] + '.apmag.out',   'aperture_phot')
    if exists5:
        subprocess.call('rm ' + image_name[:-5] + '.apmag.out', shell=True)

    # Run phot to get the aperture photometry magniutes of stars
    phot(image       = image_name_in + image_frame,       # The input image(s)
         skyfile     = '',                                # The input sky file(s)
         coords      = image_name[:-5]+coordinate_suffix, # The input coordinate file(s) (default: image.coo.)
         output      = image_name[:-5]+'.apmag.out',      # The output photometry file(s) (default: image.mag.)
         plotfile    = '',                                # The output plots metacode file
         interactive = 'no',                              # Interactive mode?
         radplots    = 'no',                              # Plot the radial profiles in interactive mode?
         icommands   = '',                                # Image cursor: [x y wcs] key [cmd]
         gcommands   = '',                                # Graphics cursor: [x y wcs] key [cmd]
         wcsin       = ')_.wcsin',                        # Input coordinate system (logical, tv, physical, world)
         wcsout      = ')_.wcsout',                       # Output coordinate system (logical, tv, physical, world)
         cache       = 'no',                              # Cache the input image pixels in memory?
         verify      = 'no',                              # Verify critical parameters in non-interactive mode?
         update      = 'no',                              # Update critical parameters in non-interactive mode?
         verbose     = 'no',                              # Print messages in non-interactive mode?
         graphics    = ')_.graphics',                     # Graphics device
         display     = 'stdgraph',                        # Display device
         mode        = 'ql'                               # IRAF mode
        )

    # Import output data from IRAF
    aperture_mags     = np.array(iraf.txdump(textfile=image_name[:-5]+'.apmag.out',fields='MAG' ,expr='yes',Stdout=1))
    aperture_mags_err = np.array(iraf.txdump(textfile=image_name[:-5]+'.apmag.out',fields='MERR',expr='yes',Stdout=1))
    aperture_sums     = np.array(iraf.txdump(textfile=image_name[:-5]+'.apmag.out',fields='SUM',expr='yes',Stdout=1))

    # Conver indefs to nan's
    aperture_mags[aperture_mags == 'INDEF'] = 'nan'
    aperture_mags_err[aperture_mags_err == 'INDEF'] = 'nan'

    # Convert to floats
    aperture_mags = aperture_mags.astype(float)
    aperture_mags_err = aperture_mags_err.astype(float)

    return aperture_mags, aperture_mags_err, aperture_sums

def zeropoint(parameters_list, system_mags, system_errors, good_cat, min_zeropoint = 2, max_zeropoint = 2, SNR_zeropoint = 1, zeropoint_sigma = 2.0, iter_zeropoint = 1, maximum_separation = 6, plot_zeropoint = True):
    '''
    Calculate the zeropoint of a set of magnitudes as compared 
    to the true stars in the catalogue.

    Parameters
    ---------------
    parameters_list    : List of relevant parameters form loadimage()
    system_mags        : List of PSF or Aperture magnitudes
    system_errors      : List of PSF or Aperture magnitude errors
    good_cat           : Catalogue with the true magnitudes
    min_zeropoint      : Crop any star that's this sigma amount below the average zeropoint
    max_zeropoint      : Crop any star that's this sigma amount above the average zeropoint
    SNR_zeropoint      : minimum SNR to calculate the zeropoint
    zeropoint_sigma    : Sigma for excluding zeropoint data
    iter_zeropoint     : Iterations for zeropoint clipping calculation
    maximum_separation : Maximum separation of stars to allow for zeropoint calculation, in arcsec
    plot_zeropoint     : Plot the actual zeropoints?

    Output
    ---------------
    Zeropoint value and error, as well as a plot with the zeropoint
    '''

    # Get color and distance
    color       = parameters_list['filter']
    distance_in = np.sqrt((good_cat['xcentroid'] - good_cat['cat_x'])**2 + (good_cat['ycentroid'] - good_cat['cat_y'])**2)

    SNR = 0.5 * (good_cat['SNR'] + 1 / system_errors)
    zero_point = system_mags - good_cat[color + 'mag'].astype(float)

    # Calculate the average zeropoint for the high SNR points
    good = np.where((SNR > SNR_zeropoint) & (distance_in < maximum_separation))

    # Relax limits if no stars were found
    if len(good[0]) <= 2:
        good = np.where((SNR > SNR_zeropoint / 5) & (distance_in < maximum_separation * 3.0))
        average_zeropoint, good_zeropoint = weighted_average(zero_point[good], SNR[good], sigma_lower = min_zeropoint, sigma_upper = max_zeropoint, iterations = int(iter_zeropoint))
    else:
        average_zeropoint, good_zeropoint = weighted_average(zero_point[good], SNR[good], sigma_lower = min_zeropoint, sigma_upper = max_zeropoint, iterations = int(iter_zeropoint))

    min_ypoint = min([average_zeropoint - 0.5, np.min(zero_point[good][good_zeropoint])])
    max_ypoint = max([average_zeropoint + 0.5, np.max(zero_point[good][good_zeropoint])])

    # Errorbar
    zero_mean, _, zero_std = sigma_clipped_stats_version(zero_point[good][good_zeropoint], sigma_low=zeropoint_sigma, sigma_hi=zeropoint_sigma, iterations=1)

    if plot_zeropoint:
        plt.subplot('334')
        plt.axhline(y = average_zeropoint, color = 'k', linestyle = '-', label = r'$zero = %s \pm %s$'%(str(np.around(average_zeropoint, decimals = 2)), str(np.around(zero_std, decimals = 2))))
        plt.legend(loc = 'upper right', framealpha=0.3)
        plt.errorbar(SNR[good][good_zeropoint],zero_point[good][good_zeropoint], yerr= system_errors[good][good_zeropoint].astype(float), color = 'g', markersize = 10, fmt = '.')
        plt.errorbar(SNR,zero_point, yerr= system_errors.astype(float), color = 'g', markersize = 10, alpha = 0.3, label = '', fmt = '.')
        plt.axhline(y = average_zeropoint + zero_std, color = 'k', linestyle = '--')
        plt.axhline(y = average_zeropoint - zero_std, color = 'k', linestyle = '--')
        plt.ylim(min_ypoint, max_ypoint)
        plt.xscale('log')
        plt.xlabel("SNR")
        plt.ylabel("Zero Point")

    return average_zeropoint, zero_std

def target_position(image_name, image_data, wcs_data, coord, coordinate_suffix, parameters_list, target_pix_range = 10, best_fwhm = 3.5, threshold_target = 2.0, image_substracted_file = '', target_fwhm_prefactor = 1.0):
    '''
    Calculate the pixel position of the target and save a file to be read
    in by IRAF's phot with the object's coordinates.

    Parameters
    ---------------
    image_name             : Name of the image to process
    image_data             : Numpy array of data
    wcs_data               : Output form loadimage() with wcs data
    coord                  : Coordiantes in SkyCoord format form loadimage()
    coordinate_suffix      : Suffix for the file with the pixel coordinates in (x, y)
    parameters_list        : List of relevant parameters form loadimage()
    target_pix_range       : Search radius for the target star
    best_fwhm              : Full Width Half Max of DAOphot to find stars.
    threshold_target       : threshold above the background that a star would be detected.
    image_substracted_file : Specify a file that comes from an image subtraction routine.
                             To do photometry with the parameters of the main image but on this one.

    Output
    ---------------
    x, y coordinates of the target to a file with suffix '_coords_targ.txt'
    It also returns the mean and std to keep the same scale in the substracted image

    '''

    # Use the alternative image if specified
    if image_substracted_file != '':
        image_name_in = image_substracted_file
        header_data   = fits.open(image_name_in)
        image_data    = header_data[0].data
        wcs_data      = wcs.WCS(header_data[0].header)
    else:
        image_name_in = image_name

    # Get pixel position of source in the image
    cat_coords_pix = wcs_data.wcs_world2pix(coord.ra.deg, coord.dec.deg, 1)

    # Crop the area to make sigma clipped calculations excluding the edges of the images
    pos_x = cat_coords_pix[0]
    pos_y = cat_coords_pix[1]
    xmin = int(np.around(pos_x-target_pix_range))
    xmax = int(np.around(pos_x+target_pix_range))
    ymin = int(np.around(pos_y-target_pix_range))
    ymax = int(np.around(pos_y+target_pix_range))

    # Crop the data
    cropped_data = image_data[ymin:ymax,xmin:xmax]

    # Get the mean of the target
    mean_target, median_target, std_target = sigma_clipped_stats_version(cropped_data, sigma_low=2.0, sigma_hi=1.0, iterations=5)

    # Detect stars in using the DAOFIND (Stetson 1987) in an image for local density maxima
    # that have a peak amplitude greater than 'threshold' and have a size and shape similar
    # to the defined 2D Gaussian kernel
    daofind = DAOStarFinder(fwhm=best_fwhm, threshold=threshold_target*std_target, sigma_radius = best_fwhm / 2.0 * np.sqrt(2 * np.log10(2)))
    sources = daofind(cropped_data - mean_target)

    if not sources:
        daofind = DAOStarFinder(fwhm=best_fwhm, threshold=threshold_target*std_target / 1.5, sigma_radius = 0.8 * best_fwhm / 2.0 * np.sqrt(2 * np.log10(2)))
        sources = daofind(cropped_data - mean_target)

        if not sources:
            daofind = DAOStarFinder(fwhm=best_fwhm, threshold=threshold_target*std_target / 3.0, sigma_radius = 0.4 * best_fwhm / 2.0 * np.sqrt(2 * np.log10(2)))
            sources = daofind(cropped_data - mean_target)

        if not sources:
            daofind = DAOStarFinder(fwhm=best_fwhm, threshold=threshold_target*std_target / 3.0, sigma_radius = 2.0 * best_fwhm / 2.0 * np.sqrt(2 * np.log10(2)))
            sources = daofind(cropped_data - mean_target)

    else:
        # If there were no sources found, decrease the threshold
        if len(sources) == 0:
            daofind = DAOStarFinder(fwhm=best_fwhm, threshold=threshold_target*std_target / 1.5, sigma_radius = best_fwhm / 2.0 * np.sqrt(2 * np.log10(2)))
            sources = daofind(cropped_data - mean_target)
        if len(sources) == 0:
            daofind = DAOStarFinder(fwhm=best_fwhm, threshold=threshold_target*std_target / 3.0, sigma_radius = best_fwhm / 0.5 * np.sqrt(2 * np.log10(2)))
            sources = daofind(cropped_data - mean_target)
        if len(sources) == 0:
            daofind = DAOStarFinder(fwhm=best_fwhm, threshold=threshold_target*std_target / 4.0, sigma_radius = best_fwhm / 0.2 * np.sqrt(2 * np.log10(2)))
            sources = daofind(cropped_data - mean_target)

    # Calculate chance coincidence from separation and magnitude of each star.
    separation = np.sqrt((sources['xcentroid'] - target_pix_range) ** 2 + (sources['ycentroid'] - target_pix_range) ** 2)
    magnitude  = sources['mag']
    chance     = calculate_coincidence(separation, magnitude)
    best_match = np.argmin(chance)

    # Get coordinates of star to match
    pixel_centroid = np.array([sources[best_match]['xcentroid'], sources[best_match]['ycentroid']])

    # Define the aperture for that star that's 2.5 X the FWHM value
    single_aperture = CircularAperture(pixel_centroid, r=(best_fwhm/2.0)*2.5)

    # Number of pixels in the aperture area
    try:
        n = single_aperture.area()
    except:
        n = single_aperture.area

    # Perform aperture photometry to obtain sum of counts in aperture
    do_phot = aperture_photometry(cropped_data, single_aperture)

    # Calculate the signal is (Signal - Background) * Gain
    signal = (do_phot['aperture_sum'] - mean_target*n) * parameters_list['gain']

    # Calculate the noise
    signal_noise = signal
    read_noise   = n * parameters_list['rdnoise']**2
    sky_noise    = n * mean_target * parameters_list['gain']
    noise_total  = np.sqrt(signal_noise + read_noise + sky_noise)

    # Calculate signal to noise
    signal_to_noise = np.array(signal / noise_total)[0]

    # Calcualte the pixel position of the target
    target_coords_img = wcs_data.wcs_world2pix(coord.ra.deg, coord.dec.deg, 1)

    # Save image with apertures
    plt.subplot('336')
    CircularAperture([(sources[best_match]['xcentroid'], sources[best_match]['ycentroid'])], r=float((best_fwhm/2.0)*2.5)).plot(color='green', linestyle = '--', lw=2.0, alpha=0.5)
    plt.scatter(target_coords_img[0] - xmin, target_coords_img[1] - ymin, color = 'r', marker = 'x', s = 100, alpha = 0.5)
    plt.imshow(cropped_data, vmin = mean_target-4.0*std_target, vmax = mean_target+10.0*std_target, cmap='Greys', origin='lower',interpolation='none')
    plt.xlabel(parameters_list['mjd'])

    # Re-offset to the correct image position.
    sources[best_match]['xcentroid'] += xmin
    sources[best_match]['ycentroid'] += ymin

    # Save pixel coordinates to file
    target_coordinates = (sources[best_match]['xcentroid'], sources[best_match]['ycentroid'])
    np.savetxt(image_name[:-5] + coordinate_suffix, target_coordinates, newline=" ")

    return mean_target, std_target

def target_psf(image_name, image_data, wcs_data, coord, coordinate_suffix, mean_target, std_target, parameters_list, n_iterations = 10, best_fwhm = 3.5, target_pix_range = 10, varorder = 0, sky_annulus = 8.0, recenter = 'yes', centering_fwhm = 3.0, zmag = 30.0, overexposed_limit = 50000, image_substracted_file = '', image_frame = ''):
    '''
    Do aperture photometry on the image with image_name with the coordinate file
    that has name image_name_coords.txt.

    Parameters
    ---------------
    image_name             : Name of the image to process
    image_data             : Numpy array of data
    wcs_data               : Output form loadimage() with wcs data
    coord                  : Coordiantes in SkyCoord format form loadimage()
    coordinate_suffix      : Suffix for the file with the pixel coordinates in (x, y)
    mean_target            : Mean counts of target from target_position()
    std_target             : Standard deviation of counts in target iamge from target_position()
    parameters_list        : List of relevant parameters form loadimage()
    n_iterations           : Maximum Number of sky fitting iterations
    best_fwhm              : FWHM value of stars as determined by estimate_fwhm() in inputs of pixels.
    target_pix_range       : Search radius for the stars in the catalogue.
    varorder               : 1 will have 3 look up tables, 2 will have 6, etc. Set to -1 to 2
    sky_annulus            : Width of sky fitting annulus
    recenter               : Recenter PSF fitting model
    centering_fwhm         : Prefactor for the box of the centering algortithm.
    zmag                   : Zero point of magnitude scale
    overexposed_limit      : Maximum number of counts before discarding the star
    image_substracted_file : Specify a file that comes from an image subtraction routine.
                             To do photometry with the parameters of the main image but on this one.
    image_frame            : Fits frame, i.e. '[0]'

    Output
    ---------------
    Prints the best aperture and returns the magnitudes
    and errors for that aperture. Returns the magnitude and error of the target
    '''

    # Use the alternative image if specified
    if image_substracted_file != '':
        image_name_in = image_substracted_file
        header_data   = fits

        image_name_in = image_substracted_file
        header_data   = fits.open(image_name_in)
        image_data    = header_data[0].data
        wcs_data      = wcs.WCS(header_data[0].header)
    else:
        image_name_in = image_name

    # Set daophot parameters
    daophot_parameters(parameters_list, n_iterations, best_fwhm, target_pix_range, varorder, sky_annulus, recenter, centering_fwhm, zmag, overexposed_limit)

    exists1 = check_existence(image_name[:-5] + '.mag2.out',   'psf_phot')
    if exists1:
        subprocess.call('rm ' + image_name[:-5] + '.mag2.out', shell=True)

    exists2 = check_existence(image_name[:-5] + '.als.2',   'psf_phot')
    if exists2:
        subprocess.call('rm ' + image_name[:-5] + '.als.2', shell=True)

    exists3 = check_existence(image_name[:-5] + '.arj.2',   'psf_phot')
    if exists3:
        subprocess.call('rm ' + image_name[:-5] + '.arj.2', shell=True)

    exists4 = check_existence(image_name[:-5] + '.sub.2.fits',   'psf_phot')
    if exists4:
        subprocess.call('rm ' + image_name[:-5] + '.sub.2.fits', shell=True)

    # Run phot to get the aperture photometry magniutes of stars
    phot(image       = image_name_in + image_frame,              # The input image(s)
         skyfile     = '',                                       # The input sky file(s)
         coords      = image_name[:-5]+coordinate_suffix,        # The input coordinate file(s) (default: image.coo.)
         output      = image_name[:-5]+'.mag2.out',              # The output photometry file(s) (default: image.mag.)
         plotfile    = '',                                       # The output plots metacode file
         interactive = 'no',                                     # Interactive mode?
         radplots    = 'no',                                     # Plot the radial profiles in interactive mode?
         icommands   = '',                                       # Image cursor: [x y wcs] key [cmd]
         gcommands   = '',                                       # Graphics cursor: [x y wcs] key [cmd]
         wcsin       = ')_.wcsin',                               # Input coordinate system (logical, tv, physical, world)
         wcsout      = ')_.wcsout',                              # Output coordinate system (logical, tv, physical, world)
         cache       = 'no',                                     # Cache the input image pixels in memory?
         verify      = 'no',                                     # Verify critical parameters in non-interactive mode?
         update      = 'no',                                     # Update critical parameters in non-interactive mode?
         verbose     = 'no',                                     # Print messages in non-interactive mode?
         graphics    = ')_.graphics',                            # Graphics device
         display     = 'stdgraph',                               # Display device
         mode        = 'ql'                                      # IRAF mode
        )

    # Group and fit PSF to multiple stars simultaneously
    daophot.allstar(image       = image_name_in + image_frame,   # Image corresponding to photometry
                    photfile    = image_name[:-5]+'.mag2.out',   # Input photometry file
                    psfimage    = image_name[:-5]+'.psf.1.fits', # PSF image
                    allstarfile = image_name[:-5]+'.als.2',      # Output photometry file
                    rejfile     = image_name[:-5]+'.arj.2',      # Output rejections file
                    subimage    = image_name[:-5]+'.sub.2.fits', # Substracted image
                    wcsin       = ')_.wcsin',                    # Input coordiante system
                    wcsout      = ')_.wcsout',                   # Output coordinate system
                    wcspsf      = ')_.wcspsf',                   # PSF coordinate system
                    cache       = 'no',                          # Cache the input image pixels in memory?
                    verify      = 'no',                          # Verify critical phot parameters?
                    update      = 'no',                          # Update critical phot parameters?
                    verbose     = 'yes',                         # Print phot messages?
                    version     = '2',                           # Version
                    mode        = 'ql'                           # IRAF Mode
                    )

    # Import output data from IRAF
    mags   = np.array(iraf.txdump(textfile=image_name[:-5]+'.als.2',fields='MAG',expr='yes',Stdout=1))
    merrs  = np.array(iraf.txdump(textfile=image_name[:-5]+'.als.2',fields='MERR',expr='yes',Stdout=1))

    if len(mags) > 0:
        # Conver indefs to nan's
        mags[mags   == 'INDEF'] = 'nan'
        merrs[merrs == 'INDEF'] = 'nan'
        # Convert to floats
        target_psf_mag     = mags.astype(float)
        target_psf_mag_err = merrs.astype(float)
    else:
        target_psf_mag     = -999
        target_psf_mag_err = -999

    # Get pixel position of source in the image
    cat_coords_pix = wcs_data.wcs_world2pix(coord.ra.deg, coord.dec.deg, 1)

    # Crop the area to make sigma clipped calculations excluding the edges of the images
    datasize_x = cat_coords_pix[0]
    datasize_y = cat_coords_pix[1]
    xmin = int(np.around(datasize_x-target_pix_range))
    xmax = int(np.around(datasize_x+target_pix_range))
    ymin = int(np.around(datasize_y-target_pix_range))
    ymax = int(np.around(datasize_y+target_pix_range))

    # Get the raw image data around the cropped area
    image_data_crop = image_data[ymin:ymax,xmin:xmax]

    # Get the Substracted image data around the cropped area
    substracted_data = fits.open(image_name[:-5]+'.sub.2.fits')[0].data
    substracted_crop = substracted_data[ymin:ymax,xmin:xmax]

    # Sum of raw image
    x_sum = np.sum(image_data_crop, axis = 0)
    y_sum = np.sum(image_data_crop, axis = 1)

    # Sum of cropped iamge
    x_crop = np.sum(substracted_crop, axis = 0)
    y_crop = np.sum(substracted_crop, axis = 1)

    # Import PSF model data
    psf_data = fits.open(image_name[:-5]+'.PSF_OUT.fits')[0].data
    mean_spf, median_spf, std_spf = sigma_clipped_stats_version(psf_data, sigma_low=1.0, sigma_hi=1.0, iterations=2)

    # Plot PSF data
    plt.subplot('337')
    plt.imshow(psf_data, vmin = mean_spf - 1 * std_spf, vmax = mean_spf + 2 * std_spf, cmap='Greys', origin='lower',interpolation='none')

    # Plot Substracted target data
    plt.subplot('338')
    try:
        x_target = np.array(iraf.txdump(textfile=image_name[:-5]+'.als.2',fields='XCENTER',expr='yes',Stdout=1)).astype(float)[0]
        y_target = np.array(iraf.txdump(textfile=image_name[:-5]+'.als.2',fields='YCENTER',expr='yes',Stdout=1)).astype(float)[0]
        CircularAperture([(x_target - xmin, y_target - ymin)], r=float((best_fwhm/2.0)*2.5)).plot(color='green',  lw=2.0, alpha=0.5)
    except:
        print("No centroid found")

    # Plot Target position
    coordinate_target = np.genfromtxt(image_name[:-5]+coordinate_suffix)
    CircularAperture([(coordinate_target[0] - xmin, coordinate_target[1] - ymin)], r=float((best_fwhm/2.0)*2.5)).plot(color='green',  lw=2.0, alpha=0.5, linestyle = '--')
    if image_substracted_file != '':
        mean_target, median_target, std_target = sigma_clipped_stats_version(substracted_crop, sigma_low=2.0, sigma_hi=1.0, iterations=5)
    plt.imshow(substracted_crop, vmin = mean_target-4.0*std_target, vmax = mean_target+10.0*std_target, cmap='Greys', origin='lower',interpolation='none')

    # Plot apertures on original image
    plt.subplot('332')
    try:
        CircularAperture([(x_target, y_target)], r=float((best_fwhm/2.0)*2.5)).plot(color='green', lw=1.0, alpha=0.5)
    except:
        print("No centroid found")

    # Plot PSF with substracted and raw
    plt.subplot('339')
    plt.plot(x_sum  / np.average(x_sum),  label = 'X-axis raw', color = 'C0', linestyle = '--')
    plt.plot(y_sum  / np.average(y_sum),  label = 'Y-axis raw', color = 'C1', linestyle = '--')
    plt.plot(x_crop / np.average(x_crop), label = 'X-axis sub', color = 'C0')
    plt.plot(y_crop / np.average(y_crop), label = 'Y-axis sub', color = 'C1')
    plt.xlabel("Pixel")

    return target_psf_mag, target_psf_mag_err

def snr_equation(x, a, b, s):
    '''
    Equation that relates the signal to noise to a magnitude

    Parameters
    -------------
    x: data
    a: Amplitude
    b: x-shift
    s: y-shift

    Output
    --------------
    log function
    '''
    return - a * np.log(x - b) + s

def calc_non_detection(best_cat, parameters_list, image_data, wcs_data, mean_target, std_target, target_pix_range = 10, n_sigma = 2):
    '''
    Calculate the magnitude of an n_sigma non_detection.

    Parameters
    -------------
    best_cat            : Catalogue with the true magnitudes
    parameters_list     : List of relevant parameters form loadimage()
    image_data          : Numpy array of data
    wcs_data            : Output form loadimage() with wcs data
    mean_target         : Mean counts of target from target_position()
    std_target          : Standard deviation of counts in target iamge from target_position()
    target_pix_range    : Search radius for the target star
    n_sigma             : Number of sigmas for which to calculate the non-detection

    Output
    --------------
    Magnitude of sigma non_detection value
    '''

    # Get SNR and Magnitude
    color = parameters_list['filter']
    snr   = best_cat['SNR']
    mag   = best_cat[color + 'mag']

    # Fit the log equation
    popt1, pcov = curve_fit(snr_equation, snr, mag)

    # Calculate the Residuals
    residuals = mag - snr_equation(snr, *popt1)
    std       = np.std(residuals)
    average   = np.average(residuals)

    # Only use good points for 2nd fit
    good = np.where((residuals < 0 + 1.0 * std) & (residuals > average - 3.0 * std) & (snr > 1.5))
    bad  = np.where(np.logical_not((residuals < 0 + 1.0 * std) & (residuals > average - 3.0 * std) & (snr > 1.5)))

    # Fit again
    popt2, pcov = curve_fit(snr_equation, snr[good], mag[good], sigma = 1 / np.sqrt(snr[good]))

    # Calculate the n_sigma non_detection
    non_detection_mag = snr_equation(n_sigma, *popt2)
    mag_label         = np.around(non_detection_mag, decimals = 2)

    # Plot output
    xarray = np.linspace(0, 200, 500)
    plt.subplot('337')
    plt.xscale('log')
    plt.xlim(1, 200)
    plt.scatter(n_sigma, non_detection_mag, color = 'k', s = 40, label = r'$%s\sigma = %s$'%(n_sigma, mag_label))
    plt.legend(loc = 'upper right')
    plt.scatter(snr[bad], mag[bad], color = 'r', s = 5, alpha = 0.1)
    plt.errorbar(snr[good], mag[good], color = 'C0', yerr = 1 / np.sqrt(snr[good]), fmt = '.', markersize = 7, alpha = 0.1)
    plt.plot(xarray, snr_equation(xarray, *popt1), color = 'k', linestyle = '--', alpha = 0.5)
    plt.plot(xarray, snr_equation(xarray, *popt2), color = 'k')
    plt.xlabel("SNR")
    plt.ylabel("Catalog Magnitude")

    plt.subplot('338')
    plt.xscale('log')
    plt.ylim(non_detection_mag - 1.5, non_detection_mag + 1.5)
    plt.xlim(n_sigma - 0.7, n_sigma + 5.0)
    plt.scatter(snr[bad], mag[bad], color = 'r', s = 5, alpha = 0.1)
    plt.errorbar(snr[good], mag[good], color = 'C0', yerr = 1 / np.sqrt(snr[good]), fmt = '.', markersize = 7, alpha = 0.1)
    plt.scatter(n_sigma, non_detection_mag, color = 'k', s = 40)
    plt.plot(xarray, snr_equation(xarray, *popt2), color = 'k')
    plt.xlabel("SNR")

    # Find the star with the magnitude closest to the calculate non detection mag
    closest = np.argmin(np.abs(mag - non_detection_mag))
    closest_star = best_cat[closest]

    # Get pixel position of source in the image
    coords_img = wcs_data.wcs_world2pix(float(closest_star['RA']), float(closest_star['DEC']), 1)

    # Crop the area to make sigma clipped calculations excluding the edges of the images
    datasize_x = coords_img[0]
    datasize_y = coords_img[1]
    xmin = int(np.around(datasize_x-target_pix_range))
    xmax = int(np.around(datasize_x+target_pix_range))
    ymin = int(np.around(datasize_y-target_pix_range))
    ymax = int(np.around(datasize_y+target_pix_range))

    # Get the raw image data around the cropped area
    image_data_crop = image_data[ymin:ymax,xmin:xmax]

    # Plot Substracted target data
    plt.subplot('339')
    plt.imshow(image_data_crop, vmin = mean_target-3.0*std_target, vmax = mean_target+3.0*std_target, cmap='Greys', origin='lower',interpolation='none')
    plt.xlabel("Sample %s mag star"%np.around(closest_star[color + 'mag'], decimals = 2))

    '''
    # Find the star with the magnitude closest to the calculate non detection mag
    closest = np.argmin(np.abs(mag - snr_equation(3.0, *popt2)))
    closest_star = best_cat[closest]

    # Get pixel position of source in the image
    coords_img = wcs_data.wcs_world2pix(float(closest_star['RA']), float(closest_star['DEC']), 1)
    print(coords_img)

    # Crop the area to make sigma clipped calculations excluding the edges of the images
    datasize_x = coords_img[0]
    datasize_y = coords_img[1]
    xmin = int(np.around(datasize_x-target_pix_range))
    xmax = int(np.around(datasize_x+target_pix_range))
    ymin = int(np.around(datasize_y-target_pix_range))
    ymax = int(np.around(datasize_y+target_pix_range))

    # Get the raw image data around the cropped area
    image_data_crop = image_data[ymin:ymax,xmin:xmax]

    # Plot Substracted target data
    plt.subplot(9,9,81)
    plt.imshow(image_data_crop, vmin = mean_target-2.0*std_target, vmax = mean_target+4.0*std_target, cmap='Greys', origin='lower',interpolation='none')
    plt.ylabel(r"$%s\sigma = %s mag$"%(3, np.around(closest_star[color + 'mag'], decimals = 2)), rotation = 0, horizontalalignment = 'right')
    plt.tick_params(axis='both', left='off', top='off', right='off', bottom='off', labelleft='off', labeltop='off', labelright='off', labelbottom='off')
    '''

    return non_detection_mag

def substract_host(target_mag, host_mag = 999):
    '''
    Take the magnitude of a transient + host and substract
    the known magnitude of the host.
    '''

    # Only do if the target was detected
    if target_mag < -900:
        return -999

    # Calculate the Luminosities
    Lt = 10 ** (-0.4 * target_mag)
    Lg = 10 ** (-0.4 * host_mag)

    # Substract
    Lr = Lt - Lg

    # Convert back to magntiude
    Mr = -2.5 * np.log10(Lr)

    return Mr

def plot_comparison(parameters_list, target_mag, target_err, stars_mag, stars_err, zeropoint, zeropoint_err, best_cat, aperture = False, plot_range = 3):
    '''
    Plot a comparison plot of the true vs. calculated magnitudes.

    Parameters
    -------------
    parameters_list : List of relevant parameters form loadimage()
    target_mag      : Magnitude of the target
    target_err      : Errorbar of target magnitude
    stars_mag       : Magnitudes of the rest of the stars
    stars_err       : Errors of the stars
    zeropoint       : Zeropoint calculated in magnitudes
    zeropoint_err   : Errorbar on the zeropoint in magnitudes
    best_cat        : Catalogue stars with the true magnitudes
    aperture        : Doing aperture photometry? If so, plot with blue colors.
    plot_range      : Plot the magnitudes plus minus this value

    Output
    -------------
    Plot and save output text file with magnitude
    '''

    # Calibrate Target Magnitude
    calibrated_mag = target_mag - zeropoint
    calibrated_err = np.sqrt(target_err**2 + zeropoint_err**2)

    # Calibrate the rest of the stars
    cal_mags = stars_mag - zeropoint
    cal_errs = np.sqrt(stars_err**2 + zeropoint_err**2)

    # Extract true magnitudes
    color = parameters_list['filter']
    true_mags = best_cat[color + 'mag']

    # If doing aperture photometry, plot in green
    if aperture:
        color_plot = 'green'
        name = 'Ap. mag'
    else:
        color_plot = 'C0'
        name = color + ' mag'

    # Convert to floats
    if type(calibrated_mag) == type(np.array(0)): calibrated_mag = calibrated_mag[0]
    if type(calibrated_err) == type(np.array(0)): calibrated_err = calibrated_err[0]

    # Plot output
    plt.subplot('335')
    plt.errorbar(true_mags, cal_mags, yerr = cal_errs, fmt = '.', alpha = 0.5, color = color_plot)
    plt.plot([min(true_mags-plot_range), max(true_mags)+plot_range], [min(true_mags-plot_range), max(true_mags)+plot_range], color = 'C1')
    plt.xlim(min(true_mags-plot_range), max(true_mags)+plot_range)
    plt.ylim(min(true_mags-plot_range), max(true_mags)+plot_range)

    # If using an upper limit, plot with '>' label
    if target_err == -1.0:
        label = r'%s > $%s$'%(name, np.around(calibrated_mag, decimals = 2))
        plt.errorbar(calibrated_mag, calibrated_mag, yerr = 0.0, fmt = '.', color = 'r', label = label)
    else:
        label = r'%s = $%s \pm %s$'%(name, np.around(calibrated_mag, decimals = 2), np.around(calibrated_err, decimals = 2))
        plt.errorbar(calibrated_mag, calibrated_mag, yerr = calibrated_err, fmt = '.', color = 'r', label = label)

    plt.xlabel("Catalog Magnitude")
    plt.ylabel("Calculated Magnitude")
    plt.legend(loc = 'upper left', frameon = False)

def calc_percentile(single_chain):
    mcmc = np.percentile(single_chain, [15.87, 50, 84.13])
    output = mcmc[1], mcmc[2] - mcmc[1], mcmc[1] - mcmc[0]
    return output

def upper_limit(parameters_list, best_fwhm, average_zeropoint, just_nearby, zmag = 30):
    '''
    Calculate the upper limit of the observation given the background distribution.

    Parameters
    -------------
    parameters_list   : List of relevant parameters form loadimage()
    best_fwhm         : Full Width Half Max of DAOphot to find stars.
    average_zeropoint : Zeropoint calculated by the zeropoint() function
    just_nearby       : Background counts near closeby stars
    zmag              : Zero point of magnitude scale

    Output
    -------------
    1, 2, and 3 sigma upper limits.
    '''

    # Get the paramters from the header and calculate the area in pixels
    gain    = float(parameters_list['gain'])
    exptime = float(parameters_list['exptime'])
    area    = np.pi * (best_fwhm / 2 * 2.5) ** 2

    # Calculate the 1 sigma percentiles of the background
    average_background, average_std_high, average_std_low = calc_percentile(just_nearby)

    def calc_mag(SNR, gain, average_std_high, area, zmag, exptime, average_zeropoint):
        a    =   1.0 / (SNR * 1.0857) ** 2
        b    = - 1.0 / gain
        c    = - (area * average_std_high**2 + area**2)
        flux = (- b + np.sqrt(b ** 2 - 4 * a * c) )/ (2 * a)
        mag  = zmag - 2.5 * np.log10 (flux) + 2.5 * np.log10 (exptime)
        return mag - average_zeropoint

    sigma1 = calc_mag(1.0, gain, average_std_high, area, zmag, exptime, average_zeropoint)
    sigma2 = calc_mag(2.0, gain, average_std_high, area, zmag, exptime, average_zeropoint)
    sigma3 = calc_mag(3.0, gain, average_std_high, area, zmag, exptime, average_zeropoint)

    plt.subplot('339')
    plt.xlim((average_background - average_std_high * 5) * area, (average_background + average_std_high * 5) * area)
    plt.hist(just_nearby * area, bins = 100, alpha = 0.4, color = 'k', weights = 0.02 * np.ones(len(just_nearby)))
    plt.ylim(ymin = 0)
    plt.scatter((average_background + average_std_high * 1) * area, sigma1, color = 'r')
    plt.scatter((average_background + average_std_high * 2) * area, sigma2, color = 'r')
    plt.scatter((average_background + average_std_high * 3) * area, sigma3, color = 'r')

    plt.axvline(x = average_background * area, color = 'k', linestyle = '-')
    plt.axvline(x = (average_background + average_std_high * 3) * area, color = 'k', linestyle = '--')
    plt.axvline(x = (average_background - average_std_high * 3) * area, color = 'k', linestyle = '--')

    plt.xlabel('Counts')
    plt.ylabel('Magnitude')

    return sigma1 + average_zeropoint, sigma2 + average_zeropoint, sigma3 + average_zeropoint

def clean_up(directory = '.', difference = False):
    '''
    Remove files created during data reduction.

    all: Also remove output magnitude files?
    '''
    os.system("rm %s/*_coords_targ.txt"%directory)
    os.system("rm %s/*_coords.txt"%directory)
    os.system("rm %s/*.mag.out"%directory)
    os.system("rm %s/*.apmag.out"%directory)
    os.system("rm %s/*.mag2.out"%directory)
    os.system("rm %s/*.psg.1"%directory)
    os.system("rm %s/*.pst.1"%directory)
    os.system("rm %s/*.pst.2"%directory)
    os.system("rm %s/*.pst.3"%directory)
    os.system("rm %s/*.als.1"%directory)
    os.system("rm %s/*.als.2"%directory)
    os.system("rm %s/*.arj.1"%directory)
    os.system("rm %s/*.arj.2"%directory)
    os.system("rm %s/*.sub.2.fits"%directory)
    os.system("rm %s/*.sub.1.fits"%directory)
    os.system("rm %s/*.PSF_OUT.fits"%directory)
    os.system("rm %s/*.psf.1.fits"%directory)
    os.system("rm %s/*_recentered.txt"%directory)
    if difference:
        os.system("rm %s/*_diff.fits"%directory)
        os.system("rm %s/*_science.fits"%directory)
        os.system("rm %s/*_template.fits"%directory)

def create_lightcurve(object_name):
    '''
    Create a file with Mag, Mag Error, MJD, Filter with the files
    generated by the do_photometry code into one single text file
    '''

    emptys = ['',' ','None','--', '-', b'',b' ',b'None',b'--', b'-', None, np.nan, 'nan', b'nan', '0']

    # Read in the output files
    files = glob.glob(object_name + '/*_PSF.txt')

    # Empty variables
    all_mags  = []
    all_errs  = []
    all_mjds  = []
    all_filts = []

    # For each file import the data
    for i in range(len(files)):
        MJD, PSF_Mag, Aperture_Mag, PSF_Err, Aperture_Err, Filter = np.genfromtxt(files[i], unpack = True, skip_header = 1, dtype = 'str')

        # Save to array
        all_mags  = np.append(all_mags,  PSF_Mag)
        all_errs  = np.append(all_errs,  PSF_Err)
        all_mjds  = np.append(all_mjds,  MJD)
        all_filts = np.append(all_filts, Filter)

    # Sort by date
    all_mags  = all_mags[np.argsort(all_mjds.astype(float))]
    all_errs  = all_errs[np.argsort(all_mjds.astype(float))]
    all_filts = all_filts[np.argsort(all_mjds.astype(float))]
    all_mjds  = all_mjds[np.argsort(all_mjds.astype(float))]

    # Remove nans
    do_use = [i not in emptys for i in all_mags]

    # Save output
    array = (all_mjds[do_use], all_mags[do_use], all_errs[do_use], all_filts[do_use])
    np.savetxt('photometry/' + object_name + ".txt", np.transpose(array), fmt = '%s', header = 'MJD Mag Mag_Err Filter')
    print("Saved %s.txt with %s datapoints"%(object_name, len(np.transpose(array))))

def aperture_create_lightcurve(object_name):
    '''
    Create a file with Mag, Mag Error, MJD, Filter with the files
    generated by the do_photometry code into one single text file
    '''

    emptys = ['',' ','None','--', '-', b'',b' ',b'None',b'--', b'-', None, np.nan, 'nan', b'nan', '0']

    # Read in the output files
    files = glob.glob(object_name + '/*_PSF.txt')

    # Empty variables
    all_mags  = []
    all_errs  = []
    all_mjds  = []
    all_filts = []

    # For each file import the data
    for i in range(len(files)):
        MJD, PSF_Mag, Aperture_Mag, PSF_Err, Aperture_Err, Filter = np.genfromtxt(files[i], unpack = True, skip_header = 1, dtype = 'str')

        # Save to array
        all_mags  = np.append(all_mags,  Aperture_Mag)
        all_errs  = np.append(all_errs,  Aperture_Err)
        all_mjds  = np.append(all_mjds,  MJD)
        all_filts = np.append(all_filts, Filter)

    # Sort by date
    all_mags  = all_mags[np.argsort(all_mjds.astype(float))]
    all_errs  = all_errs[np.argsort(all_mjds.astype(float))]
    all_filts = all_filts[np.argsort(all_mjds.astype(float))]
    all_mjds  = all_mjds[np.argsort(all_mjds.astype(float))]

    # Remove nans
    do_use = [i not in emptys for i in all_mags]

    # Save output
    array = (all_mjds[do_use], all_mags[do_use], all_errs[do_use], all_filts[do_use])
    np.savetxt('photometry/' + object_name + ".txt", np.transpose(array), fmt = '%s', header = 'MJD Mag Mag_Err Filter')
    print("Saved %s.txt with %s datapoints"%(object_name, len(np.transpose(array))))

def crop_image(image_name, output_name, edge_crop):
    '''
    Crop image and save
    '''

    # If it doesn't exist, remove
    exists   = check_existence(output_name, '', verbose = False)
    if exists:
        os.system('rm ' + output_name)

    # Read data and crop
    header_data = fits.open(image_name)
    xmin, xmax, ymin, ymax = edge_crop, header_data[0].data.shape[0] - edge_crop, edge_crop, header_data[0].data.shape[1] - edge_crop
    fits.setval(image_name, 'DATASEC',   value = '[%s:%s,%s:%s]'%(1, xmax-xmin+1, 1, ymax-ymin+1))
    fits.setval(image_name, 'TRIMSEC',   value = '[%s:%s,%s:%s]'%(1, xmax-xmin+1, 1, ymax-ymin+1))
    fits.setval(image_name, 'ORIGSEC',   value = '[%s:%s,%s:%s]'%(1, xmax-xmin+1, 1, ymax-ymin+1))
    fits.setval(image_name, 'CCDSEC',    value = '[%s:%s,%s:%s]'%(1, xmax-xmin+1, 1, ymax-ymin+1))
    iraf.imcopy(input = '%s[%s:%s,%s:%s]'%(image_name, xmin, xmax, ymin, ymax), output = output_name)

def do_photometry(image_name, overwrite = False, gain = '', rdnoise = '', RA = '', DEC = '', airmass = '', mjd = '', color = '', object_name = '', exptime = '', search_radius = 7.0, kron_psf_difference = 0.1, min_mag = 21, max_mag = 16, overexposed_limit = 50000, pix_range = 20, fwhm_guess = 3.5, sigma_l_fwhm = 2.0, sigma_h_fwhm = 2.0, iter_fwhm = 2, sigma_l_back = 2.0, sigma_h_back = 2.0, iter_back = 2, sigma_l_std = 2.0, sigma_h_std = 2.0, iter_std = 2, iter_zeropoint = 1, maximum_separation = 6, threshold = 2.0, initial_snr_cut = 10, initial_sigma_clip = 2.5, pre_calculate_background = False, zmag = 30, varorder = 0, n_iterations = 10, detection_threshold = 2.0, recenter = 'yes', target_recenter = 'yes', max_psf_stars = 15, delete_previous = True, delete_star = '', aperture_prefactor = 1, target_aperture_prefactor = 1, min_zeropoint = 2, max_zeropoint = 2, SNR_zeropoint = 15, zeropoint_sigma = 2.0, plot_aperture = False, threshold_target = 2.0, target_pix_range = 10, target_fwhm_prefactor = 1.0, host_mag = 999, plot_range = 3, sky_annulus = 8.0, coordinate_suffix = '_coords_targ.txt', n_sigma = 2, force_upper_limit = False, background_stars = 5, centering_fwhm = 3.0, image_substracted_file = '', image_frame = '', do_image_subtraction = False, show = False, upper_limit_image = 0.95, upper_limit_template = 0.95, edge_crop = 0, force_download = False, boxsize = 25):
    '''
    Main photometry function.

    Parameters
    -------------
    image_name  : Name of the .fits image to do photometry on. Image needs to have good WCS.
    overwrite   : Overwrite previous run if the files already exist?

    ### File Parameters ###
    gain        : Image Gain
    rdnoise     : Read noise
    RA/DEC      : Coordinates
    airmass     : Airmass
    mjd         : Modified Julian Date
    color       : Filter of the image
    object_name : Name of the target
    exptime     : Exposure time

    ### Parameters for 3PI query
    search_radius       : Search radius in arcminutes
    kron_psf_difference : Only return stars lower than this value
                          to remove the galaxies.

    search_radius: Radius for which to querry 3PI to get the comparison stars.
    min_mag: Dimmest magnitude allowed to be matched in the process
    max_mag: Birghtest stars allowed to be matched in the process
    color: Filter of the data
    pix_range: Search radius for the stars in the catalogue.
    fwhm_guess: Initial guess for the FWHM value, not the value that will be used
                for photometry. That will be determined by fitting and optimization.

    ### Sigma Clipping ###
    sigma_l_fwhm: Low Sigma clipping for FWHM
    sigma_h_fwhm: High Sigma clipping for FWHM
    iter_fwhm: Number of iterations for FWHM sigma clipping
    sigma_l_back: Low Sigma clipping for background
    sigma_h_back: High Sigma clipping for background
    iter_back: Number of iterations for background sigma clipping
    sigma_l_std: Low Sigma clipping for standard deviation of background
    sigma_h_std: High Sigma clipping for standard deviation of background
    iter_std: Number of iterations for standard deviation
             of background sigma clipping

    ### Get positions ###
    threshold: threshold above the background that a star would be detected.
    initial_snr_cut: Minimum signal to noise that a star should have to be accepted
    initial_sigma_clip: Sigma from the average magnitude difference from which to
                remove stars from
    pre_calculate_background: Use the background and sigma from the estimate_fwhm (True)
                              or calculate at each individual section (False)

    ### PSF Photometry ###
    varorder: 1 will have 3 look up tables, 2 will have 6, etc. Set to -1 to 2
    n_iterations: Maximum Number of sky fitting iterations
    recenter: Recenter PSF during fit
    target_recenter: Recenter PSF during target fit
    max_psf_stars: Number of stars to fit the PSF on
    delete_previous: Delete the previous PSF .fits files if found?
    delete_star: Which star to remove from the PSF list? Comma separated values
    image_frame : Fits frame, i.e. '[0]'

    aperture_prefactor: When doing aperture photometry, multiply
                        the best value of FWHM by this number.
    target_aperture_prefactor: Same but for target only

    ### Zeropoint ###
    min_zeropoint: Crop any star that's this amount below the average zeropoint
    SNR_zeropoint: minimum SNR to calculate the zeropoint
    plot_aperture: Plot the results of the zeropoint?
    zeropoint_sigma: Sigma for excluding zeropoint data
    maximum_separation : Maximum separation of stars to allow for zeropoint calculation, in arcsec

    ### Target ###
    threshold_target: threshold for detecting the target
    target_pix_range: Search radius for the target star
    target_fwhm_prefactor: Multiply the targets FWHM by this prefactor if necessary.
    background_stars: Number of background stars to use
    centering_fwhm: Prefactor for the box of the centering algortithm.
    host_mag: array of magnitudes of the host in order g, r, i. If it's only one term
              then use that instead.

    plot_range: Plot the magnitudes plus minus this value
    sky_annulus: Width of sky fitting annulus
    gain: Gain of the image, if '' use the default from header
    rdnoise: Read noise of the image, if '' use the default from header
    RA: Overwrite RA from the header
    DEC: Overwrite DEC from the header
    coordinate_suffix: Suffix for the file with the pixel coordinates in (x, y) of the target

    # Non Detection
    n_sigma: Number of sigmas for which to calculate the
             non-detection
    force_upper_limit: Force the code to calculate the upper limit instead of psf photometry.

    # Image Subtraction
    image_substracted_file : Specify a file that comes from an image subtraction routine.
                            To do photometry with the parameters of the main image but on this one.
    do_image_subtraction   : Generate Hotpnats image and do photometry on that
    show                   : Show image subtraction results
    upper_limits           : Maximum counts in images
    edge_crop              : Crop these many pixels from the edges
    force_download         : Force download a new template?
    boxsize                : Boxsize for PSF fitter
    '''

    # Check that the final output file doesn't exist
    exists = check_existence(image_name[:-5] + '_psf.pdf', '', verbose = False)
    # If the files exist, skip, unless overwrite is true
    if exists:
        if overwrite:
            print("do_photometry -- %s already existed, overwriting."%(image_name[:-5] + '_psf.pdf'))
        else:
            print("do_photometry -- %s already exists, skipping."%(image_name[:-5] + '_psf.pdf'))
            return

    # Extract information from the image
    image_data, wcs_data, coord, parameters_list = loadimage(image_name, gain, rdnoise, RA, DEC, airmass, mjd, color, object_name, exptime, edge_crop)
    color = parameters_list['filter']

    # Do tests before continuing
    if parameters_list['filter'] not in 'grizy':
        print('Filter %s not in 3PI, stopping.'%color)
        return
    if type(wcs_data) != wcs.wcs.WCS:
        print('WCS is not the correct format, stopping.')
        return
    if parameters_list['object'] == '':
        print('Object name not found, stopping.')
        return
    if type(coord) != SkyCoord:
        print('SkyCoord is not the correct format, stopping.')
        return
    if (image_data.shape[0] < 20) or (image_data.shape[1] < 20):
        print('Image too small, stopping.')
        return
    try:
        parameters_list['gain'] + parameters_list['rdnoise'] + parameters_list['airmass'] + parameters_list['mjd'] + parameters_list['exptime']
    except:
        print('One of these [gain, rdnoise, airmass, mjd, exptime] is not a float, stopping.')
        return
    print('Doing photometry for ' + image_name)

    # Check if the catalogue exists
    cat_name = parameters_list['object'] + '.cat'
    exists   = check_existence(cat_name, '', verbose = False)
    # If it doesn't exist, query 3PI and get the stars near the target.
    if exists:
        print("Catalogue File %s already exists, using existing."%(cat_name))
        cat_read = table.Table.read(cat_name, format='ascii', guess=False)
    else:
        cat_read = get3pimags(coord, cat_name, search_radius)
    # If the region is not in 3PI, quit.
    if type(cat_read) == str:
        if cat_read == 'No objects':
            print('No objects found in region, stopping.')
            return

    # Crop the catalogue
    cat_in = crop_3picatalog(cat_read, kron_psf_difference)

    # Estimate the FWHM, Background, and Sigma of the image
    best_fwhm, average_background, background_high, background_low, just_nearby = estimate_fwhm(image_data, cat_in, wcs_data, coord, parameters_list, min_mag, max_mag, background_stars, pix_range, overexposed_limit, fwhm_guess, sigma_l_fwhm, sigma_h_fwhm, iter_fwhm, sigma_l_back, sigma_h_back, iter_back)
    # Get the positions of the stars in the image and save them to the cat_matched 
    # to relate the stars in the image to the stars in the catalogue.
    good_cat = get_positions(image_name, image_data, cat_in, wcs_data, coord, parameters_list, best_fwhm, average_background, background_high, background_low, min_mag, max_mag, pix_range, detection_threshold, initial_sigma_clip, initial_snr_cut)

    # Create Image Subtraction File
    if do_image_subtraction:
        image_substracted_file = generate_hotpants_image(image_name, parameters_list, cat_in, best_fwhm * aperture_prefactor, max_mag, min_mag, show, upper_limit_image, upper_limit_template, edge_crop, force_download, boxsize)
    else:
        image_substracted_file = ''

    # Do PSF Photometry
    psf_mags, psf_mags_err, good_psf_mags = psf_photometry_iraf(image_name, image_data, average_background, background_high, background_low, parameters_list, n_iterations, best_fwhm, pix_range, varorder, sky_annulus, recenter, centering_fwhm, zmag, overexposed_limit, max_psf_stars, delete_star, image_frame)

    # Do Aperture Photometry
    aperture_mags, aperture_mags_err, aperture_sums = aperture_photometry_iraf(image_name, '_coords.txt', parameters_list, n_iterations, best_fwhm * aperture_prefactor, pix_range, varorder, sky_annulus, recenter, centering_fwhm, zmag, overexposed_limit, image_substracted_file, image_frame)

    # Calculate the zeropoint using PSF Photometry
    best_cat = good_cat[good_psf_mags]
    average_zeropoint_psf, zero_std_psf = zeropoint(parameters_list, psf_mags, psf_mags_err, best_cat, min_zeropoint, max_zeropoint, SNR_zeropoint, zeropoint_sigma, iter_zeropoint, maximum_separation, plot_aperture == False)

    # Calculate the zeropoint using Aperture photometry
    average_zeropoint_aperture, zero_std_aperture = zeropoint(parameters_list, aperture_mags, aperture_mags_err, good_cat, min_zeropoint, max_zeropoint, SNR_zeropoint, zeropoint_sigma, iter_zeropoint, maximum_separation, plot_aperture)

    # Find target's position
    mean_target, std_target = target_position(image_name, image_data, wcs_data, coord, coordinate_suffix, parameters_list, target_pix_range, best_fwhm * target_fwhm_prefactor, threshold_target, image_substracted_file, target_fwhm_prefactor)

    if mean_target > -900:
        target_psf_mag, target_psf_mag_err = target_psf(image_name, image_data, wcs_data, coord, coordinate_suffix, mean_target, std_target, parameters_list, n_iterations, best_fwhm * target_fwhm_prefactor, target_pix_range, varorder, sky_annulus, target_recenter, centering_fwhm, zmag, overexposed_limit, image_substracted_file, image_frame)
    else:
        target_psf_mag = -999.0

    # If there was no detection, calculate the upper limit instead
    if (target_psf_mag < -900) or force_upper_limit:
        sigma1, sigma2, sigma3 = upper_limit(parameters_list, best_fwhm * aperture_prefactor, average_zeropoint_aperture, just_nearby, zmag)
                                 
        # Assign the 3 sigma upper limit to the magnitude
        target_psf_mag     = sigma3
        target_psf_mag_err = -1.0

    # Do Aperture photometry on the target
    target_aperture_mag, target_aperture_mag_err, target_aperture_sum = aperture_photometry_iraf(image_name, '_recentered.txt', parameters_list, n_iterations, best_fwhm * target_aperture_prefactor, pix_range, varorder, sky_annulus, 'no', centering_fwhm, zmag, overexposed_limit, image_substracted_file, image_frame)

    # Plot PSF photometry output
    if plot_aperture:
        plot_comparison(parameters_list, target_aperture_mag, target_aperture_mag_err, aperture_mags, aperture_mags_err, average_zeropoint_aperture, zero_std_aperture, good_cat, True,  plot_range)
    else:
        plot_comparison(parameters_list, target_psf_mag,      target_psf_mag_err,      psf_mags,      psf_mags_err,      average_zeropoint_psf,      zero_std_psf,      best_cat, False, plot_range)

    # Plot aperture value if using upper limit
    if target_psf_mag_err == -1.0:
        plt.subplot('335')
        if plot_aperture:
            plt.errorbar(target_aperture_mag - average_zeropoint_aperture, target_aperture_mag - average_zeropoint_aperture, yerr = target_aperture_mag_err, fmt = '.', color = 'r', label = r'Ap. mag = $%s \pm %s$'%(np.around(target_aperture_mag - average_zeropoint_aperture, decimals = 2)[0], np.around(target_aperture_mag_err, decimals = 2)[0]))
        else:
            plt.errorbar(target_aperture_mag - average_zeropoint_psf, target_aperture_mag - average_zeropoint_psf, yerr = target_aperture_mag_err, fmt = '.', color = 'r', label = r'Ap. mag = $%s \pm %s$'%(np.around(target_aperture_mag - average_zeropoint_psf, decimals = 2)[0], np.around(target_aperture_mag_err, decimals = 2)[0]))
        plt.legend(loc = 'upper left', frameon = False)

    plt.savefig(image_name[:-5] + '_psf.pdf', bbox_inches = 'tight', figsize=(16,16))
    plt.clf()

    # Calculate the output PSF and Aperture error
    psf_err_out      = np.sqrt(target_psf_mag_err**2 + zero_std_psf**2)
    aperture_err_out = np.sqrt(target_aperture_mag_err**2 + zero_std_aperture**2)

    # Select host in correct band
    if host_mag == 999:
        host_mag_out = host_mag
    else:
        try:
            if   color == 'g': host_mag_out = host_mag[0]
            elif color == 'r': host_mag_out = host_mag[1]
            elif color == 'i': host_mag_out = host_mag[2]
            else:
                print("Color not find %s, using r"%color)
                host_mag_out = host_mag[1]
        except:
            host_mag_out = host_mag

    # Substract Host magnitude and save the output
    # Substract host magnitude
    output_mag_psf      = substract_host(target_psf_mag - average_zeropoint_psf, host_mag_out)
    output_mag_aperture = substract_host(target_aperture_mag - average_zeropoint_aperture, host_mag_out)

    # Convert to floats
    if type(psf_err_out)         == type(np.array(0)): psf_err_out         = psf_err_out[0]
    if type(aperture_err_out)    == type(np.array(0)): aperture_err_out    = aperture_err_out[0]
    if type(output_mag_psf)      == type(np.array(0)): output_mag_psf      = output_mag_psf[0]
    if type(output_mag_aperture) == type(np.array(0)): output_mag_aperture = output_mag_aperture[0]

    # Save output
    if target_psf_mag_err == -1.0:
        array = np.array([parameters_list['mjd'], target_psf_mag - average_zeropoint_psf, output_mag_aperture, target_psf_mag_err, aperture_err_out, color])
    else:
        array = np.array([parameters_list['mjd'], output_mag_psf, output_mag_aperture, psf_err_out, aperture_err_out, color])

    LC_data = table.Table(data = array, names=('MJD','PSF_Mag','Aperture_Mag','PSF_Err', 'Aperture_Err', 'Filter'), dtype = ('f', 'f', 'f', 'f', 'f', 'S'))
    LC_data.write(image_name[:-5] + '_mag_PSF.txt', format='ascii')