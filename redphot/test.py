def def_instrument():
    from pyraf import iraf
    from pyraf.iraf import noao
    from pyraf.iraf import imred
    from pyraf.iraf import ccdred
    from pyraf.iraf import setinstrument
    setinstrument(instrument = 'direct', review = 'no')
#def_instrument()

import os
workdir = os.getcwd()
import numpy as np
from astropy.io import fits
import glob
from astropy.table import Table
os.chdir(workdir)

def prepare_data(file_directory = 'raw_data/*.fits', instrument = '', crop = False, rotate = False, flip = False, variables = ['DISPERSR', 'FILTER'], break_character = '-', filter_name = 'Spectroscopic2', filterkey = 'FILTER', disperser = 'DISPERSR', data_index = 0, datasec_key = 'DATASEC', biassec_key = 'BIASSEC', rotations = 3, header_index = 0, objname = 'OBJECT', binning=False):
    '''
    Copy the raw science image into directories and rename them to something useful.
    Also crop, rotate, or flip them if specified (not yet implemented)

    Parameters
    ---------------------
    file_directory : Directory where to search for files in glob format, i.e. 'data/*.fits'
    instrument     : Instrument being used, right now it only affects the crop factor.
    crop           : Crop the images?
    rotate         : Rotate the images?
    flip           : Flip the images?
    filter_name    : IMACS = Spectroscopic2
                     Something = Bessell_R2
    disperser      : Name of disperser variable
    data_index     : Where is the data?
    rotations      : 3 means top will be left

    Returns
    ---------------------
    Nothing, saves the .fits files to their correct directories.
    '''

    # Import file names
    Files = sorted(glob.glob(file_directory))

    # Make sure there are files
    if len(Files) == 0:
        print('No Files Found With %s'%file_directory)
        return

    # Prepare each file into the right format
    for i in range(len(Files)):
        # Open File
        print(Files[i])
        File = fits.open(Files[i], ignore_missing_end=True)
        file_kind = File[header_index].header[disperser]
        if filter_name != '':
            filter_kind = File[header_index].header[filterkey]
        else:
            filter_kind = ''
        full_name = File[header_index].header[objname]

        is_it_calibration = np.array([k in full_name for k in ['BIAS', 'bias', 'Bias', 'Flat', 'FLAT', 'flat', 'ZERO']])

        if ('Gri' in file_kind) or (filter_name in filter_kind) or is_it_calibration.any():
            # Get File name
            filename    = Files[i][Files[i].find('/')+1:Files[i].find('.fits')]
            full_name   = File[header_index].header[objname]

            # Break the name in two if there are two words
            name_break  = full_name.find(break_character)
            if name_break not in [-1, 0]:
                object_name = full_name[:name_break]
                type_name   = full_name[1+name_break:].replace(' ', '')
            else:
                object_name = full_name
                type_name   = full_name.replace(' ', '')

            # Get the type of file (arc, flat, object, etc.)
            try:
                try:
                    file_type = File[header_index].header['IMAGETYP']
                except:
                    file_type = File[header_index].header['EXPTYPE']
            except:
                file_type = File[header_index].header['OBSTYPE']

            # If the object is a bias frame, copy that inot the bias folder
            if file_type in ['zero', 'Bias', 'BIAS', 'Zero', 'bias', 'ZERO']:
                print_name     = 'bias'
                directory_name = 'bias'
                if binning:
                    bin_size = File[header_index].header['BINNING']
                    directory_name += f'_{bin_size}'
                if variables[0] != '':
                    for k in range(len(variables)):
                        value            = File[header_index].header[variables[k]].replace(' ', '')
                        print_name      += ' \t ' + value
                        directory_name  += '_' + value

                if len(glob.glob(directory_name)) == 0:
                    os.system('mkdir ' + directory_name)

                os.system('cp %s %s/%s_%s.fits'%(Files[i], directory_name, object_name, filename))

            elif file_type.replace(' ', '') in ['object', 'Object', 'SPECTRUM', 'COMP', 'ARC', 'OBJECT', 'OBJNAME', 'arc', 'LAMPFLAT', 'FLAT', 'Flat', 'flat', 'comp', 'Comp']:
                print_name      = object_name
                directory_name  = object_name
                if binning:
                    bin_size = File[header_index].header['BINNING']
                    directory_name += f'_{bin_size}'

                if variables[0] != '':
                    for k in range(len(variables)):
                        value            = File[header_index].header[variables[k]].replace(' ', '')
                        print_name      += ' \t ' + value
                        directory_name  += '_' + value

                print(print_name)
                print('\n')

                # Make directory with relevant information
                if len(glob.glob(directory_name)) == 0:
                    os.system("mkdir %s"%directory_name)

                # Copy the science file in the directory and rename it to something useful
                os.system('cp %s %s/%s_%s.fits'%(Files[i], directory_name, type_name, filename))

            if file_type.replace(' ', '') in ['zero', 'Bias', 'BIAS', 'Zero', 'bias', 'ZERO', 'object', 'Object', 'SPECTRUM', 'COMP', 'ARC', 'OBJECT', 'OBJNAME', 'arc', 'FLAT', 'LAMPFLAT', 'Flat', 'flat', 'comp', 'Comp']:
                # Once saved, rotate or flip if specified
                if rotate:
                    # Rotate the Data by 90 degrees. Top will be left
                    file_name = '%s/%s_%s.fits'%(directory_name, type_name, filename)
                    fits_file = fits.open(file_name, ignore_missing_end=True)
                    fits_file[data_index].data = np.rot90(fits_file[data_index].data, k = rotations)

                    # Modify the bias and data sec variables
                    biassec = fits_file[data_index].header[biassec_key]
                    bias_coma = biassec.find(',')
                    one_bias = biassec[1:bias_coma]
                    two_bias = biassec[bias_coma+1:-1]
                    out_bias = '[%s,%s]'%(two_bias, one_bias)

                    datasec = fits_file[data_index].header[datasec_key]
                    data_coma = datasec.find(',')
                    one_data = datasec[1:data_coma]
                    two_data = datasec[data_coma+1:-1]
                    out_data = '[%s,%s]'%(two_data, one_data)

                    fits_file.writeto(file_name, overwrite = True)
                    fits.setval(file_name,   biassec_key,  value=out_bias, ext = header_index)
                    fits.setval(file_name,   datasec_key,  value=out_data, ext = header_index)
                    fits.setval(file_name,  'DISPAXIS', value='1', ext = header_index)

                    print('Rotated ' + file_name)

                if crop:
                    # Rotate the Data by 270 degrees. Top will be left
                    if file_type in ['zero', 'Bias', 'BIAS', 'Zero', 'bias', 'ZERO']:
                        crop_name = '%s/%s_%s.fits'%(directory_name, type_name, filename)
                    else:
                        crop_name = '%s/%s_%s.fits'%(directory_name, type_name, filename)

                    # Crop the spectrum image
                    if instrument == '':
                        xmin, xmax = 0, fits_file[data_index].data.shape[0]
                        ymin, ymax = 0, fits_file[data_index].data.shape[1]
                    elif instrument == 'IMACS1':
                        xmin, xmax = 60, 4105
                        ymin, ymax = 300, 800
                    elif instrument == 'IMACS2':
                        xmin, xmax = 65, 2110
                        ymin, ymax = 150, 400
                    elif instrument == 'BlueChannel':
                        xmin, xmax = 5, 2700
                        ymin, ymax = 100, 250
                    elif instrument == 'LDDS3':
                        xmin, xmax = 650, 4095
                        ymin, ymax = 320, 620
                    elif instrument == 'LDDS3_2x2':
                        xmin, xmax = 300, 2047
                        ymin, ymax = 225, 380
                    elif instrument == 'Goodman':
                        xmin, xmax = 65, 2067
                        ymin, ymax = 100, 750
                    elif instrument == 'Kosmos':
                        xmin, xmax = 1, 4096
                        ymin, ymax = 340, 1600

                    # Overwrite if binning is on
                    if binning and (instrument == 'LDDS3'):
                        if bin_size == '1x1':
                            xmin, xmax = 650, 4095
                            ymin, ymax = 320, 620
                        if bin_size == '2x2':
                            xmin, xmax = 300, 2047
                            ymin, ymax = 225, 380

                    # Crop the data
                    fits_file = fits.open(crop_name, ignore_missing_end=True)
                    fits_file[data_index].data = fits_file[data_index].data[ymin:ymax,:]
                    fits_file.writeto(crop_name, overwrite = True)
                    fits.setval(crop_name,   biassec_key,  value='[%s:%s,%s:%s]'%(xmin - xmin + 1, xmax - xmin, ymin - ymin + 1, ymax - ymin), ext = header_index)
                    fits.setval(crop_name,   datasec_key ,  value='[%s:%s,%s:%s]'%(xmin - xmin + 1, xmax - xmin, ymin - ymin + 1, ymax - ymin), ext = header_index)
                    fits.setval(crop_name,  'CCDSEC'    ,  value='[%s:%s,%s:%s]'%(xmin - xmin + 1, xmax - xmin, ymin - ymin + 1, ymax - ymin), ext = header_index)
                    fits.setval(crop_name,  'DISPAXIS', value='1', ext = header_index)
                    print('Cropped ' + crop_name)
