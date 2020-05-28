# redphot
Pipeline to do PSF photometry or image difference on astronomical images

# Setup
You will need pyraf and mastcasjobs installed. You will also need a username and key in your system to query 3PI. redspec will search for this file in your home directory:

```
/Users/username/3PI_key.txt
```

You will also need `mastcasjobs`, it should be installed automatically, but if it doesn't you need to install these two modules:
```	
pip install git+git://github.com/dfm/casjobs@master	
pip install git+git://github.com/rlwastro/mastcasjobs@master	
```	

# Example
Assuming you have your 3PI and TNS keys set up, simply run the `do_photometry` function on the transient of your choice.

```
# Example on how to run:
do_photometry('directory/image.fits') # Basic PSF Photometry
do_photometry('directory/image.fits', do_image_subtraction = True) # Do image subtraction using Hotpants
# If it fails try changing threshold_target or target_fwhm_prefactor, or removing stars from psf model
do_photometry('directory/image.fits', threshold_target = 1.0, target_fwhm_prefactor = 1.0, delete_star = '4,5')
# If the pipeline detects something where there is nothing you can force an upper limit instead
do_photometry('directory/image.fits', force_upper_limit = True)

# Afterwards you can clean up and make the lightcurve
clean_up('directory')
create_lightcurve('directory')
```
