import os
import io
import math
import argparse
import requests

from PIL import Image, ImageDraw, ImageFont
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord, get_constellation
from astropy.coordinates import Angle, Latitude, Longitude  # Angles
import numpy as np

# directory where the fits files and jpg files are stored
astrobase_backend_host = "https://woodland.uilennest.net"
DATA_HOST = f'{astrobase_backend_host}/astrobase/data'


def get_world_coordinate_system(path_to_fits_file):
    """
    read the word coordnates from the header of the fits file
    @:param path_to_fits_file
    """

    hdu = fits.open(path_to_fits_file)[0]

    # retrieve some settings from the header
    # First remove the 3rd axis from the header (RGB slices) because WCS works only in 2D.
    my_header = hdu.header
    my_header['NAXIS'] = 2

    width_in_pixels = my_header['NAXIS1']
    height_in_pixels = my_header['NAXIS2']



    ra_reference = my_header['CRVAL1']
    dec_reference = my_header['CRVAL2']
    x_reference_pixel = my_header['CRPIX1']
    y_reference_pixel = my_header['CRPIX2']

    wcs = WCS(my_header, naxis=2)

    c1 = wcs.pixel_to_world(0, 0)
    c2 = wcs.pixel_to_world(width_in_pixels - 1, 0)
    c3 = wcs.pixel_to_world(0, height_in_pixels - 1)

    width_in_degrees = c1.separation(c2).deg
    height_in_degrees = c1.separation(c3).deg

    ny, nx, nz = wcs.pixel_shape  # (Naxis2, Naxis1)

    # Pixel coords of the 4 corners (0-based, use origin=0 in pixel_to_world)
    corners_pix = np.array([
        [0, 0],
        [nx - 1, 0],
        [nx - 1, ny - 1],
        [0, ny - 1],
    ])
    sky_coords_box = wcs.pixel_to_world(corners_pix[:, 0], corners_pix[:, 1])
    for c in sky_coords_box:
        print(c.ra.deg, c.dec.deg)

    return wcs, sky_coords_box, width_in_pixels, height_in_pixels, width_in_degrees, height_in_degrees


def rotate_and_resize(wcs, sky_coords_box_image, width_in_pixels, height_in_pixels, width_in_degrees, height_in_degrees, image, x_pixels_per_degree_on_background, y_pixels_per_degree_on_background):

    # get the bounding box and central point in pixels and world coordinates
    sky0 = sky_coords_box_image[0]
    sky1 = sky_coords_box_image[1]
    sky2 = sky_coords_box_image[2]
    sky3 = sky_coords_box_image[3]

    x_pixels_per_degree_on_image = width_in_pixels / width_in_degrees
    y_pixels_per_degree_on_image = height_in_pixels / height_in_degrees

    d_pixels = math.hypot(width_in_pixels,height_in_pixels)

    d_radec = math.hypot(width_in_degrees,height_in_degrees)

    pixels_per_degree = d_pixels / d_radec


    # rotate the image
    sky1 = SkyCoord(Longitude([0], unit=u.deg), Latitude([search_dec], unit=u.deg), frame='icrs')
    sky2 = SkyCoord(Longitude([1], unit=u.deg), Latitude([search_dec], unit=u.deg), frame='icrs')
    x1, y1 = wcs.world_to_pixel(sky1)
    x2, y2 = wcs.world_to_pixel(sky2)

    dx = x1 - x2
    dy = y2 - y1
    rotation = math.degrees(math.atan(dy / dx))
    im_new = im_new.rotate(-rotation)

    # resize to a common size
    im_new = im_new.resize((size_in_pixels,size_in_pixels))

    # save result
    os.makedirs(os.path.dirname(path_to_output_image_file), exist_ok=True)
    path_to_new_file = path_to_output_image_file
    im_new.save(path_to_new_file)


def display_fits_on_background(fits_file, url_to_jpg, background_jpg):

    # retrieve the metadata from the fits

    wcs, sky_coords_box_image, width_in_pixels, height_in_pixels, width_in_degrees, heigth_in_degrees = get_world_coordinate_system(fits_file)

    # retrieve the jpg
    try:
        response = requests.get(url_to_jpg)
        response.raise_for_status()
        image = Image.open(io.BytesIO(response.content))
    except:
        raise (Exception("ERROR: " + url_to_jpg + ' not found'))

    image_new = image.copy()

    draw = ImageDraw.Draw(image_new)
    #im_new.save("astromap.jpg")
    #im_new.show()

    # open the background image and determine its size
    background_image = Image.open(background_jpg)
    print(f'background = {background_jpg}, size = {background_image.size}')

    x_pixels_per_degree_on_background = background_image.size[0]/360
    y_pixels_per_degree_on_background = background_image.size[1]/180

    rotate_and_resize(wcs, sky_coords_box_image, width_in_pixels, height_in_pixels, width_in_degrees, heigth_in_degrees, image_new, x_pixels_per_degree_on_background, y_pixels_per_degree_on_background)
    #background_image.show()

    # do the magic

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="astromap")
    parser.add_argument("--jpg", type=str, help= "original processed image",
                        default="https://woodland.uilennest.net/astrobase/data/240213005/240213005_raw.jpg")
    parser.add_argument("--fits", type=str, help="plate solved fits file containing the wcs of the original jpg",
                        default="https://woodland.uilennest.net/astrobase/data/240213005/9934006.fits")
    parser.add_argument("--background", type=str, help="background image on which the jpg's are projects",
                        default="background/noirlab2430b.jpg")
    args = parser.parse_args()

    # jpg is the original (processed) image
    # fits it its 'plate solved' counterpart fits file containing the world coordinate system.
    fits = args.fits
    jpg = args.jpg
    background = args.background

    print("=" * 55)
    print("  AstroMap")
    print("=" * 55)


    display_fits_on_background(fits, jpg, background)

if __name__ == "__main__":
    main()
