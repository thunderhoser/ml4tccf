"""Code for parallax correction, taken from the following source:

https://github.com/pytroll/satpy/blob/main/satpy/modifiers/parallax.py
"""

import datetime
import numpy as np

F = 1 / 298.257223563  # Earth flattening WGS-84
EARTH_RADIUS = 6378.137  # WGS84 Equatorial radius
MFACTOR = 7.292115E-5


def _get_satellite_elevation(sat_lon, sat_lat, sat_alt, lon, lat):
    """Get satellite elevation.

    Get the satellite elevation from satellite lon/lat/alt for positions
    lon/lat.
    """
    placeholder_date = datetime.datetime(2000, 1, 1)  # no impact on get_observer_look?
    (_, elevation) = get_observer_look(
        sat_lon, sat_lat, sat_alt/1e3,  # m → km (wanted by get_observer_look)
        placeholder_date, lon, lat, 0)
    return elevation


def _calculate_slant_cloud_distance(height, elevation):
    """Calculate slant cloud to ground distance.

    From (cloud top) height and satellite elevation, calculate the
    slant cloud-to-ground distance along the line of sight of the satellite.
    """
    if np.isscalar(elevation) and elevation == 0:
        raise NotImplementedError(
            "Parallax correction not implemented for "
            "satellite elevation 0")
    if np.isscalar(elevation) and elevation < 0:
        raise ValueError(
            "Satellite is below the horizon.  Cannot calculate parallax "
            "correction.")
    return height / np.sin(np.deg2rad(elevation))


def _get_parallax_shift_xyz(sat_lon, sat_lat, sat_alt, lon, lat, parallax_distance):
    """Calculate the parallax shift in cartesian coordinates.

    From satellite position and cloud position, get the parallax shift in
    cartesian coordinates:

    Args:
        sat_lon (number): Satellite longitude in geodetic coordinates [degrees]
        sat_lat (number): Satellite latitude in geodetic coordinates [degrees]
        sat_alt (number): Satellite altitude above the Earth surface [m]
        lon (array or number): Longitudes of pixel or pixels to be corrected,
            in geodetic coordinates [degrees]
        lat (array or number): Latitudes of pixel/pixels to be corrected, in
            geodetic coordinates [degrees]
        parallax_distance (array or number): Cloud to ground distance with parallax
            effect [m].

    Returns:
        Parallax shift in cartesian coordinates in meter.
    """
    sat_xyz = np.hstack(lonlat2xyz(sat_lon, sat_lat)) * sat_alt
    cth_xyz = np.stack(lonlat2xyz(lon, lat), axis=-1) * EARTH_RADIUS*1e3  # km → m
    delta_xyz = cth_xyz - sat_xyz
    sat_distance = np.sqrt((delta_xyz*delta_xyz).sum(axis=-1))
    dist_shape = delta_xyz.shape[:-1] + (1,)  # force correct array broadcasting
    return cth_xyz - delta_xyz*(parallax_distance/sat_distance).reshape(dist_shape)


def lonlat2xyz(lon, lat):
    """Convert lon lat to cartesian.

    For a sphere with unit radius, convert the spherical coordinates
    longitude and latitude to cartesian coordinates.

    Args:
        lon (number or array of numbers): Longitude in °.
        lat (number or array of numbers): Latitude in °.

    Returns:
        (x, y, z) Cartesian coordinates [1]
    """
    lat = np.deg2rad(lat)
    lon = np.deg2rad(lon)
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)
    return x, y, z


def xyz2lonlat(x, y, z, asin=False):
    """Convert cartesian to lon lat.

    For a sphere with unit radius, convert cartesian coordinates to spherical
    coordinates longitude and latitude.

    Args:
        x (number or array of numbers): x-coordinate, unitless
        y (number or array of numbers): y-coordinate, unitless
        z (number or array of numbers): z-coordinate, unitless
        asin (optional, bool): If true, use arcsin for calculations.
            If false, use arctan2 for calculations.

    Returns:
        (lon, lat): Longitude and latitude in °.
    """
    lon = np.rad2deg(np.arctan2(y, x))
    if asin:
        lat = np.rad2deg(np.arcsin(z))
    else:
        lat = np.rad2deg(np.arctan2(z, np.sqrt(x ** 2 + y ** 2)))
    return lon, lat


def _days(dt):
    """Get the days (floating point) from *d_t*.
    """
    return dt / np.timedelta64(1, 'D')


def dt2np(utc_time):
    try:
        return np.datetime64(utc_time)
    except ValueError:
        return utc_time.astype('datetime64[ns]')


def jdays2000(utc_time):
    """Get the days since year 2000.
    """
    return _days(dt2np(utc_time) - np.datetime64('2000-01-01T12:00'))


def gmst(utc_time):
    """Greenwich mean sidereal utc_time, in radians.

    As defined in the AIAA 2006 implementation:
    http://www.celestrak.com/publications/AIAA/2006-6753/
    """
    ut1 = jdays2000(utc_time) / 36525.0
    theta = 67310.54841 + ut1 * (876600 * 3600 + 8640184.812866 + ut1 *
                                 (0.093104 - ut1 * 6.2 * 10e-6))
    return np.deg2rad(theta / 240.0) % (2 * np.pi)


def observer_position(time, lon, lat, alt):
    """Calculate observer ECI position.

    http://celestrak.com/columns/v02n03/
    """

    lon = np.deg2rad(lon)
    lat = np.deg2rad(lat)

    theta = (gmst(time) + lon) % (2 * np.pi)
    c = 1 / np.sqrt(1 + F * (F - 2) * np.sin(lat)**2)
    sq = c * (1 - F)**2

    achcp = (EARTH_RADIUS * c + alt) * np.cos(lat)
    x = achcp * np.cos(theta)  # kilometers
    y = achcp * np.sin(theta)
    z = (EARTH_RADIUS * sq + alt) * np.sin(lat)

    vx = -MFACTOR * y  # kilometers/second
    vy = MFACTOR * x
    vz = 0

    return (x, y, z), (vx, vy, vz)


def get_observer_look(sat_lon, sat_lat, sat_alt, utc_time, lon, lat, alt):
    """Calculate observers look angle to a satellite.
    http://celestrak.com/columns/v02n02/

    :utc_time: Observation time (datetime object)
    :lon: Longitude of observer position on ground in degrees east
    :lat: Latitude of observer position on ground in degrees north
    :alt: Altitude above sea-level (geoid) of observer position on ground in km

    :return: (Azimuth, Elevation)
    """
    (pos_x, pos_y, pos_z), (vel_x, vel_y, vel_z) = observer_position(
        utc_time, sat_lon, sat_lat, sat_alt)

    (opos_x, opos_y, opos_z), (ovel_x, ovel_y, ovel_z) = \
        observer_position(utc_time, lon, lat, alt)

    lon = np.deg2rad(lon)
    lat = np.deg2rad(lat)

    theta = (gmst(utc_time) + lon) % (2 * np.pi)

    rx = pos_x - opos_x
    ry = pos_y - opos_y
    rz = pos_z - opos_z

    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    top_s = sin_lat * cos_theta * rx + \
            sin_lat * sin_theta * ry - cos_lat * rz
    top_e = -sin_theta * rx + cos_theta * ry
    top_z = cos_lat * cos_theta * rx + \
            cos_lat * sin_theta * ry + sin_lat * rz

    # Azimuth is undefined when elevation is 90 degrees, 180 (pi) will be returned.
    az_ = np.arctan2(-top_e, top_s) + np.pi
    az_ = np.mod(az_, 2 * np.pi)  # Needed on some platforms

    rg_ = np.sqrt(rx * rx + ry * ry + rz * rz)

    top_z_divided_by_rg_ = top_z / rg_

    # Due to rounding top_z can be larger than rg_ (when el_ ~ 90).
    top_z_divided_by_rg_ = top_z_divided_by_rg_.clip(max=1)
    el_ = np.arcsin(top_z_divided_by_rg_)

    return np.rad2deg(az_), np.rad2deg(el_)


def get_parallax_corrected_lonlats(sat_lon, sat_lat, sat_alt, lon, lat, height):
    """Calculate parallax corrected lon/lats.

    Satellite geolocation generally assumes an unobstructed view of a smooth
    Earth surface.  In reality, this view may be obstructed by clouds or
    mountains.

    If the view of a pixel at location (lat, lon) is blocked by a cloud
    at height h, this function calculates the (lat, lon) coordinates
    of the cloud above/in front of the invisible surface.

    For scenes that are only partly cloudy, the user might set the cloud top
    height for clear-sky pixels to NaN.  This function will return a corrected
    lat/lon as NaN as well.  The user can use the original lat/lon for those
    pixels or use the higher level :class:`ParallaxCorrection` class.

    This function assumes a spherical Earth.

    .. note::

        Be careful with units!  This code expects ``sat_alt`` and
        ``height`` to be in meter above the Earth's surface.  You may
        have to convert your input correspondingly.  Cloud Top Height
        is usually reported in meters above the Earth's surface, rarely
        in km.  Satellite altitude may be reported in either m or km, but
        orbital parameters are usually in relation to the Earth's centre.
        The Earth radius from pyresample is reported in km.

    Args:
        sat_lon (number): Satellite longitude in geodetic coordinates [degrees]
        sat_lat (number): Satellite latitude in geodetic coordinates [degrees]
        sat_alt (number): Satellite altitude above the Earth surface [m]
        lon (array or number): Longitudes of pixel or pixels to be corrected,
            in geodetic coordinates [degrees]
        lat (array or number): Latitudes of pixel/pixels to be corrected, in
            geodetic coordinates [degrees]
        height (array or number): Heights of pixels on which the correction
            will be based.  Typically this is the cloud top height. [m]

    Returns:
        tuple[float, float]: Corrected geolocation
            Corrected geolocation ``(lon, lat)`` in geodetic coordinates for
            the pixel(s) to be corrected. [degrees]
    """
    elevation = _get_satellite_elevation(sat_lon, sat_lat, sat_alt, lon, lat)
    parallax_distance = _calculate_slant_cloud_distance(height, elevation)
    shifted_xyz = _get_parallax_shift_xyz(
        sat_lon, sat_lat, sat_alt, lon, lat, parallax_distance)

    return xyz2lonlat(
        shifted_xyz[..., 0], shifted_xyz[..., 1], shifted_xyz[..., 2])
