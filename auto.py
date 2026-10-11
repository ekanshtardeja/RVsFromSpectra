import sys
import csv
from pathlib import Path
import requests
import numpy as np
from scipy.optimize import curve_fit

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astroquery.eso import Eso

# ==============================================================================
# BLOCK 1: USER INPUTS & INITIAL SETUP
# ==============================================================================
out_dir = Path("./stellar_data")
out_dir.mkdir(exist_ok=True)

csv_filename = out_dir / "stellar_rv_results.csv"

# 1. Ask for absorption lines and wavelengths
num_lines = int(input("How many absorption lines do you want to analyze? "))
target_wavelengths = []
for idx in range(num_lines):
    wl = float(input(f"  Enter rest wavelength for line {idx + 1} (Å): "))
    target_wavelengths.append(wl)

# 2. Ask for star name
star_name = input("\nEnter star name (e.g., Tau Ceti or HD 10700): ").strip()

print(f"\n[INFO] Resolving coordinates for '{star_name}' via SIMBAD/NED...")
try:
    target_coord = SkyCoord.from_name(star_name)
except Exception as e:
    print(f"[ERROR] Could not resolve object name: {e}")
    sys.exit(1)

print(f"Target: {star_name}")
print(f"  RA  (deg) : {target_coord.ra.deg:.6f}° ({target_coord.ra.to_string(unit=u.hour, sep=':')})")
print(f"  Dec (deg) : {target_coord.dec.deg:.6f}° ({target_coord.dec.to_string(unit=u.degree, sep=':')})")

# ==============================================================================
# BLOCK 2: ESO ARCHIVE QUERY (FETCH ALL SPECTRA)
# ==============================================================================
search_radius = 5 * u.arcmin
print(f"\n[INFO] Querying ESO Archive (HARPS) within {search_radius}...")

eso = Eso()
table = eso.query_surveys(
    surveys="HARPS",
    cone_ra=target_coord.ra.deg,
    cone_dec=target_coord.dec.deg,
    cone_radius=search_radius.to_value(u.deg)
)

if table is None or len(table) == 0:
    print("[ERROR] No observations found within 5 arcminutes.")
    sys.exit(0)

# Sort table by release/publication date in descending order
if 'dp_pub_date' in table.colnames:
    table.sort('dp_pub_date', reverse=True)

total_spectra = len(table)
print(f"[INFO] Found {total_spectra} observation(s) to process.\n")

# ==============================================================================
# HELPER FUNCTIONS: GAUSSIAN MODEL & FWHM ESTIMATOR
# ==============================================================================
def gaussian(x, amplitude, center, sigma, baseline):
    """Gaussian Spectral Line Model"""
    return baseline + amplitude * np.exp(-((x - center) ** 2) / (2 * sigma ** 2))

def estimate_fwhm_absorption(norm_flux, wave_array):
    """Estimates FWHM from absorption feature via linear interpolation."""
    norm_flux = np.asarray(norm_flux)
    wave_array = np.asarray(wave_array)

    base = 1.0
    i0 = np.nanargmin(norm_flux)
    dip = norm_flux[i0]

    y_half = base - 0.5 * (base - dip)

    iL = None
    for i in range(i0, 0, -1):
        if norm_flux[i] <= y_half and norm_flux[i - 1] > y_half:
            iL = i
            break

    iR = None
    for i in range(i0, len(norm_flux) - 1):
        if norm_flux[i] <= y_half and norm_flux[i + 1] > y_half:
            iR = i
            break

    if iL is None or iR is None:
        return np.nan

    waveL = np.interp(y_half, [norm_flux[iL], norm_flux[iL - 1]], [wave_array[iL], wave_array[iL - 1]])
    waveR = np.interp(y_half, [norm_flux[iR], norm_flux[iR + 1]], [wave_array[iR], wave_array[iR + 1]])

    return waveR - waveL

def extract_time_of_capture(header):
    """Extracts BJD, MJD, or ISO Timestamp from FITS header."""
    for key in ['BJD', 'HIERARCH ESO DRS BJD', 'MJD-OBS', 'DATE-OBS', 'HJD']:
        if key in header:
            return key, header[key]
    return 'UNKNOWN', 'N/A'

# ==============================================================================
# BLOCK 3: AUTOMATED DOWNLOAD, PROCESSING, AND DATA COLLECTION
# ==============================================================================
c_kms = 299792.458  # Speed of light in km/s
all_records = []

for spec_idx in range(total_spectra):
    dp_id = table['dp_id'][spec_idx]
    rel_date = table['dp_pub_date'][spec_idx] if 'dp_pub_date' in table.colnames else 'N/A'
    
    print("=" * 70)
    print(f"[{spec_idx + 1}/{total_spectra}] Processing Dataset ID: {dp_id} (Date: {rel_date})")
    print("=" * 70)

    # 1. Download FITS file if not present locally
    url = f"https://dataportal.eso.org/dataPortal/file/{dp_id}"
    fits_filename = out_dir / f"{dp_id.replace(':', '-')}.fits"

    if not fits_filename.exists():
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open(fits_filename, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        except Exception as e:
            print(f"[ERROR] Failed to download {dp_id}: {e}")
            continue

    # 2. Extract Wavelength, Flux, and Time of Capture from Header
    try:
        with fits.open(fits_filename) as hdul:
            data_hdu = next(hdu for hdu in hdul if isinstance(hdu, fits.BinTableHDU))
            raw_wave = data_hdu.data["WAVE"]
            raw_flux = data_hdu.data["FLUX"]
            
            # Read primary and extension headers for timestamp
            hdr = hdul[0].header
            hdr_ext = data_hdu.header
            
            time_type, time_val = extract_time_of_capture(hdr_ext)
            if time_val == 'N/A':
                time_type, time_val = extract_time_of_capture(hdr)

            if raw_wave.ndim > 1:
                wave = np.asarray(raw_wave[0], dtype=float)
                flux = np.asarray(raw_flux[0], dtype=float)
            else:
                wave = np.asarray(raw_wave, dtype=float)
                flux = np.asarray(raw_flux, dtype=float)
    except Exception as e:
        print(f"[ERROR] Could not read FITS file {fits_filename}: {e}")
        continue

    # Filter valid flux and wavelength values
    mask_valid = np.isfinite(wave) & np.isfinite(flux) & (flux > 0)
    wave = wave[mask_valid]
    flux = flux[mask_valid]

    spectrum_rvs = []
    spectrum_errs = []

    # 3. Fit each requested line
    for lambda_rest in target_wavelengths:
        lower_wave = lambda_rest - 1.0
        upper_wave = lambda_rest + 1.0
        region_mask = (wave >= lower_wave) & (wave <= upper_wave)
        sel_wave = wave[region_mask]
        sel_flux = flux[region_mask]

        if len(sel_wave) < 5:
            print(f"  [WARNING] Skipping line {lambda_rest:.2f} Å: region out of range or insufficient data.")
            continue

        # Continuum Normalization
        core_mask = (sel_wave < np.percentile(sel_wave, 25)) | (sel_wave > np.percentile(sel_wave, 75))
        poly_coeff = np.polyfit(sel_wave[core_mask], sel_flux[core_mask], deg=1)
        continuum_fit = np.polyval(poly_coeff, sel_wave)
        norm_flux = sel_flux / continuum_fit

        # Initial Guesses
        init_center = sel_wave[np.argmin(norm_flux)]
        init_amplitude = np.min(norm_flux) - 1.0
        init_baseline = 1.0

        est_fwhm = estimate_fwhm_absorption(norm_flux, sel_wave)
        if not np.isnan(est_fwhm) and est_fwhm > 0:
            init_sigma = est_fwhm / (2 * np.sqrt(2 * np.log(2)))
        else:
            init_sigma = 0.5

        p0 = [init_amplitude, init_center, init_sigma, init_baseline]
        bounds = ([-np.inf, sel_wave.min(), 1e-4, 0.5], [0.0, sel_wave.max(), np.inf, 1.5])

        # Curve Fitting
        try:
            popt, pcov = curve_fit(gaussian, sel_wave, norm_flux, p0=p0, bounds=bounds)
            perr = np.sqrt(np.diag(pcov))
        except Exception as e:
            print(f"  [ERROR] Fit failed for line {lambda_rest:.2f} Å: {e}")
            continue

        fit_amp, fit_center, fit_sigma, fit_base = popt
        err_amp, err_center, err_sigma, err_base = perr

        fwhm = 2.35482 * abs(fit_sigma)
        fwhm_err = 2.35482 * err_sigma

        # Radial Velocity Calculation
        rv_kms = c_kms * (fit_center - lambda_rest) / lambda_rest
        rv_err = c_kms * (err_center / lambda_rest)

        spectrum_rvs.append(rv_kms)
        spectrum_errs.append(rv_err)

        # Append record for this specific line
        all_records.append({
            'dataset_id': dp_id,
            'star_name': star_name,
            'time_type': time_type,
            'obs_time': time_val,
            'rest_wavelength_angstrom': lambda_rest,
            'fitted_center_angstrom': fit_center,
            'rv_kms': rv_kms,
            'rv_err_kms': rv_err,
            'fwhm_angstrom': fwhm,
            'fwhm_err_angstrom': fwhm_err
        })

        print(f"  Line {lambda_rest:.2f} Å | Time ({time_type}): {time_val} | RV: {rv_kms:+.2f} ± {rv_err:.2f} km/s")

    if spectrum_rvs:
        mean_rv = np.mean(spectrum_rvs)
        std_rv = np.std(spectrum_rvs) if len(spectrum_rvs) > 1 else 0.0
        print(f"  --> Spectrum Mean RV: {mean_rv:+.2f} ± {std_rv:.2f} km/s")

# ==============================================================================
# BLOCK 4: WRITE TO CSV FILE
# ==============================================================================
csv_headers = [
    'dataset_id', 
    'star_name', 
    'time_type', 
    'obs_time', 
    'rest_wavelength_angstrom', 
    'fitted_center_angstrom', 
    'rv_kms', 
    'rv_err_kms', 
    'fwhm_angstrom', 
    'fwhm_err_angstrom'
]

try:
    with open(csv_filename, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_headers)
        writer.writeheader()
        writer.writerows(all_records)
    print("\n" + "=" * 70)
    print(f"[SUCCESS] Saved {len(all_records)} line fit record(s) to CSV:")
    print(f"          Path: {csv_filename.resolve()}")
    print("=" * 70)
except Exception as e:
    print(f"\n[ERROR] Failed to save CSV file: {e}")