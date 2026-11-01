import numpy as np
import pytest
from scipy.special import erf

# ==============================================================================
# Target Functions (Extracted for standalone testing)
# ==============================================================================

def compute_slanted_edge_mtf(roi, oversample=4):
    roi = np.asarray(roi, dtype=float)
    if np.mean(roi[:, -5:]) < np.mean(roi[:, :5]):
        roi = np.fliplr(roi)
    h, w = roi.shape
    
    grad = np.gradient(roi, axis=1)
    cols = np.arange(w, dtype=float)
    rows = np.arange(h, dtype=float)
    
    init_edges = cols[np.argmax(grad, axis=1)]
    m, b = np.polyfit(rows, init_edges, 1)
    
    centroids = np.zeros(h)
    for r in range(h):
        x_est = m * r + b
        mask = np.abs(cols - x_est) <= 8.0
        w_row = np.clip(grad[r, mask], 0, None)
        if np.sum(w_row) > 0:
            centroids[r] = np.sum(cols[mask] * w_row) / np.sum(w_row)
        else:
            centroids[r] = x_est
            
    m, b = np.polyfit(rows, centroids, 1)
    
    rr, cc = np.meshgrid(rows, cols, indexing="ij")
    dist = (cc - (m * rr + b)) * np.cos(np.arctan(m))
    
    max_d = min(w / 2.0 - 4, 16.0)
    step = 1.0 / oversample
    bins = np.arange(-max_d, max_d + step, step)
    bin_idx = np.digitize(dist.ravel(), bins)
    esf = np.zeros(len(bins) - 1)
    val_flat = roi.ravel()
    for k in range(1, len(bins)):
        mask = (bin_idx == k)
        esf[k - 1] = np.mean(val_flat[mask]) if np.any(mask) else (esf[k - 2] if k > 1 else 0.0)
        
    lsf = np.diff(esf) / step
    lsf_win = lsf * np.hanning(len(lsf))
    
    mtf = np.abs(np.fft.rfft(lsf_win))
    freqs = np.fft.rfftfreq(len(lsf_win), d=step)
    if mtf[0] > 0:
        mtf /= mtf[0]
    sinc_corr = np.sinc(freqs * step)
    mtf = np.divide(mtf, sinc_corr, out=mtf, where=sinc_corr > 0.1)
    
    valid = freqs <= 1.0
    f_v, m_v = freqs[valid], mtf[valid]
    above = np.where(m_v >= 0.5)[0]
    if len(above) > 0 and above[-1] + 1 < len(f_v):
        i0 = above[-1]
        f0, f1 = f_v[i0], f_v[i0 + 1]
        m0, m1 = m_v[i0], m_v[i0 + 1]
        mtf50 = f0 + (0.5 - m0) * (f1 - f0) / (m1 - m0) if m1 != m0 else f0
    else:
        mtf50 = 0.0
    return f_v, m_v, float(mtf50)

def ciede2000(lab1, lab2):
    L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]
    C1 = np.hypot(a1, b1)
    C2 = np.hypot(a2, b2)
    C_bar = 0.5 * (C1 + C2)
    G = 0.5 * (1.0 - np.sqrt((C_bar**7) / (C_bar**7 + 25.0**7)))
    a1_p, a2_p = (1.0 + G) * a1, (1.0 + G) * a2
    C1_p, C2_p = np.hypot(a1_p, b1), np.hypot(a2_p, b2)
    h1_p = np.degrees(np.arctan2(b1, a1_p)) % 360.0
    h2_p = np.degrees(np.arctan2(b2, a2_p)) % 360.0
    dL_p, dC_p = L2 - L1, C2_p - C1_p
    dh_p = h2_p - h1_p
    dh_p = np.where(dh_p > 180.0, dh_p - 360.0, dh_p)
    dh_p = np.where(dh_p < -180.0, dh_p + 360.0, dh_p)
    dh_p = np.where((C1_p * C2_p) == 0.0, 0.0, dh_p)
    dH_p = 2.0 * np.sqrt(C1_p * C2_p) * np.sin(np.radians(dh_p / 2.0))
    L_bar_p, C_bar_p = 0.5 * (L1 + L2), 0.5 * (C1_p + C2_p)
    h_sum = h1_p + h2_p
    h_bar_p = np.where(
        (C1_p * C2_p) == 0.0, h_sum,
        np.where(np.abs(h1_p - h2_p) <= 180.0, 0.5 * h_sum,
                 np.where(h_sum < 360.0, 0.5 * (h_sum + 360.0), 0.5 * (h_sum - 360.0)))
    )
    T = (1.0 - 0.17 * np.cos(np.radians(h_bar_p - 30.0)) + 0.24 * np.cos(np.radians(2.0 * h_bar_p))
         + 0.32 * np.cos(np.radians(3.0 * h_bar_p + 6.0)) - 0.20 * np.cos(np.radians(4.0 * h_bar_p - 63.0)))
    d_theta = 30.0 * np.exp(-(((h_bar_p - 275.0) / 25.0) ** 2))
    R_C = 2.0 * np.sqrt((C_bar_p**7) / (C_bar_p**7 + 25.0**7))
    S_L = 1.0 + (0.015 * ((L_bar_p - 50.0) ** 2)) / np.sqrt(20.0 + ((L_bar_p - 50.0) ** 2))
    S_C = 1.0 + 0.045 * C_bar_p
    S_H = 1.0 + 0.015 * C_bar_p * T
    R_T = -np.sin(np.radians(2.0 * d_theta)) * R_C
    return np.sqrt((dL_p / S_L)**2 + (dC_p / S_C)**2 + (dH_p / S_H)**2 + R_T * (dC_p / S_C) * (dH_p / S_H))

# ==============================================================================
# Pytest Verification Suite
# ==============================================================================

def test_mtf50_analytic_recovery():
    """Verify slanted edge MTF50 algorithm against continuous Gaussian analytic solution."""
    sigma_true = 1.15
    expected_mtf50 = np.sqrt(np.log(2.0)) / (np.sqrt(2.0) * np.pi * sigma_true)
    
    yy, xx = np.mgrid[0:160, 0:160]
    angle_rad = np.deg2rad(5.0)
    dist_true = (xx - (80.0 + np.tan(angle_rad) * (yy - 80.0))) * np.cos(angle_rad)
    synth_edge = 0.5 * (1.0 + erf(dist_true / (np.sqrt(2.0) * sigma_true)))
    
    _, _, meas_mtf50 = compute_slanted_edge_mtf(synth_edge)
    assert abs(meas_mtf50 - expected_mtf50) < 0.01, f"Expected {expected_mtf50:.4f}, got {meas_mtf50:.4f}"

def test_ptc_gain_read_noise_wls():
    """Verify PTC pair-subtraction recovers gain and read noise under spatial PRNU via WLS."""
    rng = np.random.default_rng(513)
    g_true, r_true_e = 0.45, 5.0
    mus, vars_pair = [], []
    prnu_fixed = 1.0 + 0.03 * rng.standard_normal((200, 200))
    
    for e_level in np.geomspace(10, 12000, 10):
        sig_map = e_level * prnu_fixed
        fA = g_true * (rng.poisson(sig_map) + rng.normal(0, r_true_e, sig_map.shape))
        fB = g_true * (rng.poisson(sig_map) + rng.normal(0, r_true_e, sig_map.shape))
        mus.append(0.5 * (np.mean(fA) + np.mean(fB)))
        vars_pair.append(0.5 * np.var(fA - fB, ddof=1))

    weights = 1.0 / np.asarray(vars_pair)
    g_fit, v0_fit = np.polyfit(mus, vars_pair, 1, w=weights)
    r_fit_e = np.sqrt(max(v0_fit, 0)) / g_fit
    
    assert abs(g_fit - g_true) / g_true < 0.03, f"Gain failed: got {g_fit:.4f}, expected {g_true:.4f}"
    assert abs(r_fit_e - r_true_e) / r_true_e < 0.08, f"Read Noise failed: got {r_fit_e:.2f}, expected {r_true_e:.2f}"

def test_ccm_exact_inversion():
    """Verify 3x3 Least-Squares exact channel mixing inversion."""
    rng = np.random.default_rng(513)
    ref_rgb = rng.uniform(0.05, 0.9, (24, 3))
    mix_mat = np.array([[0.8, 0.15, 0.05], [0.1, 0.85, 0.05], [0.05, 0.15, 0.8]])
    cam_rgb = ref_rgb @ mix_mat
    ccm_recovered, _, _, _ = np.linalg.lstsq(cam_rgb, ref_rgb, rcond=None)
    max_err = np.max(np.abs((cam_rgb @ ccm_recovered) - ref_rgb))
    assert max_err < 1e-12, f"CCM inversion failed with max residual {max_err}"

def test_ciede2000_identity_symmetry():
    """Verify zero-identity and symmetry axioms of CIEDE2000 formula."""
    lab_a = np.array([[50.0, 25.0, -15.0], [75.0, -10.0, 40.0]])
    lab_b = np.array([[52.0, 22.0, -12.0], [70.0, -14.0, 35.0]])
    assert np.allclose(ciede2000(lab_a, lab_a), 0.0), "Zero-identity failed"
    assert np.allclose(ciede2000(lab_a, lab_b), ciede2000(lab_b, lab_a)), "Symmetry failed"