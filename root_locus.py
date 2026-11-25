from fractions import Fraction
from functools import lru_cache
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
from numpy.polynomial import Polynomial
from numpy.polynomial.polynomial import polyder

# https://eng.libretexts.org/Courses/California_State_Polytechnic_University_Humboldt/Measurements_Instrumentation_and_Controls/Chapter_12%3A_Root_Locus
# Characteristic equation D(s) + k*N(s) = 0
# k: gain

def trim_polynomial(coefficients):
    values = list(coefficients)
    while len(values) > 1 and values[-1] == 0: values.pop()
    return values

def div_mod_polynomial(numerator, denominator):
    remainder = trim_polynomial(numerator)
    denominator = trim_polynomial(denominator)
    if denominator == [0]: raise ZeroDivisionError("Zero polynomial divisor")
    
    quotient = [Fraction(0)] * max(1, len(remainder) - len(denominator) + 1)
    while remainder != [0] and len(remainder) >= len(denominator):
        offset = len(remainder) - len(denominator)
        factor = remainder[-1] / denominator[-1]
        quotient[offset] = factor
        for i, coefficient in enumerate(denominator): remainder[i + offset] -= factor * coefficient
        remainder = trim_polynomial(remainder)
    return trim_polynomial(quotient), remainder

def quotient_divisable(numerator, denominator):
    quotient, remainder = div_mod_polynomial(numerator, denominator)
    if remainder != [0]: raise ArithmeticError("Numerator is not divisable by denominator")
    return quotient

def to_monic(poly):
    poly = trim_polynomial(poly)
    if poly == [0]: return poly
    return [value / poly[-1] for value in poly]

def gcd_polynomial(first, second):
    first, second = to_monic(first), to_monic(second)
    while second != [0]:
        _, remainder = div_mod_polynomial(first, second)
        first, second = second, to_monic(remainder)
    return to_monic(first)

def square_free_factors(poly):
    # Return factors paired with their multiplicities (Yun's method)
    poly = to_monic(poly)
    if len(poly) <= 1: return []
    
    repeated = gcd_polynomial(poly, polyder(poly).tolist())
    remaining = quotient_divisable(poly, repeated)
    
    multiplicity = 1
    factors = []
    while len(remaining) > 1:
        shared = gcd_polynomial(remaining, repeated)
        factor = quotient_divisable(remaining, shared)
        if len(factor) > 1: factors.append((factor, multiplicity))
        remaining = shared
        repeated = quotient_divisable(repeated, shared)
        multiplicity += 1
    return factors

def shift_polynomial(poly, center):
    # Compute poly(t + center) before converting to floating point
    result = [Fraction(0)]
    for coefficient in reversed(poly):
        updated = [Fraction(0)] * (len(result) + 1)
        for index, value in enumerate(result):
            updated[index] += center * value
            updated[index + 1] += value
        updated[0] += coefficient
        result = trim_polynomial(updated)
    return result

def to_float_polynomial(poly):
    try:
        coefficients = np.array([float(value) for value in poly])
    except OverflowError as error:
        raise RuntimeError("Exact factors exceed the floating-point coefficient range") from error
    if not np.isfinite(coefficients).all():
        raise RuntimeError("Factored coefficients are not finite")
    if any(value != 0 and converted == 0 for value, converted in zip(poly, coefficients)):
        raise RuntimeError("Factored coefficients underflow")
    return Polynomial(coefficients)

def exact_roots(poly):
    roots = []
    for factor, multiplicity in square_free_factors(poly):
        degree = len(factor) - 1
        if degree == 1: values = np.array([complex(float(-factor[0] / factor[1]))])
        else:
            center = -factor[-2] / (degree * factor[-1])
            shifted = to_float_polynomial(shift_polynomial(factor, center))
            values = polynomial_roots(shifted) + float(center)
        roots.extend(np.repeat(values, multiplicity))
    return np.asarray(roots, dtype=complex)

@lru_cache(maxsize=32)
def _prepare_exact_model(d_coefficients, n_coefficients):
    d = [Fraction(str(float(value))) for value in d_coefficients]
    n = [Fraction(str(float(value))) for value in n_coefficients]
    common = gcd_polynomial(d, n)
    reduced_d = quotient_divisable(d, common)
    reduced_n = quotient_divisable(n, common)

    repeated_d = gcd_polynomial(d, polyder(d).tolist())
    repeated_n = gcd_polynomial(n, polyder(n).tolist())
    anchor = common if len(common) > 1 else max((repeated_d, repeated_n), key=len)
    center = -anchor[-2] / ((len(anchor) - 1) * anchor[-1]) if len(anchor) > 1 else Fraction(0)
    shifted_d = shift_polynomial(reduced_d, center)
    shifted_n = shift_polynomial(reduced_n, center)
    return (
        to_float_polynomial(shifted_d),
        to_float_polynomial(shifted_n),
        float(center),
        exact_roots(common),
        exact_roots(shifted_d),
        exact_roots(shifted_n),
    )

def calculate_locus(D, N, **samplingOptions):
    # The characteristic equation is factored as C(s)*[Dr(s) + K*Nr(s)], we only sample the remainings
    minimum_gain = float(samplingOptions.get('kMin', 0))
    if not np.isfinite(minimum_gain) or minimum_gain < 0:
        raise ValueError("Minimum gain must be finite and nonnegative")
    moving_d, moving_n, shift, fixed, moving_poles, moving_zeros = _prepare_exact_model(tuple(D.trim().coef), tuple(N.trim().coef))
    if moving_d.degree() == 0 and moving_n.degree() == 0:
        singular_gain = -moving_d.coef[0] / moving_n.coef[0]
        if singular_gain >= minimum_gain: raise ValueError(f"At K = {singular_gain:g}, the characteristic equation is identically zero")
        if not len(fixed): raise ValueError("D and N are constants; there are no roots to plot")
        gains = [minimum_gain, max(1.0, minimum_gain + 1)]
        movingLoci = np.empty((len(gains), 0), dtype=complex)
        moving_breakpoints = []
    else:
        moving_breakpoints = find_breakpoints(moving_d, moving_n)
        gains, cache = compute_roots_gains(
            moving_d, moving_n, moving_poles, moving_zeros,
            moving_breakpoints, **samplingOptions
        )
        movingLoci = build_loci(gains, cache, moving_poles, moving_zeros) + shift

    fixed_loci = np.tile(fixed, (len(gains), 1))
    return {
        'gains': np.asarray(gains),
        'loci': np.hstack((fixed_loci, movingLoci)),
        'poles': np.concatenate((fixed, moving_poles + shift)),
        'zeros': np.concatenate((fixed, moving_zeros + shift)),
        'breakpoints': [(point + shift, gain) for point, gain in moving_breakpoints],
        'fixed_roots': fixed.copy(),
        'shift': shift
    }

def parse(userInput, coefList):
    try:
        temp = list(map(float, userInput.split(',')))
        if (temp[0] != 0 and np.isfinite(temp).all()):
            coefList[:] = temp
            return True
        return False
    except ValueError: return False

def polynomial_roots(poly):
    poly = poly.trim()
    roots = poly.roots()

    def error(values):
        if not len(values): return 0.0
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            denominator = Polynomial(np.abs(poly.coef))(np.abs(values))
            residual = np.abs(poly(values)) / np.maximum(denominator, np.finfo(float).tiny)
        return float(np.max(residual)) if np.isfinite(residual).all() else np.inf

    quality = error(roots)
    if quality > 1e-10:
        alternate = np.roots(poly.coef[::-1])
        alternate_quality = error(alternate)
        if alternate_quality < quality: roots, quality = alternate, alternate_quality
    if quality > 1e-8: raise RuntimeError("Polynomial roots have excessive backward error")
    return roots

def find_roots(D, N, kList, result=None):
    if result is None: result = {}
    for k in kList:
        if k not in result:
            if not np.isfinite(k): raise RuntimeError("Gain is not finite")
            if abs(k) > 1: cEq = D / k + N
            else: cEq = D + k * N

            # At this gain the leading coefficient cancels
            if k != 0 and D.degree() == N.degree() and k == -D.coef[-1] / N.coef[-1]:
                coef = cEq.coef.copy()
                if len(coef) == D.degree() + 1: coef[-1] = 0
                cEq = Polynomial(coef).trim()
            if not np.isfinite(cEq.coef).all(): raise RuntimeError("Characteristic equation coefficients are not finite")
            if not np.any(cEq.coef): raise ValueError(f"At k = {k:g}, the characteristic equation is identically zero")
            result[k] = polynomial_roots(cEq)
    return result

def find_breakpoints(D, N):
    # Breakpoints: D(s) + k*N(s) = 0 && D'(s) + k*N'(s) = 0 --> k = -D(s) / N(s)
    realTol = 1e-10
    result = []
    # Avoid overflow
    scaledD = D / np.max(np.abs(D.coef))
    scaledN = N / np.max(np.abs(N.coef))
    B = scaledD.deriv() * scaledN - scaledD * scaledN.deriv()

    degD = D.degree()
    degN = N.degree()
    if degD == degN and B.degree() == degD + degN - 1:
        coef = B.coef.copy()
        coef[-1] = 0
        B = Polynomial(coef).trim()
    if not np.any(B.coef): return result
    sBreakArr = B.roots()
    for candidate in sBreakArr:
        if not np.isfinite(candidate): continue
        elif not abs(candidate.imag) <= realTol * (1 + abs(candidate.real)): continue
        sB = candidate.real
        denominator = N(sB)
        scale_N = Polynomial(np.abs(N.coef))(abs(sB))
        if not (abs(denominator) > realTol * scale_N): continue
        kB = -D(sB) / denominator
        if (np.isfinite(kB) and kB >= 0): result.append((sB, kB))
    return result

def matching_cost(ref, target):
    ref = np.asarray(ref, dtype=complex)
    target = np.asarray(target, dtype=complex)
    cost = np.abs(target[:, None] - ref[None, :])
    if not cost.size:
        return cost
    ref_sign = np.where(np.abs(ref.imag) > 1e-10 * (1 + np.abs(ref)), np.sign(ref.imag), 0)
    target_sign = np.where(np.abs(target.imag) > 1e-10 * (1 + np.abs(target)), np.sign(target.imag), 0)
    opposite = target_sign[:, None] * ref_sign[None, :] < 0
    penalty = (min(cost.shape) + 1) * (1 + np.max(cost))
    return cost + penalty * opposite

def align(ref, target, rootCount=None):
    if rootCount is None:
        refLen = len(ref)
        if len(target) != refLen: raise ValueError("Root arrays must have the same length")
        elif refLen == 0: raise ValueError("Cannot align null arrays")
        costMat = matching_cost(ref, target)
        optimizedRows, optimizedCols = opt.linear_sum_assignment(costMat) # Hungarian algorithm
        alignedTarget = np.zeros(refLen, dtype=complex)
        alignedTarget[optimizedCols] = target[optimizedRows]
        return alignedTarget
    else:
        # Missing roots are represented by NaN's
        alignedTarget = np.full(rootCount, np.nan + 1j*np.nan, dtype=complex)
        target = np.asarray(target, dtype=complex)
        if not len(target): return alignedTarget
        if ref is None:
            alignedTarget[:len(target)] = target
            return alignedTarget

        validIdx = np.flatnonzero(np.isfinite(ref))
        remaining = np.ones(len(target), dtype=bool)
        if len(validIdx):
            costMat = matching_cost(ref[validIdx], target)
            optimizedRows, optimizedCols = opt.linear_sum_assignment(costMat)
            alignedTarget[validIdx[optimizedCols]] = target[optimizedRows]
            remaining[optimizedRows] = False

        emptyIdx = np.flatnonzero(~np.isfinite(alignedTarget))
        alignedTarget[emptyIdx[:np.count_nonzero(remaining)]] = target[remaining]
        return alignedTarget

def build_loci(kList, rootCache, openPoles, openZeros):
    rootCount = max(len(openPoles), len(openZeros))
    alignedLoci = []
    prevRoots = None
    for k in kList:
        currRoots = openPoles if k == 0 else rootCache[k]
        currRoots = align(prevRoots, currRoots, rootCount)
        alignedLoci.append(currRoots)
        prevRoots = currRoots
    return np.array(alignedLoci, dtype=complex)

def get_plot_scale(openPoles, openZeros, breakpoints):
    sBreaks = [sB for sB, kB in breakpoints]
    allPts = np.concatenate([openPoles, openZeros, np.array(sBreaks, dtype=complex)])
    center = np.mean(allPts)
    span = np.max(np.abs(allPts - center))
    magnitude = np.max(np.abs(allPts))
    if span <= 1e-12 * max(1.0, magnitude): span = max(1.0, 0.1 * magnitude)
    else: span = max(span, 0.1 * magnitude)
    return center, span

def compute_roots_gains(D, N, openPoles, openZeros, breakpoints, maxIterations=100, kMin=0):
    sBreaks = [sB for sB, kB in breakpoints]
    kBreaks = [kB for sB, kB in breakpoints]
    degD = D.degree()
    degN = N.degree()
    if degD == 0 and degN == 0: raise ValueError("D and N are constants")
    if not np.isfinite(kMin) or kMin < 0: raise ValueError("Minimum gain must be finite and nonnegative")
    if degD < degN: # D(s) + k*N(s) = 0 --> N(s) + q*D(s) = 0 for q = 1/k
        qBreakpoints = find_breakpoints(N, D)
        qList, qRoots = compute_roots_gains(N, D, openZeros, openPoles, qBreakpoints, maxIterations=maxIterations, kMin=0)
        rootCache = {}
        for q in qList:
            if q == 0: continue
            with np.errstate(over='ignore', divide='ignore'): k = np.divide(1.0, q)
            if np.isfinite(k) and k >= kMin: rootCache[k] = qRoots[q]
        # Include the starting gain, which is lost when q = 0 is skipped
        find_roots(D, N, [float(kMin)], rootCache)
        kList = sorted(rootCache)
        if not kList: raise ValueError("No sampled gains remain above the minimum gain")
        return kList, rootCache

    DHighestCoef = D.coef[-1]
    NHighestCoef = N.coef[-1]
    r = degD - degN
    center, span = get_plot_scale(openPoles, openZeros, breakpoints)
    viewRadius = 4 * span
    rootCache = {}

    # Find kMax =========================================================================================================================================
    kMax = 1
    k0 = None
    if r > 0:
        pi = np.pi
        centroid = (openPoles.sum() - openZeros.sum()) / r
        phi = np.angle(-NHighestCoef/DHighestCoef)
        asympAngles = [(phi + 2*pi*n) / r for n in range(r)]
        pointsOfInterest = np.concatenate((openPoles, openZeros, sBreaks))

        drawRange = 4 * max(float(np.max(np.abs(pointsOfInterest - centroid))), span)
        farPoints = [centroid + drawRange*np.exp(1j*theta) for theta in asympAngles]

        kApprox = [abs(D(q) / N(q)) for q in farPoints if N(q) != 0]
        validKs = [k for k in kApprox + kBreaks if (np.isfinite(k) and k > 0)]
        if not validKs: raise ValueError("Cannot estimate finite positive gain")
        kMax = max(validKs)
    else: # r == 0
        # For k --> inf, the solutions approach those of N(s) = 0
        convergenceTol = 0.01
        candidateK = 2 * max(kBreaks + [abs(DHighestCoef / NHighestCoef)])
        k0 = -DHighestCoef / NHighestCoef
        candidateK = max(candidateK, 2 * k0)
        for _ in range(maxIterations):
            find_roots(D, N, [candidateK], rootCache)
            candidateSols = rootCache[candidateK]
            if len(candidateSols) != degN:
                candidateK *= 2
                continue
            errorMatrix = np.abs(candidateSols[:, np.newaxis] - openZeros)
            optimizedRows, optimizedCols = opt.linear_sum_assignment(errorMatrix)
            matchedErrors = errorMatrix[optimizedRows, optimizedCols]
            if matchedErrors.max() < convergenceTol:
                kMax = candidateK
                break
            candidateK *= 2
        else: raise RuntimeError("Maximum iterations reached without convergence")

    if r > 0 and degN > 0:
        convergenceTol = 0.005 * span
        candidateK = kMax
        for _ in range(maxIterations):
            find_roots(D, N, [candidateK], rootCache)
            candidateSols = rootCache[candidateK]
            errorMatrix = np.abs(candidateSols[:, np.newaxis] - openZeros)
            optimizedRows, optimizedCols = opt.linear_sum_assignment(errorMatrix)
            matchedErrors = errorMatrix[optimizedRows, optimizedCols]
            if matchedErrors.max() <= convergenceTol:
                kMax = candidateK
                break
            if not np.isfinite(candidateK * 2): break
            candidateK *= 2
        # Fall back to asymptote-based kMax if convergence cannot be reached

    if kMax <= kMin: kMax = max(1.0, 2 * kMin)
    # =======================================================================================================================================

    # Adaptive sampling =====================================================================================================================
    # Local spatial scale
    interest = np.concatenate((openPoles, openZeros))
    magnitudes = np.abs(interest)
    meaningful = magnitudes[magnitudes > 1e-12 * max(span, 1e-300)]
    featureScale = min(span, np.min(meaningful)) if len(meaningful) else span
    featureScale = max(featureScale, np.finfo(float).tiny)
    maxStepDist = 0.01 * featureScale
    maxDepth = 48
    maxPoints = 50000

    dc, nc = D.coef, N.coef
    shared = min(len(dc), len(nc))
    mask = (dc[:shared] != 0) & (nc[:shared] != 0)
    with np.errstate(over='ignore', under='ignore', divide='ignore'): scales = np.abs(dc[:shared][mask] / nc[:shared][mask])
    scales = scales[np.isfinite(scales) & (scales > 0)]
    gainScale = np.min(scales) if len(scales) else 1.0
    logHigh = np.log(kMax)
    logLow = max(np.log(np.finfo(float).tiny), min(logHigh, np.log(gainScale) - 12 * np.log(10)))
    if kMin > 0: logLow = max(logLow, np.log(kMin))
    count = max(30, int(np.ceil(8 * (logHigh - logLow) / np.log(10))) + 1)
    initialKs = [float(kMin), float(kMax)]
    initialKs.extend(float(k) for k in np.exp(np.linspace(logLow, logHigh, count))[:-1] if kMin < k < kMax)
    initialKs.extend(k for k in kBreaks if kMin <= k <= kMax)

    if k0 is not None and kMin <= k0 <= kMax:
        find_roots(D, N, [k0], rootCache)
        finiteRoots = rootCache[k0]
        initialKs.append(k0)

        # Approach each side until the diverging roots lie outside the view
        for direction in (-1, 1):
            if direction < 0 and k0 <= kMin: continue
            if direction > 0 and k0 >= kMax: continue
            gap = min(k0 * 0.25, (k0 - kMin) if direction < 0 else (kMax - k0))
            for _ in range(maxIterations):
                k = k0 + direction * gap
                if k == k0: raise RuntimeError("Cannot resolve the gain interval near the degree change")
                find_roots(D, N, [k], rootCache)
                fullRoots = rootCache[k]
                if len(fullRoots) != degD: raise RuntimeError("Cannot resolve the full root set near the degree change")

                diverging = np.ones(len(fullRoots), dtype=bool)
                finiteClose = True
                if len(finiteRoots):
                    costMat = np.abs(fullRoots[:, np.newaxis] - finiteRoots)
                    rows, cols = opt.linear_sum_assignment(costMat)
                    diverging[rows] = False
                    finiteClose = np.max(costMat[rows, cols]) <= maxStepDist
                if finiteClose and np.all(np.abs(fullRoots[diverging] - center) > viewRadius):
                    initialKs.append(k)
                    break
                gap *= 0.5
            else: raise RuntimeError("Cannot isolate the degree-changing gain")

    initialKs = sorted(set(initialKs))
    find_roots(D, N, initialKs, rootCache)

    def project_to_view(points):
        offsets = points - center
        factors = np.maximum(1.0, np.abs(offsets) / viewRadius)
        return center + offsets / factors

    def refine_interval(k1, k3, r1, r3, depth=0):
        # The lower-degree sample separates the two sides of a singular gain
        if len(r1) != len(r3): return [k1, k3]
        if not len(r1): return [k1, k3]

        k2 = k1 + 0.5 * (k3 - k1)
        if k2 == k1 or k2 == k3:
            # Adjacent floating-point gains cannot be bisected further
            p1 = project_to_view(r1)
            p3 = project_to_view(align(r1, r3))
            scale = featureScale + np.minimum(np.abs(p1), np.abs(p3))
            if np.max(np.abs(p3 - p1) / scale) <= 0.01:
                return [k1, k3]
            raise RuntimeError("Gain resolution is insufficient to follow some roots")

        find_roots(D, N, [k2], rootCache)
        r2 = rootCache[k2]
        if len(r2) != len(r1): return [k1, k2, k3]
        r2Aligned = align(r1, r2)
        r3Aligned = align(r2Aligned, r3)

        # Bound the sampling effort for branches already outside the visible region
        p1 = project_to_view(r1)
        p2 = project_to_view(r2Aligned)
        p3 = project_to_view(r3Aligned)
        localScale = featureScale + np.minimum.reduce([np.abs(p1), np.abs(p2), np.abs(p3)])
        step = np.maximum(np.abs(p2 - p1), np.abs(p3 - p2)) / localScale
        bend = np.abs(p2 - 0.5 * (p1 + p3)) / localScale

        if np.max(step) > 0.01 or np.max(bend) > 0.001:
            if depth >= maxDepth or len(rootCache) >= maxPoints:
                raise RuntimeError(
                    f"Sampling unresolved for K in [{k1:.12g}, {k3:.12g}]. "
                    "Increase the sampling limits or inspect polynomial conditioning."
                )
            leftKs = refine_interval(k1, k2, r1, r2Aligned, depth + 1)
            rightKs = refine_interval(k2, k3, r2Aligned, r3Aligned, depth + 1)
            return leftKs[:-1] + rightKs
        # Preserve the midpoint used to verify this interval
        else: return [k1, k2, k3]

    kList = [initialKs[0]]
    for i in range(len(initialKs) - 1):
        kStart = initialKs[i]
        kEnd = initialKs[i + 1]
        rStart = rootCache[kStart]
        rEnd = rootCache[kEnd]
        segmentKs = refine_interval(kStart, kEnd, rStart, rEnd)
        kList.extend(segmentKs[1:])
    # =======================================================================================================================================
    return kList, rootCache

def main():
    inD = input("Characteristic equation: D(s) + K * N(s) = 0\nD(s) coefficients (descending order):")
    D = list()
    while not parse(inD, D):
        print("First coefficient must be a nonzero number, all coeficient must be real finite numbers")
        inD = input("D(s) coefficients (descending order):")
    N = list()
    inN = input("N(s) coefficients (descending order):")
    while not parse(inN, N):
        print("First coefficient must be a nonzero number, all coeficient must be real finite numbers")
        inN = input("N(s) coefficients (descending order):")

    D = Polynomial(np.flip(np.array(D)))
    N = Polynomial(np.flip(np.array(N)))

    try:
        result = calculate_locus(D, N)
        openPoles = result['poles']
        openZeros = result['zeros']
        breakpoints = result['breakpoints']
        lociMat = result['loci']

        fig, ax = plt.subplots(figsize=(10, 7))
        ax.set_aspect('equal', adjustable='box')
        # Preserve finite branches at degree-changing gains
        for branchIdx in range(lociMat.shape[1]):
            branch = lociMat[:, branchIdx]
            ax.plot(branch.real, branch.imag, color="#0d9488", zorder=1)
        ax.scatter(openPoles.real, openPoles.imag, marker='x', color="#e11d48", s=30, zorder=2)
        ax.scatter(openZeros.real, openZeros.imag, marker='o', facecolors="none", edgecolors="#1d4ed8", s=30, zorder=2)

        center, span = get_plot_scale(openPoles, openZeros, breakpoints)
        viewRadius = 4 * span
        ax.set_xlim(center.real - viewRadius, center.real + viewRadius)
        ax.set_ylim(center.imag - viewRadius, center.imag + viewRadius)
        ax.axhline(0, color="#94a3b8", linewidth=0.6, zorder=0)
        ax.axvline(0, color="#94a3b8", linewidth=0.6, zorder=0)
        ax.set_xlabel(r"Real axis ($\sigma$)")
        ax.set_ylabel(r"Imaginary axis ($j\omega$)")
        ax.set_title("Root-locus diagram")
        ax.ticklabel_format(axis='both', style='plain', useOffset=False)
        ax.grid(True, linestyle="--", alpha=0.3)
        plt.show()
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
        print(error)
        return

if __name__ == "__main__":
    main()