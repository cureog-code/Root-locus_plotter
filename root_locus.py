import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
from numpy.polynomial import Polynomial
from numpy.polynomial.polynomial import polyadd

# https://eng.libretexts.org/Courses/California_State_Polytechnic_University_Humboldt/Measurements_Instrumentation_and_Controls/Chapter_12%3A_Root_Locus
# Characteristic equation D(s) + k*N(s) = 0
# k: gain

def parse(userInput, coefList):
    try:
        temp = list(map(float, userInput.split(',')))
        if (temp[0] != 0 and np.isfinite(temp).all()):
            coefList[:] = temp
            return True
        return False
    except ValueError: return False

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
            result[k] = cEq.roots()
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

def align(ref, target, rootCount=None):
    if rootCount is None:
        refLen = len(ref)
        if len(target) != refLen: raise ValueError("Root arrays must have the same length")
        elif refLen == 0: raise ValueError("Cannot align null arrays")
        costMat = np.abs(target[:, np.newaxis] - ref)
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
            costMat = np.abs(target[:, np.newaxis] - ref[validIdx])
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

    # At k --> inf, all finite roots approach the open-loop zeros
    if len(openZeros): alignedLoci.append(align(prevRoots, openZeros, rootCount))
    return np.array(alignedLoci, dtype=complex)

def get_plot_scale(openPoles, openZeros, breakpoints):
    sBreaks = [sB for sB, kB in breakpoints]
    allPts = np.concatenate([openPoles, openZeros, np.array(sBreaks, dtype=complex)])
    center = np.mean(allPts)
    span = np.max(np.abs(allPts - center))
    span = max(span, 0.1 * np.max(np.abs(allPts)))
    if span == 0: span = 1.0
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

        drawRange = 4 * max(abs(pointsOfInterest - centroid))
        if drawRange == 0: drawRange = 1
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
    maxStepDist = 0.01 * span
    bendTol = 0.001 * span
    maxDepth = 40
    maxPoints = 50000
    samplingLimited = False

    initialKs = np.linspace(kMin, kMax, 30).tolist()
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
        nonlocal samplingLimited
        # The lower-degree sample separates the two sides of a singular gain
        if len(r1) != len(r3): return [k1, k3]
        if not len(r1): return [k1, k3]

        k2 = k1 + 0.5 * (k3 - k1)
        if depth >= maxDepth or k2 == k1 or k2 == k3 or len(rootCache) >= maxPoints:
            samplingLimited = True
            return [k1, k3]

        find_roots(D, N, [k2], rootCache)
        r2 = rootCache[k2]
        if len(r2) != len(r1): return [k1, k2, k3]
        r2Aligned = align(r1, r2)
        r3Aligned = align(r2Aligned, r3)

        # Bound the sampling effort for branches already outside the visible region
        p1 = project_to_view(r1)
        p2 = project_to_view(r2Aligned)
        p3 = project_to_view(r3Aligned)
        maxDist = max(np.max(np.abs(p2 - p1)), np.max(np.abs(p3 - p2)))
        maxBend = np.max(np.abs(p2 - 0.5 * (p1 + p3)))

        if maxDist > maxStepDist or maxBend > bendTol:
            leftKs = refine_interval(k1, k2, r1, r2Aligned, depth + 1)
            rightKs = refine_interval(k2, k3, r2Aligned, r3Aligned, depth + 1)
            return leftKs[:-1] + rightKs
        else: return [k1, k3]

    kList = [initialKs[0]]
    for i in range(len(initialKs) - 1):
        kStart = initialKs[i]
        kEnd = initialKs[i + 1]
        rStart = rootCache[kStart]
        rEnd = rootCache[kEnd]
        segmentKs = refine_interval(kStart, kEnd, rStart, rEnd)
        kList.extend(segmentKs[1:])
    if samplingLimited: print("Sampling limit reached; some intervals may need more points")
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
        openZeros = N.roots()
        openPoles = D.roots()
        breakpoints = find_breakpoints(D, N)
        kList, roots = compute_roots_gains(D, N, openPoles, openZeros, breakpoints)

        fig, ax = plt.subplots(figsize=(10, 7))
        ax.set_aspect('equal', adjustable='box')
        # Preserve finite branches at degree-changing gains & endpoints
        lociMat = build_loci(kList, roots, openPoles, openZeros)
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
        ax.grid(True, linestyle="--", alpha=0.3)
        plt.show()
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
        print(error)
        return

if __name__ == "__main__":
    main()