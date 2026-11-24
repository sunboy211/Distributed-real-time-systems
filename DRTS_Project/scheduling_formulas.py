# scheduling_formulas.py
import math
import bisect

def compute_busy_window(task_params, max_iter=1000):
    prev = 0
    for _ in range(max_iter):
        nxt = sum(math.ceil(prev / T) * C for C, T in task_params if T > 0)
        if nxt == prev:
            return nxt
        prev = nxt
    raise ValueError('Busy-window did not converge')


def demand_bound_edf(task_params, t):
    return sum(math.floor(t / T) * C for C, T in task_params if T > 0)


def supply_bound_bdr(alpha, latency, t):
    return alpha * (t - latency) if t >= latency else 0.0


def lcm(a, b):
    if a == 0 or b == 0:
        return 0
    val_a = int(a)
    val_b = int(b)
    return abs(val_a * val_b) // math.gcd(val_a, val_b)


def rta_rm_bdr(tasks, alpha, latency, eps=1e-9):
    # sort by Rate Monotonic priority (shorter T → higher prio)
    sorted_tasks = sorted(tasks, key=lambda x: x['prio'])
    Ts = [tk['T'] for tk in sorted_tasks]
    Cs = [tk['C'] for tk in sorted_tasks]
    prefix = [0]
    for c in Cs:
        prefix.append(prefix[-1] + c)
    results = {}
    for i, tk in enumerate(sorted_tasks):
        C, T = tk['C'], tk['T']
        R = C / alpha + latency
        while True:
            p = bisect.bisect_right(Ts, R, 0, i)
            I1 = sum(math.ceil(R / Ts[j]) * Cs[j] for j in range(p))
            I2 = prefix[i] - prefix[p]
            I = I1 + I2
            Rn = (C + I) / alpha + latency
            if abs(Rn - R) < eps or Rn > T:
                R = Rn
                break
            R = Rn
        results[tk['name']] = (R <= T, R)
    return results


def half_half_interface(alpha, latency):
    if alpha >= 1.0:
        raise ValueError('alpha must <1')
    C_sup = alpha * latency / (2 * (1 - alpha))
    T_sup = latency / (2 * (1 - alpha))
    return C_sup, T_sup
