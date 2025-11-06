
import math

def fak_1(t, params): return max(0.0, min(0.99, params[0] * t + params[1]))
def fak_2(t, params): return max(0.0, min(0.99, params[1] * params[0] * t**2 + params[1] * t + params[2]))
def fak_3(t, params): return max(0.0, min(0.99, params[2]))
def fak_4(t, params): return max(0.0, min(0.99, -params[0] * t + params[1]))
def fak_6(t, params): return max(0.0, min(0.99, params[0] * t + params[1]))
def fak_7(t, params): return max(0.0, min(0.99, params[0] * t + params[2]))

def fx(x, params, initial_value=None):
    if initial_value is None:
        initial_value = params[3]
    result = (params[0] * x)**3 + (params[1] * x)**2 + (params[2] * x) + params[3]
    if result >= 0.999:
        result = initial_value + (result - 1) * 0.5
    elif result <= 0.001:
        result = initial_value + (0 - result) * 0.5
    return max(0.0, min(1.0, result))

initial_values = None

def pend(u, t, faks, f):
    global initial_values
    if initial_values is None:
        initial_values = u.copy()
    for i in range(len(u)):
        if u[i] <= 0.001 or u[i] >= 0.999:
            u[i] = initial_values[i]
    seq = list(range(317))
    fxu = lambda x: fx(u[x], f[seq.pop(0)], initial_values[x])
    alpha = 0.25
    dudt = []
    for i in range(8):
        dudt.append(alpha * (
            fxu((i+1)%len(u)) * (fak_1(t,faks[0]) + 0.5*fak_2(t,faks[1])) -
            fxu((i+2)%len(u)) * fak_3(t,faks[2])
        ))
    for i in range(8, 23):
        left = u[i-1]
        right = u[i+1] if i+1 < len(u) else u[i]
        neighbor_avg = (left + right) / 2
        d = alpha * (
            1.2 * (neighbor_avg - u[i]) +
            0.8 * math.sin(1.5*t + i) +
            0.6 * math.cos(0.7*t + i/2) +
            0.5 * (fak_1(t,faks[0]) - fak_3(t,faks[2]))
        )
        dudt.append(max(-1.0, min(1.0, d)))
    for i in range(len(u)):
        if u[i] <= 0.001 or u[i] >= 0.999:
            dudt[i] = 0.0
    return dudt
