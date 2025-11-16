import math

def fak_1(t, params): return max(0.1, min(0.99, params[0] * t + params[1]))
def fak_2(t, params): return max(0.1, min(0.99, params[1] * params[0] * t**2 + params[1] * t + params[2]))
def fak_3(t, params): return max(0.1, min(0.99, params[2]))
def fak_4(t, params): return max(0.1, min(0.99, -params[0] * t + params[1]))
def fak_6(t, params): return max(0.1, min(0.99, params[0] * t + params[1]))
def fak_7(t, params): return max(0.1, min(0.99, params[0] * t + params[2]))

def fx(x, params, initial_value=None):
    if initial_value is None:
        initial_value = params[3]
    result = (params[0] * x)**3 + (params[1] * x)**2 + (params[2] * x) + params[3]
    if result >= 0.999:
        result = initial_value + (result - 1) * 0.5
    elif result <= 0.001:
        result = initial_value + (0 - result) * 0.5
    return max(0.0, min(1.0, result))

def normalize_line(values, threshold_min=0.05, threshold_max=0.95):
    if len(values) == 0:
        return values
    
    min_val = np.min(values)
    max_val = np.max(values)
    if min_val < threshold_min or max_val > threshold_max:
        if max_val - min_val > 0:
            normalized = 0.1 + 0.8 * (values - min_val) / (max_val - min_val)
        else:
            normalized = np.full_like(values, 0.5)
        return normalized
    else:
        return values

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
    
    dudt = []
    for i in range(8):
        chaos_factor = 0.4 + 0.3 * (i / 7)
        oscillation = math.sin(2.5 * t + i * 0.9) * 0.3
        noise = math.cos(2.0 * t + i * 1.3) * 0.2
        
        d = chaos_factor * (
            fxu((i+1)%len(u)) * (fak_1(t,faks[0]) + 0.6*fak_2(t,faks[1])) -
            fxu((i+2)%len(u)) * fak_3(t,faks[2]) +
            oscillation +
            noise
        )
        
        dudt.append(max(-1.0, min(1.0, d)))
    for i in range(8, 23):
        left = u[i-1]
        right = u[i+1] if i+1 < len(u) else u[i]
        neighbor_avg = (left + right) / 2
        
        d = 0.25 * (
            1.0 * (neighbor_avg - u[i]) +
            0.5 * math.sin(1.0*t + i) +
            0.3 * math.cos(0.6*t + i/2) +
            0.2 * (fak_1(t,faks[0]) - fak_3(t,faks[2]))
        )
            
        dudt.append(max(-0.5, min(0.5, d)))
    
    for i in range(len(u)):
        if u[i] <= 0.001 or u[i] >= 0.999:
            dudt[i] = 0.0
    
    return dudt