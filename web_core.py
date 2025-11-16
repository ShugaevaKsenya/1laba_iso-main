import os
import base64
import io
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
try:
    from labellines import labelLines
except ImportError:
    def labelLines(*args, **kwargs):
        return None
from scipy.integrate import odeint
from scipy import interpolate
from functions import pend, fak_1, fak_2, fak_3, fak_4, fak_6, fak_7
from radar_diagram import RadarDiagram

U_LABELS = [
    "X₁ : качество",
    "X₂ : доступность", 
    "X₃ : завершенность",
    "X₄ : устойчивость к ошибкам",
    "X₅ : восстановляемость",
    "X₆ : сбои и отказы при работы системы",
    "X₇ : ошибки завершенности",
    "X₈ : отказы при работе программного обеспечения",
    "X₉ : сбои при работе программного обеспечения",
    "X₁₀ : отсутствие требований по восстановлению данных при отказах операционной системы и аппаратного обеспечения",
    "X₁₁ : потери данных при отказах операционной системы и аппаратного обеспечения",
    "X₁₂ : ошибка восстановления предшествующего состояния системы после повторного запуска программного обеспечения",
    "X₁₃ : отсутствие требований по восстановлению вычислительного процесса в случае сбоя операционной системы и аппаратного обеспечения",
    "X₁₄ : ошибка восстановления процесса в случае сбоев оборудования",
    "X₁₅ : ошибка восстановления данных в случае их искажений или разрушения",
    "X₁₆ : несоответствие требованиям стандартов, соглашений, законов или других предписаний, связанных с качеством",
    "X₁₇ : неполнота обработки ошибочных ситуаций",
    "X₁₈ : неполнота контроля корректности, полноты и непротиворечивости входных, выходных данных и баз данных",
    "X₁₉ : отсутствие возможности функционирования в сокращенном объеме в случае ошибок или помех",
    "X₂₀ : недостатки средств контроля работоспособности и диагностирования аппаратных и программных средств",
    "X₂₁ : отсутствие диагностического сообщения в случае сбоя или отказа",
    "X₂₂ : неполнота контроля непротиворечивости входных и баз данных",
    "X₂₃ : надежность системы"
]

def _fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('ascii')

def smooth_data(values, window_size=5):
    if len(values) < window_size:
        return values
    
    window = np.ones(window_size) / window_size
    smoothed = np.convolve(values, window, mode='same')
    
    half_window = window_size // 2
    for i in range(half_window):
        smoothed[i] = np.mean(values[:i*2+1])
        smoothed[-(i+1)] = np.mean(values[-(i*2+1):])
    
    return smoothed

def normalize_line_for_plot(values, threshold_min=0.05, threshold_max=0.95):
    if len(values) == 0:
        return values
    
    smoothed_values = smooth_data(values)
    
    min_val = np.min(smoothed_values)
    max_val = np.max(smoothed_values)
    
    if min_val < threshold_min or max_val > threshold_max:
        if max_val - min_val > 0:
            normalized = 0.1 + 0.8 * (smoothed_values - min_val) / (max_val - min_val)
        else:
            normalized = np.full_like(smoothed_values, 0.5)
        return normalized
    else:
        return smoothed_values

def normalize_for_radar(values, restrictions=None):
    if len(values) == 0:
        return values
    
    smoothed = smooth_data(values, window_size=3)
    
    if restrictions is not None:
        capped_values = np.minimum(smoothed, restrictions)
        
        min_val = np.min(capped_values)
        max_val = np.max(capped_values)
        
        if max_val - min_val > 0:
            normalized = 0.05 + 0.9 * (capped_values - min_val) / (max_val - min_val)
        else:
            normalized = capped_values
        
        return np.clip(normalized, 0.05, 0.95)
    else:
        min_val = np.min(smoothed)
        max_val = np.max(smoothed)
        
        if max_val > 0.95 or min_val < 0.05:
            if max_val - min_val > 0:
                normalized = 0.05 + 0.9 * (smoothed - min_val) / (max_val - min_val)
            else:
                normalized = smoothed
            return np.clip(normalized, 0.05, 0.95)
        else:
            return smoothed

def create_smooth_line(t_original, values, num_points=1000):
    if len(t_original) < 4:
        return t_original, values
    
    try:
        interp_func = interpolate.interp1d(t_original, values, kind='cubic', 
                                         fill_value="extrapolate")
        
        t_smooth = np.linspace(t_original[0], t_original[-1], num_points)
        values_smooth = interp_func(t_smooth)
        
        return t_smooth, values_smooth
    except:
        return t_original, values

def draw_faks(t, faks):
    fig, ax = plt.subplots(figsize=(10, 5))
    fak_functions = [fak_1, fak_2, fak_3, fak_4, fak_6, fak_7]
    sub = {1: '₁', 2: '₂', 3: '₃', 4: '₄', 6: '₆', 7: '₇'}
    labels = [f"Fak{sub[i]}" for i in [1, 2, 3, 4, 6, 7]]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    for i, (fak_func, color, label) in enumerate(zip(fak_functions, colors, labels)):
        y_values = []
        for v in t:
            raw_val = fak_func(v, faks[i])
            y_values.append(raw_val)
        
        y_values = np.array(y_values)
        y_normalized = normalize_line_for_plot(y_values)
        
        t_smooth, y_smooth = create_smooth_line(t, y_normalized)
        
        line, = ax.plot(t_smooth, y_smooth, color=color, label=label, 
                       linewidth=2.5, antialiased=True, alpha=0.8)
        
        if len(t_smooth) >= 10:
            x_pos = t_smooth[-10]
            y_pos = y_smooth[-10]
        else:
            x_pos = t_smooth[-1]
            y_pos = y_smooth[-1]
        ax.text(
            x_pos, y_pos, label,
            color=color,
            fontsize=8,
            verticalalignment='center',
            horizontalalignment='left',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.6, pad=0.3)
        )
    
    ax.set_xlabel("Время (t)")
    ax.set_ylabel("Значения")
    ax.set_title("Графики возмущений")
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.0)
    plt.tight_layout()
    return fig

def create_split_graphics(t, data, faks):
    figs_b64 = []
    sub = {
        1: '₁', 2: '₂', 3: '₃', 4: '₄', 5: '₅', 6: '₆', 7: '₇', 8: '₈',
        9: '₉', 10: '₁₀', 11: '₁₁', 12: '₁₂', 13: '₁₃', 14: '₁₄', 15: '₁₅',
        16: '₁₆', 17: '₁₇', 18: '₁₈', 19: '₁₉', 20: '₂₀', 21: '₂₁', 22: '₂₂',
        23: '₂₃'
    }

    normalized_data = np.zeros_like(data)
    for i in range(23):
        y_data = np.maximum(0, data[:, i])
        normalized_data[:, i] = normalize_line_for_plot(y_data)

    fig1, ax1 = plt.subplots(figsize=(10, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, 8))
    
    for i in range(8):
        y_data = normalized_data[:, i]
        
        if len(t) > 1:
            t_simple = np.array([t[0], t[-1]])
            y_simple = np.array([y_data[0], y_data[-1]])
            
            t_plot = np.linspace(t[0], t[-1], 50)
            y_plot = np.interp(t_plot, t_simple, y_simple)
        else:
            t_plot = t
            y_plot = y_data
        
        label = f"X{sub[i+1]}"
        line, = ax1.plot(t_plot, y_plot, color=colors[i], label=label, 
                        linewidth=2.5, antialiased=True, alpha=0.8)
        
        if len(t_plot) > 0:
            x_pos = t_plot[0]
            y_pos = y_plot[0]
            ax1.text(x_pos, y_pos, label, color=colors[i], fontsize=9,
                     verticalalignment='center', horizontalalignment='left',
                     bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=0.5))
    
    ax1.set_xlabel("Время (t)")
    ax1.set_ylabel("Значения характеристик")
    ax1.set_title("Характеристики X₁–X₈")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.0)
    figs_b64.append(_fig_to_base64(fig1))
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, 8))
    for i in range(8, 16):
        y_data = normalized_data[:, i]
        
        if len(t) > 1:
            t_simple = np.array([t[0], t[-1]])
            y_simple = np.array([y_data[0], y_data[-1]])
            t_plot = np.linspace(t[0], t[-1], 50)
            y_plot = np.interp(t_plot, t_simple, y_simple)
        else:
            t_plot = t
            y_plot = y_data
        
        label = f"X{sub[i+1]}"
        line, = ax2.plot(t_plot, y_plot, color=colors[i-8], label=label, 
                        linewidth=2.5, antialiased=True, alpha=0.8)
        
        if len(t_plot) > 0:
            x_pos = t_plot[0]
            y_pos = y_plot[0]
            ax2.text(x_pos, y_pos, label, color=colors[i-8], fontsize=9,
                     verticalalignment='center', horizontalalignment='left',
                     bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=0.5))
    
    ax2.set_xlabel("Время (t)")
    ax2.set_ylabel("Значения характеристик")
    ax2.set_title("Характеристики X₉–X₁₆")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.0)
    figs_b64.append(_fig_to_base64(fig2))
    plt.close(fig2)

    fig3, ax3 = plt.subplots(figsize=(10, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, 7))
    for i in range(16, 23):
        y_data = normalized_data[:, i]
        
        if len(t) > 1:
            t_simple = np.array([t[0], t[-1]])
            y_simple = np.array([y_data[0], y_data[-1]])
            t_plot = np.linspace(t[0], t[-1], 50)
            y_plot = np.interp(t_plot, t_simple, y_simple)
        else:
            t_plot = t
            y_plot = y_data
        
        label = f"X{sub[i+1]}"
        line, = ax3.plot(t_plot, y_plot, color=colors[i-16], label=label, 
                        linewidth=2.5, antialiased=True, alpha=0.8)
        
        if len(t_plot) > 0:
            x_pos = t_plot[0]
            y_pos = y_plot[0]
            ax3.text(x_pos, y_pos, label, color=colors[i-16], fontsize=9,
                     verticalalignment='center', horizontalalignment='left',
                     bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=0.5))
    
    ax3.set_xlabel("Время (t)")
    ax3.set_ylabel("Значения характеристик")
    ax3.set_title("Характеристики X₁₇–X₂₃")
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1.0)
    figs_b64.append(_fig_to_base64(fig3))
    plt.close(fig3)

    fig4 = draw_faks(t, faks)
    figs_b64.append(_fig_to_base64(fig4))
    plt.close(fig4)
    
    return figs_b64

def draw_radar_series(data, initial_equations, restrictions):
    radar = RadarDiagram()
    imgs = []
    
    normalized_initial = normalize_for_radar(np.array(initial_equations), restrictions)
    
    time_points = [
        int(len(data) / 4),
        int(len(data) / 2), 
        int(len(data) / 4 * 3),
        -1
    ]
    
    normalized_points = []
    for point_idx in time_points:
        point_data = data[point_idx, :]
        normalized_point = normalize_for_radar(point_data, restrictions)
        normalized_points.append(normalized_point)
    
    imgs.append(base64.b64encode(radar.draw_bytes(
        normalized_initial, 
        [f"X$_{i+1}$" for i in range(23)], 
        "Характеристики системы в начальный момент времени", 
        restrictions,
        normalized_initial
    )).decode('ascii'))
    imgs.append(base64.b64encode(radar.draw_bytes(
        normalized_points[0], 
        [f"X$_{i+1}$" for i in range(23)], 
        "Характеристики системы в 1 четверти", 
        restrictions,
        normalized_initial
    )).decode('ascii'))
    imgs.append(base64.b64encode(radar.draw_bytes(
        normalized_points[1], 
        [f"X$_{i+1}$" for i in range(23)], 
        "Характеристики системы во 2 четверти", 
        restrictions,
        normalized_initial
    )).decode('ascii'))
    imgs.append(base64.b64encode(radar.draw_bytes(
        normalized_points[2], 
        [f"X$_{i+1}$" for i in range(23)], 
        "Характеристики системы в 3 четверти", 
        restrictions,
        normalized_initial
    )).decode('ascii'))
    imgs.append(base64.b64encode(radar.draw_bytes(
        normalized_points[3], 
        [f"X$_{i+1}$" for i in range(23)], 
        "Характеристики системы в последний момент времени", 
        restrictions,
        normalized_initial
    )).decode('ascii'))
    return imgs

def run_simulation(initial_equations, faks, equations, restrictions):
    init_eq = np.array(initial_equations, dtype=float)

    for i in range(8):
        direction = np.random.choice([-1, 1])
        magnitude = np.random.uniform(0.2, 0.6)
        
        if direction == 1:
            init_eq[i] = min(init_eq[i] + magnitude * np.random.random(), 0.95)
        else:
            init_eq[i] = max(init_eq[i] - magnitude * np.random.random(), 0.05)
    
    noise = np.random.uniform(-0.15, 0.15, size=8)
    init_eq[:8] = np.clip(init_eq[:8] + noise, 0.05, 0.95)
    
    variation_later = np.random.uniform(-0.1, 0.1, size=15)
    init_eq[8:] = np.clip(init_eq[8:] + variation_later, 0.05, 0.95)
    
    t = np.linspace(0, 1, 50)
    
    data_sol = odeint(pend, init_eq, t, args=(faks, equations))
    data_sol = np.maximum(data_sol, 0)
    
    figure_b64 = create_split_graphics(t, data_sol, faks)
    radar_imgs = draw_radar_series(data_sol, initial_equations, restrictions)
    return {
        'images_b64': {
            'figure1': figure_b64[0],
            'figure2': figure_b64[1], 
            'figure3': figure_b64[2],
            'figure4': figure_b64[3],
            'diagram': radar_imgs[0],
            'diagram2': radar_imgs[1],
            'diagram3': radar_imgs[2],
            'diagram4': radar_imgs[3],
            'diagram5': radar_imgs[4],
        },
        'success': 'Расчет выполнен успешно!'
    }

def build_default_inputs():
    rng = np.random.default_rng()
    u_values = []
    
    for i in range(23):
        if i < 8:
            if i == 0:
                val = rng.uniform(0.05, 0.15)
            elif i == 1:
                val = rng.uniform(0.85, 0.95)
            elif i == 2:
                val = rng.uniform(0.2, 0.35)
            elif i == 3:
                val = rng.uniform(0.7, 0.8)
            elif i == 4:
                val = rng.uniform(0.4, 0.6)
            elif i == 5:
                val = rng.uniform(0.9, 0.98)
            elif i == 6:
                val = rng.uniform(0.02, 0.1)
            else:
                val = rng.uniform(0.6, 0.75)
        else:
            if i % 3 == 0:
                val = rng.uniform(0.3, 0.5)
            elif i % 3 == 1:
                val = rng.uniform(0.5, 0.7)
            else:
                val = rng.uniform(0.7, 0.9)
        
        u_values.append(round(float(val), 2))
    
    defaults = {
        'u': u_values,
        'u_restrictions': [round(float(rng.random() * 0.3 + 0.7), 2) for _ in range(23)],
        'faks': [],
        'equations': []
    }
    
    for _ in [1, 2, 3, 4, 6, 7]:
        defaults['faks'].append([
            round(float(rng.random() * 0.3 + 0.1), 2),
            round(float(rng.random() * 0.4 + 0.3), 2),
            round(float(rng.random() * 0.3 + 0.4), 2),
            round(float(rng.random() * 0.2 + 0.1), 2),
        ])
    
    for _ in range(316):
        defaults['equations'].append([
            round(float(rng.uniform(-0.6, 0.6)), 2),
            round(float(rng.uniform(-0.6, 0.6)), 2),
            round(float(rng.uniform(-0.5, 0.5)), 2),
            round(float(rng.uniform(-0.4, 0.4)), 2),
        ])
    
    return defaults

def get_u_variable_for_equation(equation_number):
    u_mapping = {
        1: 12, 2: 13, 3: 14, 4: 57, 5: 59, 6: 58, 7: 61, 8: 60, 9: 71, 10: 70,
        11: 72, 12: 74, 13: 63, 14: 67, 15: 62, 16: 12, 17: 13, 18: 57, 19: 59,
        20: 58, 21: 61, 22: 60, 23: 68,
        24: 65, 25: 2, 26: 15, 27: 13, 28: 14, 29: 57, 30: 59, 31: 58, 32: 61,
        33: 60, 34: 71, 35: 74, 36: 67, 37: 2, 38: 12, 39: 14, 40: 57, 41: 59,
        42: 58, 43: 61, 44: 60, 45: 71, 46: 70, 47: 72, 48: 63, 49: 65, 50: 66,
        51: 14, 52: 57, 55: 59, 56: 58, 57: 71, 58: 73, 59: 68, 60: 70, 61: 72,
        62: 57, 63: 58, 64: 61, 65: 60, 66: 69, 67: 71, 68: 73, 69: 68, 70: 70,
        71: 72, 72: 74, 73: 63, 74: 65, 75: 67, 76: 62, 77: 64, 78: 66, 79: 2,
        80: 15, 81: 12, 82: 13, 83: 14, 84: 57, 85: 59, 86: 61, 87: 60, 88: 71,
        89: 73, 90: 70, 91: 72, 92: 67, 93: 64, 94: 2, 95: 15, 96: 12, 97: 13,
        98: 57, 99: 59, 100: 61,
        101: 60, 102: 71, 103: 73, 104: 70, 105: 72, 106: 66, 107: 2, 108: 15,
        109: 12, 110: 13, 111: 14, 112: 57, 113: 59, 114: 58, 115: 61, 116: 71,
        117: 73, 118: 70, 119: 72, 120: 66, 121: 2, 122: 15, 123: 12, 124: 13,
        125: 14, 126: 57, 127: 59, 128: 58, 129: 71, 130: 73, 131: 70, 132: 72,
        133: 66, 134: 2, 135: 15, 136: 12, 137: 13, 138: 14, 139: 57, 140: 69,
        141: 68, 142: 74, 143: 63, 144: 62, 145: 64, 146: 66, 147: 12, 148: 14,
        149: 57, 150: 59,
        151: 58, 152: 61, 153: 60, 154: 69, 155: 71, 156: 68, 157: 70, 158: 72,
        159: 74, 160: 63, 161: 65, 162: 67, 163: 66, 164: 2, 165: 15, 166: 12,
        167: 13, 168: 14, 169: 57, 170: 59, 171: 58, 172: 61, 173: 71, 174: 68,
        175: 70, 176: 66, 177: 2, 178: 15, 179: 12, 180: 13, 181: 57, 182: 59,
        183: 58, 184: 61, 185: 60, 186: 71, 187: 73, 188: 70, 189: 72, 190: 63,
        191: 65, 192: 67, 193: 62, 194: 64, 195: 66, 196: 12, 197: 57, 198: 59,
        199: 58, 200: 61,
        201: 60, 202: 69, 203: 73, 204: 70, 205: 63, 206: 65, 207: 66, 208: 2,
        209: 15, 210: 12, 211: 13, 212: 14, 213: 57, 214: 59, 215: 58, 216: 61,
        217: 60, 218: 69, 219: 73, 220: 68, 221: 70, 222: 63, 223: 65, 224: 66,
        225: 2, 226: 15, 227: 12, 228: 13, 229: 14, 230: 57, 231: 61, 232: 69,
        233: 73, 234: 67, 235: 62, 236: 64, 237: 2, 238: 15, 239: 12, 240: 13,
        241: 14, 242: 57, 243: 59, 244: 58, 245: 61, 246: 60, 247: 71, 248: 73,
        249: 70, 250: 72,
        251: 2, 252: 15, 253: 12, 254: 13, 255: 14, 256: 57, 257: 59, 258: 58,
        259: 61, 260: 60, 261: 69, 262: 71, 263: 73, 264: 70, 265: 72, 266: 63,
        267: 2, 268: 15, 269: 12, 270: 13, 271: 14, 272: 57, 273: 59, 274: 58,
        275: 61, 276: 60, 277: 69, 278: 71, 279: 73, 280: 70, 281: 72, 282: 63,
        283: 2, 284: 15, 285: 12, 286: 13, 287: 14, 288: 57, 289: 58, 290: 61,
        291: 71, 292: 62, 293: 67, 294: 12, 295: 13, 296: 59, 297: 58, 298: 61,
        299: 60, 300: 71, 301: 70, 302: 72, 303: 64, 304: 66, 305: 58, 306: 61,
        307: 71, 308: 68, 309: 74, 310: 63, 311: 62, 312: 2, 313: 15, 314: 12,
        315: 13, 316: 14
    }
    return u_mapping.get(equation_number, "?")

def parse_form(form):
    u = []
    u_restrictions = []
    
    subscript_numbers = {
        '₁': '1', '₂': '2', '₃': '3', '₄': '4', '₅': '5', '₆': '6', '₇': '7', '₈': '8', 
        '₉': '9', '₁₀': '10', '₁₁': '11', '₁₂': '12', '₁₃': '13', '₁₄': '14', '₁₅': '15',
        '₁₆': '16', '₁₇': '17', '₁₈': '18', '₁₉': '19', '₂₀': '20', '₂₁': '21', '₂₂': '22', 
        '₂₃': '23'
    }
    reverse_subscript = {v: k for k, v in subscript_numbers.items()}
    
    for i in range(1, 24):
        field_name_with_sub = f'u{reverse_subscript[str(i)]}'
        field_name_normal = f'u{i}'
        value = form.get(field_name_with_sub) or form.get(field_name_normal, '0.1')
        u.append(float(value or 0.1))
        
        restriction_field_name_with_sub = f'u_restrictions{reverse_subscript[str(i)]}'
        restriction_field_name_normal = f'u_restrictions{i}'
        restriction_value = form.get(restriction_field_name_with_sub) or form.get(restriction_field_name_normal, '1.0')
        u_restrictions.append(float(restriction_value or 1.0))
    
    fak_ids = [1, 2, 3, 4, 6, 7]
    faks = []
    
    for fid in fak_ids:
        fak_values = []
        
        for param_num in [1, 2, 3, 4]:
            field_name_normal = f'fak{fid}_{param_num}'
            value_normal = form.get(field_name_normal)
            
            subscript_fid = reverse_subscript[str(fid)]
            field_name_with_sub = f'fak{subscript_fid}_{param_num}'
            value_with_sub = form.get(field_name_with_sub)
            
            final_value = value_with_sub or value_normal or '0'
            fak_values.append(float(final_value))
        
        faks.append(fak_values)

    equations = []
    for i in range(1, 317):
        eq_values = []
        for param_num in [1, 2, 3, 4]:
            field_name = f'f{i}_{param_num}'
            value = form.get(field_name, '0')
            eq_values.append(float(value))
        equations.append(eq_values)
    
    return u, faks, equations, u_restrictions