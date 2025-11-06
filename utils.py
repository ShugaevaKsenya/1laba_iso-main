def get_initial_equations_from_inputs(ui):
    return [float(ui.lineEdits[f"u{i}"].text()) for i in range(1, 24)]


def get_faks_from_inputs(ui):
    result = []
    for i in [1, 2, 3, 4, 6, 7]:
        a = float(ui.lineEdits[f"fak{i}_1"].text())
        b = float(ui.lineEdits[f"fak{i}_2"].text())
        c = float(ui.lineEdits[f"fak{i}_3"].text())
        d = float(ui.lineEdits[f"fak{i}_4"].text())
        result.append([a, b, c, d])
    return result


def get_equations_from_inputs(ui):
    result = []
    for i in range(1, 317):
        a = float(ui.lineEdits[f"f{i}_1"].text())
        b = float(ui.lineEdits[f"f{i}_2"].text())
        c = float(ui.lineEdits[f"f{i}_3"].text())
        d = float(ui.lineEdits[f"f{i}_4"].text())
        result.append([a, b, c, d])
    return result


def get_restrictions(ui):
    return [float(ui.lineEdits[f"u_restrictions{i}"].text()) for i in range(1, 24)]


lines = (
    ('g', '-'),
    ('c', '-'),
    ('r', '-'),
    ('y', '-'),
    ('m', '-'),
    ('b', '-'),
    ('teal', '-'),
    ('gray', '-'),
    ('olive', '-'),
    ('g', '--'),
    ('c', '--'),
    ('r', '--'),
    ('y', '--'),
    ('m', '--'),
    ('b', '--'),
    ('teal', '--'),
    ('gray', '--'),
    ('olive', '--'),
    ('g', '-.'),
    ('c', '-.'),
    ('r', '-.'),
    ('y', '-.'),
    ('m', '-.')
)
