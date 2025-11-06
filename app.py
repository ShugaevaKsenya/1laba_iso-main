import os
import random

from flask import Flask, render_template, request, redirect, url_for, session

import matplotlib

matplotlib.use('Agg')

from web_core import run_simulation, build_default_inputs, parse_form, U_LABELS, get_u_variable_for_equation


app = Flask(__name__, template_folder='templates', static_folder='static')
app.secret_key = 'change-me-in-production'

@app.template_filter('subscript')
def subscript_filter(s):
    """Convert number to subscript unicode characters"""
    subscript_digits = str.maketrans('0123456789', '₀₁₂₃₄₅₆₇₈₉')
    return str(s).translate(subscript_digits)

@app.route('/', methods=['GET', 'POST'])
def index():
    try:
        defaults = session.get('defaults')
        if not defaults:
            defaults = build_default_inputs()
            session['defaults'] = defaults

        if request.method == 'POST':
            u, faks, equations, restrictions = parse_form(request.form)
            session['defaults'] = {
                'u': u,
                'faks': faks,
                'equations': equations,
                'u_restrictions': restrictions,
            }
            outputs = run_simulation(u, faks, equations, restrictions)
            return render_template('index.html', **outputs, ran=True, error=None, defaults=session['defaults'], u_labels=U_LABELS, values={
                'u': u, 'faks': faks, 'equations': equations, 'u_restrictions': restrictions
            }, get_u_variable_for_equation=get_u_variable_for_equation)

        if request.args.get('run') == '1':
            defaults = build_default_inputs()
            session['defaults'] = defaults
            u, faks, equations, restrictions = defaults['u'], defaults['faks'], defaults['equations'], defaults['u_restrictions']
            outputs = run_simulation(u, faks, equations, restrictions)
            return render_template('index.html', **outputs, ran=True, error=None, defaults=defaults, u_labels=U_LABELS, values={
                'u': u, 'faks': faks, 'equations': equations, 'u_restrictions': restrictions
            }, get_u_variable_for_equation=get_u_variable_for_equation)

        return render_template('index.html', ran=False, error=None, defaults=defaults, u_labels=U_LABELS, get_u_variable_for_equation=get_u_variable_for_equation)
    except Exception as exc:
        return render_template('index.html', ran=False, error=str(exc))


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)


