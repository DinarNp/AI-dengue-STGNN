import json
import os
from flask import session, request, redirect, Blueprint

_translations = {}
_TRANSLATIONS_DIR = os.path.join(os.path.dirname(__file__), 'translations')
_SUPPORTED_LANGS = ('id', 'en')
_DEFAULT_LANG = 'id'


def _load(lang):
    if lang not in _translations:
        path = os.path.join(_TRANSLATIONS_DIR, f'{lang}.json')
        with open(path, encoding='utf-8') as f:
            _translations[lang] = json.load(f)
    return _translations[lang]


def get_lang():
    lang = session.get('lang', _DEFAULT_LANG)
    return lang if lang in _SUPPORTED_LANGS else _DEFAULT_LANG


def make_t(lang):
    strings = _load(lang)
    fallback = _load(_DEFAULT_LANG)

    def t(key, **kwargs):
        text = strings.get(key) or fallback.get(key) or key
        if kwargs:
            try:
                text = text.format(**kwargs)
            except (KeyError, ValueError):
                pass
        return text

    return t


lang_bp = Blueprint('lang', __name__)


@lang_bp.route('/lang/<code>')
def switch(code):
    if code in _SUPPORTED_LANGS:
        session['lang'] = code
    ref = request.referrer or '/'
    return redirect(ref)


def register_i18n(app):
    app.register_blueprint(lang_bp)

    @app.context_processor
    def inject_i18n():
        lang = get_lang()
        return {'t': make_t(lang), 'lang': lang}
